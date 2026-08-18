# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate Attention Grill static SmoothQK anchors for a QAD student bundle.

The student is loaded from a complete ModelOpt-aware Diffusers bundle, so GEMM
fake quantization remains active while post-QK-norm, post-RoPE Q/K statistics
are collected. Calibration follows either Qwen-Image-Flash's native four-step
path without CFG or regular Qwen-Image's native 50-step true-CFG path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import attention_grill
import torch
import torch.nn.functional as F
from attention_grill._registry import register_kernel
from attention_grill.integrations.diffusers import install as install_diffusers_backends
from datasets import load_dataset
from diffusers import QwenImagePipeline
from diffusers.models.attention_dispatch import attention_backend
from safetensors.torch import save_file

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer
from qad.artifacts import (
    _find_quantized_transformer_blocks,
    _load_supported_static_anchor_recipe,
    _validate_regular_bundle,
    _validate_svdquant_bundle,
)

TARGET_MODULES = tuple(f"transformer_blocks.{index}.attn" for index in range(2, 58))
IGNORED_MODULES = (
    "transformer_blocks.0.attn",
    "transformer_blocks.1.attn",
    "transformer_blocks.58.attn",
    "transformer_blocks.59.attn",
)
FLASH_TIMESTEPS = torch.tensor([1000.0, 900.0, 750.0, 500.0], dtype=torch.float32)
FLASH_SIGMAS = torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0], dtype=torch.float32)
EXPECTED_HEAD_SHAPE = (24, 128)
SVDQUANT_ADAPTER = "modelopt_svdquant"
INFERENCE_PROFILES = {
    "qwen_image_flash": {
        "num_inference_steps": 4,
        "true_cfg_scale": 1.0,
        "negative_prompt": None,
        "calls_per_prompt": 4,
    },
    "qwen_image": {
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "calls_per_prompt": 100,
    },
}


class ModuleAccumulator:
    """Prompt-balanced FP32 streaming state for one attention module."""

    def __init__(self, module_name: str, expected_calls_per_prompt: int) -> None:
        self.module_name = module_name
        self.expected_calls_per_prompt = expected_calls_per_prompt
        self.current_q_sum: torch.Tensor | None = None
        self.current_k_sum: torch.Tensor | None = None
        self.prompt_q_mean_sum: torch.Tensor | None = None
        self.prompt_k_mean_sum: torch.Tensor | None = None
        self.current_calls = 0
        self.current_tokens = 0
        self.prompt_count = 0
        self.total_calls = 0
        self.total_tokens = 0
        self.shape_histogram: dict[str, int] = {}

    def observe(self, query: torch.Tensor, key: torch.Tensor) -> None:
        for label, tensor in (("query", query), ("key", key)):
            if tensor.ndim != 4 or tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda":
                raise RuntimeError(
                    f"{self.module_name} {label} contract mismatch: "
                    f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
                )
            if tuple(tensor.shape[2:]) != EXPECTED_HEAD_SHAPE:
                raise RuntimeError(
                    f"{self.module_name} {label} expected [...,24,128], found {tuple(tensor.shape)}"
                )
        if query.shape != key.shape:
            raise RuntimeError(
                f"{self.module_name} Q/K shapes differ: {tuple(query.shape)} != {tuple(key.shape)}"
            )

        if self.current_q_sum is None:
            self.current_q_sum = torch.zeros(EXPECTED_HEAD_SHAPE, device=query.device)
            self.current_k_sum = torch.zeros_like(self.current_q_sum)
            self.prompt_q_mean_sum = torch.zeros_like(self.current_q_sum)
            self.prompt_k_mean_sum = torch.zeros_like(self.current_q_sum)

        assert self.current_k_sum is not None
        self.current_q_sum.add_(query.sum(dim=(0, 1), dtype=torch.float32))
        self.current_k_sum.add_(key.sum(dim=(0, 1), dtype=torch.float32))
        tokens = int(query.shape[0]) * int(query.shape[1])
        self.current_calls += 1
        self.current_tokens += tokens
        signature = "x".join(str(value) for value in query.shape)
        self.shape_histogram[signature] = self.shape_histogram.get(signature, 0) + 1

    def finish_prompt(self) -> dict[str, int]:
        if self.current_q_sum is None or self.current_k_sum is None:
            raise RuntimeError(f"{self.module_name} received no Q/K tensors")
        if self.current_calls != self.expected_calls_per_prompt:
            raise RuntimeError(
                f"{self.module_name} saw {self.current_calls} calls; "
                f"expected {self.expected_calls_per_prompt} inference forwards"
            )
        if self.current_tokens <= 0:
            raise RuntimeError(f"{self.module_name} observed zero tokens")
        assert self.prompt_q_mean_sum is not None and self.prompt_k_mean_sum is not None
        self.prompt_q_mean_sum.add_(self.current_q_sum, alpha=1.0 / self.current_tokens)
        self.prompt_k_mean_sum.add_(self.current_k_sum, alpha=1.0 / self.current_tokens)

        record = {"calls": self.current_calls, "tokens": self.current_tokens}
        self.prompt_count += 1
        self.total_calls += self.current_calls
        self.total_tokens += self.current_tokens
        self.current_q_sum.zero_()
        self.current_k_sum.zero_()
        self.current_calls = 0
        self.current_tokens = 0
        return record

    def anchors(self, expected_prompts: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prompt_count != expected_prompts:
            raise RuntimeError(
                f"{self.module_name} has {self.prompt_count} prompts; expected {expected_prompts}"
            )
        assert self.prompt_q_mean_sum is not None and self.prompt_k_mean_sum is not None
        return (
            (self.prompt_q_mean_sum / expected_prompts).float().cpu().contiguous(),
            (self.prompt_k_mean_sum / expected_prompts).float().cpu().contiguous(),
        )


class CalibrationCollector:
    """Route each target attention module through native SDPA while collecting Q/K."""

    def __init__(self, expected_calls_per_prompt: int) -> None:
        self.accumulators = OrderedDict(
            (module_name, ModuleAccumulator(module_name, expected_calls_per_prompt))
            for module_name in TARGET_MODULES
        )
        self.active_prompt: str | None = None
        self.prompt_records: OrderedDict[str, dict[str, int]] = OrderedDict()

    def backend(self, module_name: str):
        accumulator = self.accumulators[module_name]

        def collect_and_native_sdpa(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: float | None = None,
        ) -> torch.Tensor:
            if self.active_prompt is None:
                raise RuntimeError(f"collector ran outside an active prompt: {module_name}")
            if attn_mask is not None or dropout_p != 0.0 or is_causal:
                raise NotImplementedError(
                    "Attention Grill calibration requires dense, noncausal attention "
                    "without a mask or dropout"
                )
            if value.shape != key.shape or value.dtype != query.dtype:
                raise RuntimeError(f"{module_name} Q/K/V contract mismatch")
            accumulator.observe(query, key)
            output = F.scaled_dot_product_attention(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=False,
            ).permute(0, 2, 1, 3)
            if output.shape != query.shape or output.dtype != query.dtype:
                raise RuntimeError(
                    f"{module_name} native SDPA output contract mismatch: "
                    f"shape={tuple(output.shape)} dtype={output.dtype} device={output.device}"
                )
            if output.device != query.device:
                raise RuntimeError(f"{module_name} native SDPA moved devices")
            return output

        collect_and_native_sdpa.__name__ = "collect_" + module_name.replace(".", "_")
        return collect_and_native_sdpa

    def begin_prompt(self, prompt_id: str) -> None:
        if self.active_prompt is not None or prompt_id in self.prompt_records:
            raise RuntimeError(f"invalid prompt transition: {prompt_id}")
        self.active_prompt = prompt_id

    def finish_prompt(self, prompt_id: str) -> dict[str, int]:
        if self.active_prompt != prompt_id:
            raise RuntimeError(f"active prompt mismatch: {self.active_prompt} != {prompt_id}")
        module_records = [accumulator.finish_prompt() for accumulator in self.accumulators.values()]
        first = module_records[0]
        if any(record != first for record in module_records[1:]):
            raise RuntimeError(f"per-module accounting differs for {prompt_id}")
        self.prompt_records[prompt_id] = first
        self.active_prompt = None
        return first

    def finalize(
        self, expected_prompts: int
    ) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, dict[str, Any]]]:
        if self.active_prompt is not None:
            raise RuntimeError("cannot finalize while a prompt is active")
        if len(self.prompt_records) != expected_prompts:
            raise RuntimeError(
                f"collector has {len(self.prompt_records)} prompts; expected {expected_prompts}"
            )
        tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        module_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for module_name, accumulator in self.accumulators.items():
            q_anchor, k_anchor = accumulator.anchors(expected_prompts)
            if not bool(torch.isfinite(q_anchor).all() and torch.isfinite(k_anchor).all()):
                raise FloatingPointError(f"nonfinite anchors for {module_name}")
            tensors[f"{module_name}.q_anchor"] = q_anchor
            tensors[f"{module_name}.k_anchor"] = k_anchor
            module_records[module_name] = {
                "q_shape": list(q_anchor.shape),
                "k_shape": list(k_anchor.shape),
                "prompts": accumulator.prompt_count,
                "calls": accumulator.total_calls,
                "tokens": accumulator.total_tokens,
                "shape_histogram": dict(sorted(accumulator.shape_histogram.items())),
            }
        return tensors, module_records


def _processor_snapshot(transformer: torch.nn.Module) -> OrderedDict[str, str | None]:
    snapshot: OrderedDict[str, str | None] = OrderedDict()
    for name, module in transformer.named_modules():
        processor = getattr(module, "processor", None)
        if processor is None or not hasattr(processor, "_attention_backend"):
            continue
        backend = processor._attention_backend
        snapshot[name] = None if backend is None else str(getattr(backend, "value", backend))
    return snapshot


def _install_collectors(transformer: torch.nn.Module, collector: CalibrationCollector) -> None:
    module_index = dict(transformer.named_modules())
    if tuple(name for name in module_index if name in TARGET_MODULES) != TARGET_MODULES:
        raise RuntimeError("Qwen-Image target attention module set/order differs from blocks 2-57")

    type_names: OrderedDict[str, str] = OrderedDict()
    for module_name in TARGET_MODULES:
        block = int(module_name.split(".")[1])
        type_name = f"qad-static-smooth-qk-calibration-block-{block}"
        register_kernel(type_name)(collector.backend(module_name))
        type_names[module_name] = type_name
    installed = install_diffusers_backends()

    for module_name, type_name in type_names.items():
        if installed.get(type_name) != f"grill-{type_name}":
            raise RuntimeError(f"failed to install collector backend for {module_name}")
        report = attention_grill.replace(module_index[module_name], type=type_name)
        if report.replaced != [""] or report.ignored or report.recipe is not None:
            raise RuntimeError(
                f"unexpected collector replacement report for {module_name}: {report}"
            )


def _load_prompts(args: argparse.Namespace) -> list[str]:
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    if args.dataset_column not in dataset.column_names:
        raise ValueError(
            f"dataset {args.dataset!r} has no column {args.dataset_column!r}: "
            f"{dataset.column_names}"
        )
    prompts = [str(value).strip() for value in dataset[args.dataset_column][: args.num_prompts]]
    if len(prompts) != args.num_prompts or any(not prompt for prompt in prompts):
        raise ValueError(f"could not load {args.num_prompts} nonempty calibration prompts")
    return prompts


def _validate_student(pipe: QwenImagePipeline, mode: str, inference_profile: str) -> None:
    transformer = pipe.transformer
    if not mto.ModeloptStateManager.is_converted(transformer):
        raise RuntimeError("student transformer did not restore ModelOpt state")
    if mode == "nvfp4":
        _validate_regular_bundle(transformer)
    else:
        _validate_svdquant_bundle(transformer)
        svdquant_adapters = [
            (name, module)
            for name, module in transformer.named_modules()
            if SVDQUANT_ADAPTER in getattr(module, "lora_A", {})
        ]
        inactive_adapters = [
            name
            for name, module in svdquant_adapters
            if list(module.active_adapters) != [SVDQUANT_ADAPTER]
            or bool(module.disable_adapters)
            or bool(module.merged)
        ]
        if not svdquant_adapters or inactive_adapters:
            raise RuntimeError(
                "SVDQuant PEFT adapters are missing, inactive, disabled, or merged: "
                + ", ".join(inactive_adapters[:5])
            )
        logging.info("Validated %d active, unmerged SVDQuant adapters", len(svdquant_adapters))

    active_quantizers = [
        (name, module)
        for name, module in transformer.named_modules()
        if isinstance(module, TensorQuantizer) and module.is_enabled
    ]
    inactive_fake_quant = [
        name
        for name, quantizer in active_quantizers
        if not quantizer._if_quant or not quantizer.fake_quant or quantizer._if_calib
    ]
    if not active_quantizers or inactive_fake_quant:
        raise RuntimeError(
            "enabled ModelOpt quantizers are not executing fake quantization: "
            + ", ".join(inactive_fake_quant[:5])
        )
    slot_counts = {
        slot: sum(f"{slot}_quantizer" in name.split(".") for name, _ in active_quantizers)
        for slot in ("weight", "input")
    }
    logging.info(
        "Validated active fake quantization: total=%d weight=%d input=%d",
        len(active_quantizers),
        slot_counts["weight"],
        slot_counts["input"],
    )
    quantized_blocks = _find_quantized_transformer_blocks(transformer)
    if quantized_blocks != tuple(range(2, 58)):
        raise RuntimeError(
            f"expected ModelOpt-quantized transformer blocks 2-57, found {quantized_blocks}"
        )
    if (
        len(transformer.transformer_blocks) != 60
        or int(transformer.config.num_attention_heads) != 24
        or int(transformer.config.attention_head_dim) != 128
        or bool(transformer.config.guidance_embeds)
    ):
        raise RuntimeError("student is not the expected 60-block Qwen-Image architecture")

    scheduler = pipe.scheduler.config
    if inference_profile == "qwen_image_flash":
        expected_scheduler = {
            "num_train_timesteps": 1000,
            "shift": 3.0,
            "shift_terminal": None,
            "use_dynamic_shifting": False,
            "invert_sigmas": False,
        }
    else:
        expected_scheduler = {
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": 0.02,
            "use_dynamic_shifting": True,
            "invert_sigmas": False,
        }
    actual_scheduler = {name: scheduler.get(name) for name in expected_scheduler}
    if actual_scheduler != expected_scheduler:
        raise RuntimeError(
            f"student scheduler does not match {inference_profile}: "
            f"{actual_scheduler} != {expected_scheduler}"
        )


def _run_inference(
    pipe: QwenImagePipeline,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    inference_profile: str,
) -> tuple[torch.Tensor, list[float], float]:
    observed_timesteps: list[float] = []

    def capture_timestep(_pipe, _step, timestep, callback_kwargs):
        observed_timesteps.append(float(timestep.detach().float().cpu().item()))
        return callback_kwargs

    generator = torch.Generator(device="cuda").manual_seed(seed)
    profile = INFERENCE_PROFILES[inference_profile]
    inference_kwargs = {
        "prompt": prompt,
        "true_cfg_scale": profile["true_cfg_scale"],
        "height": height,
        "width": width,
        "num_inference_steps": profile["num_inference_steps"],
        "generator": generator,
        "output_type": "latent",
        "callback_on_step_end": capture_timestep,
    }
    if profile["negative_prompt"] is not None:
        inference_kwargs["negative_prompt"] = profile["negative_prompt"]
    torch.cuda.synchronize()
    started = time.perf_counter()
    with attention_backend("native"), torch.no_grad():
        output = pipe(**inference_kwargs)
    torch.cuda.synchronize()
    return output.images, observed_timesteps, time.perf_counter() - started


def _validate_schedule(
    pipe: QwenImagePipeline,
    observed: list[float],
    inference_profile: str,
) -> None:
    actual_observed = torch.tensor(observed, dtype=torch.float32)
    actual_timesteps = pipe.scheduler.timesteps.detach().float().cpu()
    actual_sigmas = pipe.scheduler.sigmas.detach().float().cpu()
    if not torch.allclose(actual_observed, actual_timesteps, rtol=0.0, atol=1e-4):
        raise RuntimeError(
            f"executed {inference_profile} timesteps differ from the scheduler: "
            f"{actual_observed.tolist()} != {actual_timesteps.tolist()}"
        )
    if inference_profile == "qwen_image_flash":
        if not torch.allclose(actual_timesteps, FLASH_TIMESTEPS, rtol=0.0, atol=1e-4):
            raise RuntimeError(f"scheduler Flash timesteps differ: {actual_timesteps.tolist()}")
        if not torch.allclose(actual_sigmas, FLASH_SIGMAS, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"scheduler Flash sigmas differ: {actual_sigmas.tolist()}")
        return

    if len(actual_timesteps) != 50 or len(actual_sigmas) != 51:
        raise RuntimeError(
            "regular Qwen-Image inference did not execute the native 50-step schedule: "
            f"timesteps={len(actual_timesteps)} sigmas={len(actual_sigmas)}"
        )
    if not bool(torch.isfinite(actual_timesteps).all() and torch.isfinite(actual_sigmas).all()):
        raise RuntimeError("regular Qwen-Image scheduler produced nonfinite values")
    if not bool(
        torch.all(actual_timesteps[:-1] > actual_timesteps[1:])
        and torch.all(actual_sigmas[:-1] > actual_sigmas[1:])
    ):
        raise RuntimeError("regular Qwen-Image scheduler trajectory is not strictly decreasing")
    if not torch.allclose(actual_timesteps, actual_sigmas[:-1] * 1000.0, rtol=0.0, atol=1e-3):
        raise RuntimeError("regular Qwen-Image scheduler timestep/sigma relation differs")
    if abs(float(actual_sigmas[-1])) > 1e-7:
        raise RuntimeError("regular Qwen-Image scheduler terminal sigma is not zero")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_artifact(
    *,
    output_dir: Path,
    tensors: OrderedDict[str, torch.Tensor],
    module_records: OrderedDict[str, dict[str, Any]],
    args: argparse.Namespace,
    inference_record: dict[str, Any],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=False)
    anchors_path = output_dir / "anchors.safetensors"
    temporary_anchors = output_dir / "anchors.safetensors.tmp"
    save_file(
        dict(tensors),
        str(temporary_anchors),
        metadata={
            "format": "attention-grill-static-smooth-qk",
            "activation_point": "post_qk_norm_post_rope",
            "aggregation": "sample_balanced_global_token_mean",
        },
    )
    os.replace(temporary_anchors, anchors_path)
    anchors_sha256 = _sha256(anchors_path)

    modules = OrderedDict(
        (
            module_name,
            {
                "q_anchor": {"shape": record["q_shape"]},
                "k_anchor": {"shape": record["k_shape"]},
            },
        )
        for module_name, record in module_records.items()
    )
    manifest = {
        "schema_version": 1,
        "format": "attention-grill-static-smooth-qk",
        "activation_point": "post_qk_norm_post_rope",
        "granularity": "per_layer_head_channel",
        "aggregation": "sample_balanced_global_token_mean",
        "anchors_file": "anchors.safetensors",
        "anchors_sha256": anchors_sha256,
        "modules": modules,
        "producer": "FastGen QAD Attention Grill calibration",
        "student_model": str(args.student_model.resolve()),
        "student_mode": args.student_mode,
        "prompt_source": {
            "dataset": args.dataset,
            "split": args.dataset_split,
            "column": args.dataset_column,
            "count": args.num_prompts,
        },
        "inference": inference_record,
        "module_statistics": module_records,
    }
    temporary_manifest = output_dir / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, output_dir / "manifest.json")
    return anchors_sha256


def _validate_saved_artifact(
    pipe: QwenImagePipeline,
    output_dir: Path,
    recipe: dict[str, Any],
    anchors_sha256: str,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    inference_profile: str,
) -> None:
    artifact = attention_grill.load_calibration(
        output_dir,
        expected_modules=TARGET_MODULES,
        allow_extra_modules=False,
    )
    if artifact.anchors_sha256 != anchors_sha256:
        raise RuntimeError("loaded Attention Grill artifact hash differs from the saved artifact")

    report = attention_grill.replace(
        pipe.transformer,
        recipe=recipe,
        calibration=output_dir,
        ignore=list(IGNORED_MODULES),
    )
    try:
        if tuple(report.replaced) != TARGET_MODULES or tuple(report.ignored) != IGNORED_MODULES:
            raise RuntimeError(f"static Attention Grill replacement scope differs: {report}")
        if report.calibration is None or report.calibration["anchors_sha256"] != anchors_sha256:
            raise RuntimeError(f"static Attention Grill calibration report differs: {report}")
        latent, observed_timesteps, seconds = _run_inference(
            pipe, prompt, seed, height, width, inference_profile
        )
        _validate_schedule(pipe, observed_timesteps, inference_profile)
        if not bool(torch.isfinite(latent).all().item()):
            raise FloatingPointError("static Attention Grill smoke produced nonfinite latents")
        logging.info(
            "Static Attention Grill %s kernel smoke passed: %.2fs latent=%s",
            inference_profile,
            seconds,
            tuple(latent.shape),
        )
    finally:
        restored = attention_grill.restore(pipe.transformer)
        if tuple(restored) != TARGET_MODULES:
            raise RuntimeError(f"static Attention Grill restore scope differs: {restored}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--student-mode", choices=("nvfp4", "nvfp4_svdquant"), required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--dataset", default="Gustavosta/Stable-Diffusion-Prompts")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-column", default="Prompt")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inference-profile",
        choices=tuple(INFERENCE_PROFILES),
        default="qwen_image_flash",
    )
    args = parser.parse_args()
    if args.num_prompts <= 0:
        parser.error("--num-prompts must be positive")
    if args.height != 1024 or args.width != 1024:
        parser.error("Qwen-Image Attention Grill calibration is fixed at 1024x1024")
    if not args.student_model.is_dir():
        parser.error(f"student bundle does not exist: {args.student_model}")
    if not args.recipe.is_file():
        parser.error(f"Attention Grill recipe does not exist: {args.recipe}")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite output: {args.output_dir}")
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = _parse_args()
    recipe = _load_supported_static_anchor_recipe(attention_grill, args.recipe)
    mto.enable_huggingface_checkpointing()
    logging.info("Loading ModelOpt student bundle: %s", args.student_model)
    pipe = QwenImagePipeline.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval()
    _validate_student(pipe, args.student_mode, args.inference_profile)
    logging.info("ModelOpt quantizer summary follows")
    mtq.print_quant_summary(pipe.transformer)

    prompts = _load_prompts(args)
    logging.info(
        "Calibrating %s with %d prompts using native %s inference",
        args.student_mode,
        len(prompts),
        args.inference_profile,
    )
    baseline_processors = _processor_snapshot(pipe.transformer)
    if tuple(baseline_processors) != tuple(
        f"transformer_blocks.{index}.attn" for index in range(60)
    ):
        raise RuntimeError("student attention processor module set/order differs")

    native_first, native_timesteps, native_seconds = _run_inference(
        pipe, prompts[0], args.seed, args.height, args.width, args.inference_profile
    )
    _validate_schedule(pipe, native_timesteps, args.inference_profile)
    logging.info(
        "Native parity reference complete: %.2fs timesteps=%s sigmas=%s",
        native_seconds,
        native_timesteps,
        pipe.scheduler.sigmas.detach().float().cpu().tolist(),
    )

    profile = INFERENCE_PROFILES[args.inference_profile]
    collector = CalibrationCollector(int(profile["calls_per_prompt"]))
    _install_collectors(pipe.transformer, collector)
    try:
        for index, prompt in enumerate(prompts):
            prompt_id = f"prompt_{index:03d}"
            collector.begin_prompt(prompt_id)
            latent, observed_timesteps, seconds = _run_inference(
                pipe,
                prompt,
                args.seed + index,
                args.height,
                args.width,
                args.inference_profile,
            )
            _validate_schedule(pipe, observed_timesteps, args.inference_profile)
            accounting = collector.finish_prompt(prompt_id)
            if index == 0:
                if not torch.equal(native_first, latent):
                    max_error = (native_first.float() - latent.float()).abs().max().item()
                    raise RuntimeError(
                        "collector changed the native first-prompt latent trajectory: "
                        f"max_abs_error={max_error}"
                    )
                logging.info("Collector/native first-prompt latent parity: bit-exact")
            logging.info(
                "Calibration prompt %d/%d: %.2fs calls/module=%d tokens/module=%d text=%r",
                index + 1,
                len(prompts),
                seconds,
                accounting["calls"],
                accounting["tokens"],
                prompt[:100],
            )
    finally:
        restored = attention_grill.restore(pipe.transformer)
        if tuple(restored) != TARGET_MODULES:
            raise RuntimeError(f"collector restore scope differs: {restored}")
    if _processor_snapshot(pipe.transformer) != baseline_processors:
        raise RuntimeError("native attention processors were not restored after calibration")

    tensors, module_records = collector.finalize(len(prompts))
    for module_name, record in module_records.items():
        logging.info(
            "Calibrated Attention Grill module %s: q=%s k=%s prompts=%d calls=%d "
            "tokens=%d shapes=%s",
            module_name,
            record["q_shape"],
            record["k_shape"],
            record["prompts"],
            record["calls"],
            record["tokens"],
            record["shape_histogram"],
        )
    inference_record = {
        "pipeline": "QwenImagePipeline",
        "profile": args.inference_profile,
        "height": args.height,
        "width": args.width,
        "true_cfg_scale": profile["true_cfg_scale"],
        "negative_prompt_provided": profile["negative_prompt"] is not None,
        "negative_prompt": profile["negative_prompt"],
        "num_inference_steps": profile["num_inference_steps"],
        "calls_per_prompt": profile["calls_per_prompt"],
        "timesteps": native_timesteps,
        "sigmas": pipe.scheduler.sigmas.detach().float().cpu().tolist(),
        "base_seed": args.seed,
    }
    anchors_sha256 = _save_artifact(
        output_dir=args.output_dir,
        tensors=tensors,
        module_records=module_records,
        args=args,
        inference_record=inference_record,
    )
    _validate_saved_artifact(
        pipe,
        args.output_dir,
        recipe,
        anchors_sha256,
        prompts[0],
        args.seed,
        args.height,
        args.width,
        args.inference_profile,
    )
    logging.info(
        "Attention Grill calibration complete: output=%s modules=%d tensors=%d sha256=%s",
        args.output_dir,
        len(module_records),
        len(tensors),
        anchors_sha256,
    )


if __name__ == "__main__":
    main()
