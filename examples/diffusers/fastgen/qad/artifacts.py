# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Student artifact restore and train-scope handling for the QAD example.

The generic AutoModel diffusion builder intentionally owns FSDP and optimizer
construction. QAD only needs two narrowly-scoped hooks around that builder:

* validate the ModelOpt topology restored by a native Diffusers training bundle
  before FSDP;
* after FSDP, optionally freeze everything except ModelOpt SVDQuant's HF PEFT A/B
  and optional magnitude parameters, then retarget AdamW to the live sharded parameters.

Quantization itself is never calibrated here. The complete topology, weights,
and quantizer buffers must already be present in the student bundle.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import logging
import math
import os
import re
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import torch

import modelopt.torch.opt as mto
from modelopt.torch.quantization.nn import TensorQuantizer

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from torch import nn

_SVDQUANT_PARAMETER_RE = re.compile(
    r"(?:^|\.)(?:lora_[AB]\.modelopt_svdquant\.weight|svdquant_magnitude_delta)$"
)
_SUPPORTED_STUDENT_MODES = frozenset({"nvfp4", "nvfp4_svdquant"})
_SUPPORTED_TRAIN_SCOPES = frozenset({"all", "lora_only"})
_QWEN_IMAGE_FLASH_TIMESTEPS = (1000.0, 900.0, 750.0, 500.0)
_QWEN_IMAGE_FLASH_SIGMAS = (1.0, 0.9, 0.75, 0.5, 0.0)
_SUPPORTED_ATTENTION_GRILL_PROFILES = frozenset({"qwen_image", "qwen_image_flash"})


@dataclasses.dataclass(frozen=True)
class StudentSettings:
    """Resolved ``qad.student`` configuration."""

    mode: str
    model_name_or_path: str
    train_scope: str = "all"

    def validate(self) -> None:
        if self.mode not in _SUPPORTED_STUDENT_MODES:
            raise ValueError(
                f"qad.student.mode must be one of {sorted(_SUPPORTED_STUDENT_MODES)}, "
                f"got {self.mode!r}."
            )
        if not self.model_name_or_path:
            raise ValueError("model.pretrained_model_name_or_path is required for the student.")
        if self.train_scope not in _SUPPORTED_TRAIN_SCOPES:
            raise ValueError(
                f"qad.student.train_scope must be 'all' or 'lora_only', got {self.train_scope!r}."
            )

        if self.mode == "nvfp4" and self.train_scope != "all":
            raise ValueError("Regular NVFP4 supports only qad.student.train_scope=all.")


@dataclasses.dataclass(frozen=True)
class AttentionGrillSettings:
    """Resolved optional calibrated-attention configuration for the student."""

    enabled: bool = False
    recipe_path: str | None = None
    calibration_path: str | None = None
    ignore: tuple[str, ...] = ()
    expected_replaced: int = 0
    expected_ignored: int = 0
    inference_profile: str | None = None

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.recipe_path:
            raise ValueError(
                "qad.attention_grill.recipe_path is required when calibrated attention is enabled."
            )
        if not self.calibration_path:
            raise ValueError(
                "qad.attention_grill.calibration_path is required when calibrated attention is enabled."
            )
        if not os.path.isfile(self.recipe_path):
            raise ValueError(f"Attention Grill recipe not found: {self.recipe_path}")
        if not os.path.isdir(self.calibration_path):
            raise ValueError(
                f"Attention Grill calibration directory not found: {self.calibration_path}"
            )
        if not self.ignore:
            raise ValueError(
                "qad.attention_grill.ignore must explicitly identify the uncalibrated attention blocks."
            )
        if self.expected_replaced <= 0 or self.expected_ignored <= 0:
            raise ValueError(
                "qad.attention_grill expected_replaced and expected_ignored must be positive."
            )
        if self.inference_profile not in _SUPPORTED_ATTENTION_GRILL_PROFILES:
            raise ValueError(
                "Calibrated Attention Grill requires qad.timestep.schedule=qwen_image "
                "or qwen_image_flash."
            )


@dataclasses.dataclass
class StudentBuildState:
    """Information captured while AutoModel builds the student."""

    parallel_scheme: dict[str, dict[str, Any]] | None = None
    quantizer_count: int = 0
    quantized_block_indices: tuple[int, ...] = ()
    svdquant_parameter_names: tuple[str, ...] = ()
    attention_grill_summary: dict[str, Any] | None = None


def _is_block16_nvfp4(quantizer: TensorQuantizer) -> bool:
    block_sizes = quantizer.block_sizes or {}
    return bool(
        (quantizer.is_nvfp4_dynamic or quantizer.is_nvfp4_static) and block_sizes.get(-1) == 16
    )


def _enabled_quantizer_leaves(module: Any) -> tuple[TensorQuantizer, ...]:
    if module is None or not hasattr(module, "modules"):
        return ()
    return tuple(
        child
        for child in module.modules()
        if isinstance(child, TensorQuantizer) and child.is_enabled
    )


def _validate_nvfp4_quantizers(
    model: nn.Module,
    *,
    artifact_name: str,
    required_targets: tuple[str, ...] = (),
) -> None:
    """Reject non-NVFP4 artifacts before FSDP obscures their module topology."""
    enabled_by_slot: dict[str, list[tuple[str, TensorQuantizer]]] = {
        "weight": [],
        "input": [],
    }
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or not module.is_enabled:
            continue
        path_parts = name.split(".")
        for slot in enabled_by_slot:
            if f"{slot}_quantizer" in path_parts:
                enabled_by_slot[slot].append((name, module))

    missing_slots = [slot for slot, entries in enabled_by_slot.items() if not entries]
    if missing_slots:
        raise RuntimeError(
            f"{artifact_name} is not an NVFP4 W4A4 training artifact: no enabled "
            + "/".join(missing_slots)
            + " quantizers were found."
        )

    incompatible = [
        name
        for entries in enabled_by_slot.values()
        for name, quantizer in entries
        if not _is_block16_nvfp4(quantizer)
    ]
    if incompatible:
        raise RuntimeError(
            f"{artifact_name} contains enabled GEMM quantizers that are not block-16 NVFP4 "
            "(E2M1 values with E4M3 scales): " + ", ".join(incompatible[:5])
        )

    for target_name in required_targets:
        target = model.get_submodule(target_name)
        get_base_layer = getattr(target, "get_base_layer", None)
        base_layer = get_base_layer() if callable(get_base_layer) else target
        for slot in ("weight", "input"):
            leaves = _enabled_quantizer_leaves(getattr(base_layer, f"{slot}_quantizer", None))
            if not leaves or any(not _is_block16_nvfp4(quantizer) for quantizer in leaves):
                raise RuntimeError(
                    f"SVDQuant target {target_name!r} does not have an enabled block-16 "
                    f"NVFP4 {slot}_quantizer."
                )

    logging.info(
        "[QAD] validated block-16 NVFP4 W4A4 quantizers before FSDP: %d weight, %d input",
        len(enabled_by_slot["weight"]),
        len(enabled_by_slot["input"]),
    )


def _find_quantized_transformer_blocks(model: nn.Module) -> tuple[int, ...]:
    """Return blocks containing at least one enabled weight quantizer."""
    blocks = getattr(model, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("The QAD student does not expose transformer_blocks.")

    indices = tuple(
        index
        for index, block in enumerate(blocks)
        if any(
            isinstance(module, TensorQuantizer)
            and module.is_enabled
            and "weight_quantizer" in name.split(".")
            for name, module in block.named_modules()
        )
    )
    if not indices:
        raise RuntimeError(
            "The QAD student has no transformer block with an enabled weight quantizer."
        )
    logging.info(
        "[QAD] discovered %d quantized transformer blocks before FSDP: %s",
        len(indices),
        ",".join(str(index) for index in indices),
    )
    return indices


def _load_supported_static_anchor_recipe(
    attention_grill: Any,
    recipe_source: str | os.PathLike[str] | Mapping[str, Any],
) -> dict[str, Any]:
    """Load any registered Attention Grill recipe that consumes static anchors."""
    recipe = attention_grill.load_recipe(recipe_source)
    kernel = recipe.get("kernel")
    if kernel not in attention_grill.available_types():
        raise ValueError(f"Attention Grill kernel {kernel!r} is not registered in this checkout.")
    if (recipe.get("smooth") or {}).get("mode") != "static_anchor":
        raise ValueError(
            "This path requires an Attention Grill recipe whose normalized "
            "smooth.mode is 'static_anchor'."
        )
    return recipe


def _validate_attention_grill_manifest(
    manifest: Mapping[str, Any],
    settings: AttentionGrillSettings,
    student: StudentSettings,
) -> dict[str, Any]:
    manifest = dict(manifest)

    artifact_mode = str(manifest.get("student_mode", "")).lower()
    if artifact_mode != student.mode:
        raise RuntimeError(
            "Attention Grill calibration/student mode mismatch: artifact has "
            f"{artifact_mode!r}, QAD requested {student.mode!r}."
        )

    artifact_student = manifest.get("student_model")
    if not artifact_student:
        raise RuntimeError(
            "Attention Grill calibration manifest does not record its student_model."
        )
    if os.path.realpath(str(artifact_student)) != os.path.realpath(student.model_name_or_path):
        raise RuntimeError(
            "Attention Grill calibration was produced for a different student bundle: "
            f"artifact={artifact_student!r}, QAD={student.model_name_or_path!r}."
        )

    inference = manifest.get("inference") or {}
    artifact_profile = inference.get("profile")
    # Calib64 artifacts created before regular Qwen-Image support predate the
    # explicit profile field and are all exact Flash four-step artifacts.
    if artifact_profile is None and int(inference.get("num_inference_steps", 0)) == len(
        _QWEN_IMAGE_FLASH_TIMESTEPS
    ):
        artifact_profile = "qwen_image_flash"
    if artifact_profile != settings.inference_profile:
        raise RuntimeError(
            "Attention Grill calibration/inference profile mismatch: artifact has "
            f"{artifact_profile!r}, QAD requested {settings.inference_profile!r}."
        )

    timesteps = tuple(float(value) for value in inference.get("timesteps", ()))
    sigmas = tuple(float(value) for value in inference.get("sigmas", ()))
    if artifact_profile == "qwen_image_flash":
        if (
            int(inference.get("num_inference_steps", 0)) != len(_QWEN_IMAGE_FLASH_TIMESTEPS)
            or timesteps != _QWEN_IMAGE_FLASH_TIMESTEPS
            or float(inference.get("true_cfg_scale", 0.0)) != 1.0
            or len(sigmas) != len(_QWEN_IMAGE_FLASH_SIGMAS)
            or any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(sigmas, _QWEN_IMAGE_FLASH_SIGMAS)
            )
        ):
            raise RuntimeError(
                "Qwen-Image-Flash Attention Grill calibration must use no-CFG and the "
                f"exact timesteps {_QWEN_IMAGE_FLASH_TIMESTEPS}; manifest has {timesteps}."
            )
    else:
        regular_contract_valid = (
            int(inference.get("num_inference_steps", 0)) == 50
            and float(inference.get("true_cfg_scale", 0.0)) == 4.0
            and bool(inference.get("negative_prompt_provided", False))
            and inference.get("negative_prompt") == " "
            and int(inference.get("calls_per_prompt", 0)) == 100
            and int(inference.get("height", 0)) == 1024
            and int(inference.get("width", 0)) == 1024
            and len(timesteps) == 50
            and len(sigmas) == 51
            and all(math.isfinite(value) for value in (*timesteps, *sigmas))
            and all(left > right for left, right in pairwise(timesteps))
            and all(left > right for left, right in pairwise(sigmas))
            and abs(sigmas[-1]) <= 1e-7
            and all(
                abs(timestep - sigma * 1000.0) <= 1e-3 for timestep, sigma in zip(timesteps, sigmas)
            )
        )
        if not regular_contract_valid:
            raise RuntimeError(
                "Regular Qwen-Image Attention Grill calibration must record the native "
                "1024x1024 50-step true-CFG inference trajectory."
            )

    modules = manifest.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise RuntimeError("Attention Grill calibration manifest has no module mapping.")
    return manifest


def _install_attention_grill(
    model: nn.Module,
    *,
    student: StudentSettings,
    settings: AttentionGrillSettings,
) -> dict[str, Any] | None:
    """Install one validated static-attention artifact before FSDP mutates the tree."""
    if not settings.enabled:
        return None

    try:
        import attention_grill
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "qad.attention_grill.enabled=true requires the optional attention-grill "
            "package. Install it or add its source directory to PYTHONPATH."
        ) from exc

    recipe = _load_supported_static_anchor_recipe(attention_grill, settings.recipe_path)
    artifact = attention_grill.load_calibration(settings.calibration_path)
    manifest = _validate_attention_grill_manifest(artifact.manifest, settings, student)
    calibrated_modules = artifact.module_names
    if len(calibrated_modules) != settings.expected_replaced:
        raise RuntimeError(
            "Attention Grill calibration module count does not match "
            f"qad.attention_grill.expected_replaced={settings.expected_replaced}: "
            f"found {len(calibrated_modules)}."
        )

    parameters_before = tuple(
        (name, id(parameter), parameter.requires_grad)
        for name, parameter in model.named_parameters()
    )
    report = attention_grill.replace(
        model,
        recipe=recipe,
        calibration=artifact.root,
        ignore=settings.ignore,
    )
    if report.recipe != recipe:
        raise RuntimeError(
            "Attention Grill replacement did not report the normalized recipe selected by QAD."
        )
    calibration = report.calibration
    if calibration is None:
        raise RuntimeError("Attention Grill replacement did not consume calibration anchors.")
    if calibration.get("anchors_sha256") != artifact.anchors_sha256:
        raise RuntimeError(
            "Attention Grill replacement reported a different calibration anchor hash."
        )
    if os.path.realpath(str(calibration.get("directory", ""))) != os.path.realpath(artifact.root):
        raise RuntimeError(
            "Attention Grill replacement reported a different calibration directory."
        )
    if len(report.replaced) != settings.expected_replaced:
        raise RuntimeError(
            f"Attention Grill replaced {len(report.replaced)} modules; expected "
            f"{settings.expected_replaced}."
        )
    if len(report.ignored) != settings.expected_ignored:
        raise RuntimeError(
            f"Attention Grill ignored {len(report.ignored)} modules; expected "
            f"{settings.expected_ignored}."
        )
    if set(report.replaced) != set(calibrated_modules):
        missing = sorted(set(calibrated_modules) - set(report.replaced))
        extra = sorted(set(report.replaced) - set(calibrated_modules))
        raise RuntimeError(
            "Attention Grill replacement does not exactly match the calibrated modules "
            f"(missing={missing[:5]}, extra={extra[:5]})."
        )

    parameters_after = tuple(
        (name, id(parameter), parameter.requires_grad)
        for name, parameter in model.named_parameters()
    )
    if parameters_after != parameters_before:
        raise RuntimeError(
            "Installing Attention Grill unexpectedly changed student parameter identity "
            "or trainability before FSDP."
        )

    anchor_backends = tuple(
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("._grill_static_smooth_qk_backend")
    )
    if len(anchor_backends) != settings.expected_replaced:
        raise RuntimeError(
            "Attention Grill did not bind one static anchor module per replaced attention: "
            f"found {len(anchor_backends)}, expected {settings.expected_replaced}."
        )
    for name, module in anchor_backends:
        for buffer_name in ("q_anchor", "k_anchor"):
            if buffer_name not in module._buffers:
                raise RuntimeError(f"{name} is missing frozen {buffer_name} buffer.")
            if buffer_name not in module._non_persistent_buffers_set:
                raise RuntimeError(
                    f"{name}.{buffer_name} must stay non-persistent and be restored "
                    "from the calibrated artifact on every QAD startup."
                )

    summary = {
        "kernel": report.type,
        "recipe_path": settings.recipe_path,
        "calibration_path": calibration.get("directory", settings.calibration_path),
        "anchors_sha256": calibration.get("anchors_sha256"),
        "replaced": tuple(report.replaced),
        "ignored": tuple(report.ignored),
        "prompt_count": int((manifest.get("prompt_source") or {}).get("count", 0)),
    }
    logging.info(
        "[QAD] installed calibrated Attention Grill before FSDP: kernel=%s "
        "replaced=%d ignored=%d prompts=%d anchors=%s",
        summary["kernel"],
        len(summary["replaced"]),
        len(summary["ignored"]),
        summary["prompt_count"],
        str(summary["anchors_sha256"])[:12],
    )
    return summary


def _modelopt_mode_states(model: nn.Module) -> dict[str, dict[str, Any]]:
    if not mto.ModeloptStateManager.is_converted(model):
        return {}
    return dict(mto.modelopt_state(model)["modelopt_state_dict"])


def _reject_non_training_modes(mode_states: dict[str, dict[str, Any]]) -> None:
    if "real_quantize" in mode_states:
        raise RuntimeError(
            "QAD cannot train a compressed real-quantized bundle. Recalibrate without "
            "quantize.py --compress and provide the resulting fake-quantized training bundle."
        )


def _validate_regular_bundle(model: nn.Module) -> int:
    mode_states = _modelopt_mode_states(model)
    if not mode_states:
        raise RuntimeError(
            "qad.student.mode=nvfp4 requires a ModelOpt-aware Diffusers training bundle. "
            "Calibrate it with quantize.py --output-bundle before starting QAD."
        )
    _reject_non_training_modes(mode_states)
    if "svdquant_calibrate" in mode_states:
        raise RuntimeError(
            "qad.student.mode=nvfp4 received an SVDQuant bundle; use mode=nvfp4_svdquant."
        )
    quantizers = [module for module in model.modules() if isinstance(module, TensorQuantizer)]
    if not quantizers:
        raise RuntimeError("The regular NVFP4 student bundle restored no TensorQuantizers.")
    _validate_nvfp4_quantizers(model, artifact_name="The regular NVFP4 student bundle")
    logging.info(
        "[QAD] validated regular ModelOpt NVFP4 bundle before FSDP: %d quantizers",
        len(quantizers),
    )
    return len(quantizers)


def _validate_svdquant_bundle(model: nn.Module) -> tuple[str, ...]:
    mode_states = _modelopt_mode_states(model)
    _reject_non_training_modes(mode_states)
    mode_state = mode_states.get("svdquant_calibrate")
    if mode_state is None:
        raise RuntimeError(
            "qad.student.mode=nvfp4_svdquant requires a bundle containing the "
            "svdquant_calibrate ModelOpt mode."
        )
    metadata = mode_state.get("metadata", {}).get("svdquant_peft")
    if not metadata:
        raise RuntimeError(
            "The SVDQuant bundle is malformed or predates the HF PEFT contract: its "
            "svdquant_calibrate mode has no svdquant_peft metadata. Recalibrate it "
            "with quantize.py --output-bundle."
        )

    expected_targets = tuple(metadata.get("target_modules", ()))
    named_parameters = dict(model.named_parameters())
    names = tuple(name for name in named_parameters if _SVDQUANT_PARAMETER_RE.search(name))
    expected_names = {
        f"{target_name}.lora_{factor}.modelopt_svdquant.weight"
        for target_name in expected_targets
        for factor in ("A", "B")
    }
    magnitude_gate_version = int(metadata.get("magnitude_gate_version", 0))
    if magnitude_gate_version not in (0, 1):
        raise RuntimeError(f"Unsupported SVDQuant magnitude gate version: {magnitude_gate_version}")
    if magnitude_gate_version:
        expected_names.update(
            f"{target_name}.svdquant_magnitude_delta" for target_name in expected_targets
        )
    if not expected_targets or set(names) != expected_names:
        raise RuntimeError(
            "The SVDQuant bundle did not restore the complete versioned set of "
            "lora_A/lora_B/magnitude parameters for every target module. "
            "A weight-free quantizer state or a deployment export is not a valid "
            "QAD training bundle."
        )
    if magnitude_gate_version:
        invalid_magnitude_shapes = []
        for target_name in expected_targets:
            target = model.get_submodule(target_name)
            get_base_layer = getattr(target, "get_base_layer", None)
            base_layer = get_base_layer() if callable(get_base_layer) else target
            parameter = named_parameters[f"{target_name}.svdquant_magnitude_delta"]
            if tuple(parameter.shape) != (base_layer.out_features,):
                invalid_magnitude_shapes.append(target_name)
        if invalid_magnitude_shapes:
            raise RuntimeError(
                "SVDQuant magnitude deltas must have one value per output channel: "
                + ", ".join(invalid_magnitude_shapes[:5])
            )
    _validate_nvfp4_quantizers(
        model,
        artifact_name="The SVDQuant student bundle",
        required_targets=expected_targets,
    )
    missing_pre_quant_scale_buffers: list[str] = []
    for target_name in expected_targets:
        target = model.get_submodule(target_name)
        get_base_layer = getattr(target, "get_base_layer", None)
        base_layer = get_base_layer() if callable(get_base_layer) else target
        input_quantizer = getattr(base_layer, "input_quantizer", None)
        pre_quant_scale = getattr(input_quantizer, "_pre_quant_scale", None)
        if (
            pre_quant_scale is None
            or getattr(input_quantizer, "_buffers", {}).get("_pre_quant_scale")
            is not pre_quant_scale
        ):
            missing_pre_quant_scale_buffers.append(target_name)
    if missing_pre_quant_scale_buffers:
        raise RuntimeError(
            "SVDQuant pre_quant_scale must be restored as frozen TensorQuantizer buffer "
            "state for every target; missing or non-buffer targets: "
            + ", ".join(missing_pre_quant_scale_buffers[:5])
        )
    logging.info(
        "[QAD] validated SVDQuant training bundle before FSDP: %d targets, %d trainable tensors",
        len(expected_targets),
        len(names),
    )
    return names


def _apply_train_scope(model: nn.Module, scope: str) -> list[nn.Parameter]:
    if scope == "lora_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(_SVDQUANT_PARAMETER_RE.search(name) is not None)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"qad.student.train_scope={scope!r} selected no parameters.")

    if scope == "lora_only":
        live_names = tuple(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        invalid = [name for name in live_names if not _SVDQUANT_PARAMETER_RE.search(name)]
        if invalid:
            raise RuntimeError(
                "lora_only left non-SVDQuant parameters trainable: " + ", ".join(invalid[:5])
            )

    parameter_pre_scales = [
        name for name, _ in model.named_parameters() if "pre_quant_scale" in name
    ]
    if parameter_pre_scales:
        raise RuntimeError(
            "pre_quant_scale must remain a buffer and must never enter the optimizer: "
            + ", ".join(parameter_pre_scales[:5])
        )
    return trainable


def _rebuild_optimizer_from_live_parameters(
    optimizer: torch.optim.Optimizer,
    parameters: list[nn.Parameter],
) -> torch.optim.Optimizer:
    """Retarget the just-built optimizer without carrying stale parameter refs."""
    if optimizer.state:
        raise RuntimeError("QAD expected a newly-created optimizer with no state.")
    if len(optimizer.param_groups) != 1:
        raise RuntimeError(
            "QAD lora_only currently expects AutoModel to create one optimizer parameter group."
        )
    # Optimizer.defaults may contain normalized/internal fields that its public
    # constructor does not accept (for example AdamW's
    # ``decoupled_weight_decay``). Preserve the fresh optimizer and its exact
    # configured group hyperparameters, replacing only the owned parameters.
    optimizer.param_groups[0]["params"] = list(parameters)
    return optimizer


def _validate_optimizer_membership(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual_list = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    actual = {id(parameter) for parameter in actual_list}
    if len(actual) != len(actual_list):
        raise RuntimeError("The student optimizer contains duplicate parameter references.")
    if actual != expected:
        raise RuntimeError(
            "Student optimizer membership does not exactly match the live post-FSDP "
            f"trainable parameters (missing={len(expected - actual)}, extra={len(actual - expected)})."
        )


def _validate_attention_grill_after_fsdp(
    model: nn.Module,
    settings: AttentionGrillSettings,
) -> None:
    if not settings.enabled:
        return

    anchor_backends = tuple(
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("._grill_static_smooth_qk_backend")
    )
    if len(anchor_backends) != settings.expected_replaced:
        raise RuntimeError(
            "FSDP/activation checkpointing did not preserve the calibrated Attention "
            f"Grill modules: found {len(anchor_backends)}, expected {settings.expected_replaced}."
        )
    for name, module in anchor_backends:
        for buffer_name in ("q_anchor", "k_anchor"):
            buffer = module._buffers.get(buffer_name)
            if buffer is None:
                raise RuntimeError(f"Post-FSDP {name} is missing {buffer_name}.")
            if buffer.requires_grad:
                raise RuntimeError(f"Attention Grill anchor {name}.{buffer_name} became trainable.")
            if buffer.device.type != "cuda" or buffer.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"Attention Grill anchor {name}.{buffer_name} must remain local CUDA BF16, "
                    f"got device={buffer.device} dtype={buffer.dtype}."
                )
            if buffer_name not in module._non_persistent_buffers_set:
                raise RuntimeError(
                    f"Attention Grill anchor {name}.{buffer_name} became checkpoint-persistent."
                )

    anchor_parameters = [
        name for name, _ in model.named_parameters() if "_grill_static_smooth_qk_backend" in name
    ]
    if anchor_parameters:
        raise RuntimeError(
            "Attention Grill must not add optimizer-owned parameters: "
            + ", ".join(anchor_parameters[:5])
        )

    bound_count = 0
    for module in model.modules():
        processor = getattr(module, "processor", None)
        if processor is None or not hasattr(processor, "_grill_bound_backend_handle"):
            continue
        handle = processor._grill_bound_backend_handle
        if processor._attention_backend != handle.member:
            raise RuntimeError(
                "The calibrated Attention Grill backend was overwritten after FSDP. "
                "Do not set model.attention_backend for this QAD run."
            )
        bound_count += 1
    if bound_count != settings.expected_replaced:
        raise RuntimeError(
            f"Post-FSDP Attention Grill has {bound_count} bound processors; expected "
            f"{settings.expected_replaced}."
        )


def _guard_automodel_hooks(diffusion_train: Any, auto_pipeline: Any) -> None:
    builder_parameters = inspect.signature(diffusion_train.build_model_and_optimizer).parameters
    required_builder_parameters = {
        "model_id",
        "learning_rate",
        "device",
        "dtype",
        "optimizer_cfg",
    }
    missing = required_builder_parameters - set(builder_parameters)
    if missing:
        raise RuntimeError(
            "Unsupported nemo_automodel diffusion builder; missing parameters: "
            + ", ".join(sorted(missing))
        )
    if not hasattr(auto_pipeline, "_apply_parallelization"):
        raise RuntimeError(
            "Unsupported nemo_automodel: auto_diffusion_pipeline._apply_parallelization is missing."
        )


@contextlib.contextmanager
def patch_student_build(
    settings: StudentSettings,
    attention_grill: AttentionGrillSettings | None = None,
) -> Iterator[StudentBuildState]:
    """Patch the two example-local seams needed during the parent ``setup`` call.

    Both module globals are restored in ``finally``. The patch is active only while
    the one student is being constructed; teacher construction happens afterwards.
    """
    from nemo_automodel._diffusers import auto_diffusion_pipeline as auto_pipeline
    from nemo_automodel.recipes.diffusion import train as diffusion_train

    _guard_automodel_hooks(diffusion_train, auto_pipeline)
    original_apply_parallelization = auto_pipeline._apply_parallelization
    original_build_model_and_optimizer = diffusion_train.build_model_and_optimizer
    state = StudentBuildState()
    attention_grill = attention_grill or AttentionGrillSettings()
    attention_grill.validate()
    apply_calls = 0

    def apply_parallelization(pipe, parallel_scheme):
        nonlocal apply_calls
        apply_calls += 1
        if apply_calls != 1:
            raise RuntimeError(
                "QAD's guarded student build expected exactly one parallelized component load."
            )
        state.parallel_scheme = parallel_scheme
        transformer = pipe.transformer
        if settings.mode == "nvfp4":
            state.quantizer_count = _validate_regular_bundle(transformer)
        else:
            state.svdquant_parameter_names = _validate_svdquant_bundle(transformer)
            state.quantizer_count = sum(
                isinstance(module, TensorQuantizer) for module in transformer.modules()
            )
        state.quantized_block_indices = _find_quantized_transformer_blocks(transformer)
        state.attention_grill_summary = _install_attention_grill(
            transformer,
            student=settings,
            settings=attention_grill,
        )
        return original_apply_parallelization(pipe, parallel_scheme)

    def build_model_and_optimizer(**kwargs):
        pipe, optimizer, device_mesh = original_build_model_and_optimizer(**kwargs)
        _validate_attention_grill_after_fsdp(pipe.transformer, attention_grill)
        trainable = _apply_train_scope(pipe.transformer, settings.train_scope)
        if settings.train_scope == "lora_only":
            optimizer = _rebuild_optimizer_from_live_parameters(optimizer, trainable)
            logging.info(
                "[QAD] rebuilt optimizer after FSDP for lora_only: %d live SVDQuant tensors",
                len(trainable),
            )
        _validate_optimizer_membership(pipe.transformer, optimizer)
        return pipe, optimizer, device_mesh

    auto_pipeline._apply_parallelization = apply_parallelization
    diffusion_train.build_model_and_optimizer = build_model_and_optimizer
    try:
        yield state
    finally:
        diffusion_train.build_model_and_optimizer = original_build_model_and_optimizer
        auto_pipeline._apply_parallelization = original_apply_parallelization

    if apply_calls != 1 or state.parallel_scheme is None:
        raise RuntimeError(
            "QAD did not observe the expected pre-FSDP student parallelization point."
        )
