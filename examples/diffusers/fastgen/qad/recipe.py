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

"""FastGen quantization-aware distillation recipe.

QAD is deliberately separate from DMD2: one frozen Diffusers teacher and one
quantized student see the same noisy latent, timestep, and conditioning, and
ModelOpt's standard ``kd_loss`` API supplies output and optional representation
MSE or cosine losses.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
from typing import Any

import torch
import yaml
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import wandb
from modelopt.torch.quantization.nn import TensorQuantizer

try:
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
    from nemo_automodel.components.training.utils import (
        clip_grad_norm,
        prepare_after_first_microbatch,
        prepare_for_final_backward,
        prepare_for_grad_accumulation,
    )
    from nemo_automodel.recipes.base_recipe import (
        _find_latest_checkpoint,
        _resolve_restore_from_to_ckpt_dir,
    )
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe, is_main_process
except ImportError as exc:
    raise ImportError(
        "The FastGen QAD example requires nemo_automodel. Install dependencies with:\n"
        "    pip install -r examples/diffusers/fastgen/requirements.txt"
    ) from exc

from fastgen_checkpoint import make_optimizer_partial_load_tolerant

from .artifacts import AttentionGrillSettings, StudentSettings, patch_student_build
from .modeling import build_distillation_controller, clear_captured_outputs
from .pipeline import QADPipeline, configure_qad_timestep_sampling


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)


def _resolve_layerwise_config(layerwise: dict[str, Any]) -> dict[str, Any]:
    supported_keys = {
        "enabled",
        "selection",
        "type",
        "weight",
        "reduction",
        "streams",
        "pairs",
        "log_per_block",
    }
    unknown_keys = set(layerwise) - supported_keys
    if unknown_keys:
        raise ValueError("Unsupported qad.layerwise field(s): " + ", ".join(sorted(unknown_keys)))

    enabled = bool(layerwise.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "selection": None,
            "type": "mse",
            "weight": 0.0,
            "reduction": "mean",
            "streams": (),
            "pairs": (),
            "log_per_block": False,
        }

    loss_type = str(layerwise.get("type", "mse")).lower()
    if loss_type not in {"mse", "cosine"}:
        raise ValueError("qad.layerwise.type must be mse or cosine.")

    raw_pairs = tuple(_as_dict(pair) for pair in layerwise.get("pairs", ()))
    raw_selection = layerwise.get("selection")
    selection = str(raw_selection).lower() if raw_selection is not None else None
    if selection is not None and raw_pairs:
        raise ValueError("qad.layerwise.selection and qad.layerwise.pairs are mutually exclusive.")
    if selection is None and not raw_pairs:
        raise ValueError(
            "Enabled qad.layerwise requires selection=quantized_blocks or explicit pairs."
        )

    if selection is None:
        auto_only_keys = {"weight", "reduction", "streams", "log_per_block"} & set(layerwise)
        if auto_only_keys:
            raise ValueError(
                "Explicit qad.layerwise.pairs use each pair's own weight and do not accept: "
                + ", ".join(sorted(auto_only_keys))
            )
        pairs = []
        for index, pair in enumerate(raw_pairs):
            unknown_pair_keys = set(pair) - {
                "student_layer",
                "teacher_layer",
                "selector",
                "weight",
            }
            if unknown_pair_keys:
                raise ValueError(
                    f"Unsupported qad.layerwise.pairs[{index}] field(s): "
                    + ", ".join(sorted(unknown_pair_keys))
                )
            student_layer = pair.get("student_layer")
            if not student_layer:
                raise ValueError(f"qad.layerwise.pairs[{index}].student_layer is required.")
            pairs.append(
                {
                    "student_layer": str(student_layer),
                    "teacher_layer": str(pair.get("teacher_layer", student_layer)),
                    "selector": str(pair.get("selector", "hidden_states")),
                    "weight": float(pair.get("weight", 1.0)),
                }
            )
        return {
            "enabled": True,
            "selection": None,
            "type": loss_type,
            "weight": 0.0,
            "reduction": "mean",
            "streams": (),
            "pairs": tuple(pairs),
            "log_per_block": False,
        }

    if selection != "quantized_blocks":
        raise ValueError("qad.layerwise.selection must be quantized_blocks.")
    reduction = str(layerwise.get("reduction", "mean")).lower()
    if reduction != "mean":
        raise ValueError("qad.layerwise.reduction currently supports only mean.")

    default_streams = (
        {"selector": "encoder_hidden_states", "weight": 0.2},
        {"selector": "hidden_states", "weight": 0.8},
    )
    selector_aliases = {
        "text": "encoder_hidden_states",
        "encoder_hidden_states": "encoder_hidden_states",
        "image": "hidden_states",
        "hidden_states": "hidden_states",
    }
    streams: list[dict[str, Any]] = []
    for index, stream in enumerate(layerwise.get("streams", default_streams)):
        stream = _as_dict(stream)
        unknown_stream_keys = set(stream) - {"selector", "weight"}
        if unknown_stream_keys:
            raise ValueError(
                f"Unsupported qad.layerwise.streams[{index}] field(s): "
                + ", ".join(sorted(unknown_stream_keys))
            )
        selector = str(stream.get("selector", "")).lower()
        if selector not in selector_aliases:
            raise ValueError(
                f"qad.layerwise.streams[{index}].selector must identify the Qwen text "
                "or image stream."
            )
        streams.append(
            {
                "selector": selector_aliases[selector],
                "weight": float(stream.get("weight", 0.0)),
            }
        )
    if not streams:
        raise ValueError("qad.layerwise.streams must contain at least one stream.")
    selectors = [stream["selector"] for stream in streams]
    if len(selectors) != len(set(selectors)):
        raise ValueError("qad.layerwise.streams contains a duplicate text/image stream.")
    stream_weights = [float(stream["weight"]) for stream in streams]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in stream_weights):
        raise ValueError("QAD stream weights must be finite and non-negative.")
    if not math.isclose(sum(stream_weights), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("qad.layerwise.stream weights must sum to 1.0.")

    return {
        "enabled": True,
        "selection": selection,
        "type": loss_type,
        "weight": float(layerwise.get("weight", 1.0)),
        "reduction": reduction,
        "streams": tuple(streams),
        "pairs": (),
        "log_per_block": bool(layerwise.get("log_per_block", False)),
    }


class QADDiffusionRecipe(TrainDiffusionRecipe):
    """AutoModel diffusion recipe with a ModelOpt KD controller."""

    def __init__(self, cfg) -> None:
        # AutoModel's dotted CLI setter updates live ConfigNodes but not the
        # raw_config later written beside checkpoints. Materialize the resolved
        # runtime values so QAD paths, scope, teacher, and losses are reproducible.
        if hasattr(cfg, "to_yaml_dict"):
            cfg.__dict__["_raw_config"] = cfg.to_yaml_dict(
                resolve_env=False,
                redact_sensitive=True,
                use_orig_values=False,
            )
        super().__init__(cfg)

    def setup(self) -> None:
        settings, attention_grill, loss_config = self._resolve_qad_config()
        self.__dict__["_qad_resume_signature"] = self._resume_signature(
            settings,
            attention_grill,
            loss_config,
        )

        if self.cfg.get("peft", None) is not None:
            raise ValueError(
                "Do not set AutoModel's top-level peft block for QAD. SVDQuant's "
                "modelopt_svdquant HF PEFT topology comes from the student bundle."
            )
        if str(self.cfg.get("model.mode", "finetune")).lower() != "finetune":
            raise ValueError("QAD supports model.mode=finetune only.")
        if attention_grill.enabled and self.cfg.get("model.attention_backend", None) is not None:
            raise ValueError(
                "Do not set model.attention_backend when qad.attention_grill.enabled=true; "
                "AutoModel would overwrite the calibrated per-layer backends after FSDP."
            )
        if self.cfg.get("ddp", None) is not None:
            raise ValueError("QAD currently supports AutoModel FSDP2, not DDP.")
        fsdp = _as_dict(self.cfg.get("fsdp"))
        model_parallel_sizes = {
            name: int(fsdp.get(name, 1)) for name in ("tp_size", "cp_size", "pp_size")
        }
        if any(size != 1 for size in model_parallel_sizes.values()):
            raise ValueError(
                "QAD currently supports data-parallel FSDP2 only; "
                f"found model parallel sizes {model_parallel_sizes}."
            )
        if attention_grill.enabled and bool(fsdp.get("enable_compile", False)):
            raise ValueError(
                "Calibrated Attention Grill QAD does not support fsdp.enable_compile=true."
            )

        # Diffusers' ModelMixin must be patched before from_pretrained so both
        # regular NVFP4 and SVDQuant bundles rebuild their ModelOpt topology and
        # load component-local modelopt_state.pth before AutoModel applies FSDP.
        mto.enable_huggingface_checkpointing()

        with patch_student_build(settings, attention_grill) as build_state:
            super().setup()

        # Diffusers loads ModelMixin objects in eval mode. QAD owns the student
        # train/eval boundary because the controller delegate intentionally does
        # not register the live FSDP module as a child.
        self.model.train()

        timestep_summary = configure_qad_timestep_sampling(
            self.flow_matching_pipeline,
            student_model_name_or_path=settings.model_name_or_path,
            timestep_config=loss_config["timestep_config"],
        )

        # Parent checkpoint restore established the exact next-step RNG state.
        # Teacher construction/sharding is transient setup and must not perturb
        # the first fresh or resumed training sample.
        training_rng_state = self.rng.state_dict()
        try:
            parallel_scheme = build_state.parallel_scheme
            if parallel_scheme is None:
                raise RuntimeError("QAD failed to capture the student's parallel scheme.")
            teacher = self._load_frozen_teacher(
                loss_config["teacher_model_name_or_path"],
                parallel_scheme,
            )
            controller, loss_layout = build_distillation_controller(
                student=self.model,
                teacher=teacher,
                output_weight=loss_config["output_weight"],
                task_weight=loss_config["task_weight"],
                layerwise_config=loss_config["layerwise_config"],
                quantized_block_indices=build_state.quantized_block_indices,
            )
            if any(True for _ in controller.parameters()):
                raise RuntimeError(
                    "The QAD controller must remain parameter-free; optimizer/checkpoint "
                    "ownership belongs exclusively to self.model."
                )

            # BaseRecipe tracks nn.Module assignments. Bypass it for the frozen teacher
            # and transient controller so checkpoint selection cannot mistake either for
            # the student.
            object.__setattr__(self, "_qad_teacher", teacher)
            object.__setattr__(self, "_qad_controller", controller)
            object.__setattr__(
                self,
                "_qad_pipeline",
                QADPipeline(self.flow_matching_pipeline, controller, loss_layout),
            )
            object.__setattr__(self, "_qad_student_settings", settings)
            object.__setattr__(self, "_qad_attention_grill_settings", attention_grill)
            object.__setattr__(self, "_qad_loss_config", loss_config)
            object.__setattr__(self, "_qad_timestep_summary", timestep_summary)

            tracked = self.__dict__.get("__state_tracked", set())
            forbidden = {"_qad_teacher", "_qad_controller", "_qad_pipeline"} & set(tracked)
            if forbidden:
                raise RuntimeError(
                    f"QAD transient objects were accidentally state-tracked: {forbidden}"
                )
            self._validate_state_ownership()
        finally:
            self.rng.load_state_dict(training_rng_state)

        if is_main_process():
            logging.info(
                "[QAD] initialized: teacher=%s student=%s mode=%s train_scope=%s "
                "task_weight=%g output_weight=%g layerwise=%s",
                loss_config["teacher_model_name_or_path"],
                settings.model_name_or_path,
                settings.mode,
                settings.train_scope,
                loss_config["task_weight"],
                loss_config["output_weight"],
                loss_config["layerwise_config"],
            )
            if build_state.attention_grill_summary is not None:
                summary = build_state.attention_grill_summary
                logging.info(
                    "[QAD] Attention Grill is frozen/non-persistent: kernel=%s "
                    "replaced=%d ignored=%d calibration=%s",
                    summary["kernel"],
                    len(summary["replaced"]),
                    len(summary["ignored"]),
                    summary["calibration_path"],
                )
            if loss_config["layerwise_config"]["selection"] == "quantized_blocks":
                logging.info(
                    "[QAD] blockwise distillation targets %d discovered blocks: %s",
                    len(build_state.quantized_block_indices),
                    ",".join(str(index) for index in build_state.quantized_block_indices),
                )
            logging.info("[QAD] student quantizer summary:")
            mtq.print_quant_summary(self.model)
            logging.info("[QAD] timestep sampling: %s", timestep_summary)

    @staticmethod
    def _resolve_timestep_config(
        qad: dict[str, Any],
        flow_matching: dict[str, Any],
    ) -> dict[str, Any]:
        timestep = _as_dict(qad.get("timestep"))
        supported_keys = {
            "schedule",
            "num_inference_steps",
            "inference_step_start",
            "inference_step_end",
            "image_seq_len",
        }
        unknown_keys = set(timestep) - supported_keys
        if unknown_keys:
            raise ValueError(
                "Unsupported qad.timestep field(s): " + ", ".join(sorted(unknown_keys))
            )
        schedule = str(timestep.get("schedule", "")).lower()
        if schedule not in {"qwen_image", "qwen_image_flash"}:
            raise ValueError("qad.timestep.schedule must be qwen_image or qwen_image_flash.")

        config: dict[str, Any] = {"schedule": schedule}
        if schedule == "qwen_image_flash":
            if "image_seq_len" in timestep:
                raise ValueError("qad.timestep.image_seq_len applies only to schedule=qwen_image.")
            config["num_inference_steps"] = 4
            if int(timestep.get("num_inference_steps", 4)) != 4:
                raise ValueError("Qwen-Image-Flash QAD requires num_inference_steps=4.")
            range_keys = {"inference_step_start", "inference_step_end"}
            provided_range_keys = range_keys & set(timestep)
            if provided_range_keys:
                if provided_range_keys != range_keys:
                    missing = range_keys - provided_range_keys
                    raise ValueError(
                        "Qwen-Image-Flash timestep range is missing: " + ", ".join(sorted(missing))
                    )
                start_step = int(timestep["inference_step_start"])
                end_step = int(timestep["inference_step_end"])
                if not 0 <= start_step < end_step <= 4:
                    raise ValueError(
                        "Qwen-Image-Flash timestep range must satisfy 0 <= start < end <= 4."
                    )
                config["inference_step_range"] = {
                    "num_inference_steps": 4,
                    "start": start_step,
                    "end": end_step,
                }
            elif "num_inference_steps" in timestep:
                raise ValueError(
                    "qad.timestep.num_inference_steps requires both "
                    "inference_step_start and inference_step_end."
                )
            return config

        sampling = str(flow_matching.get("timestep_sampling", "logit_normal")).lower()
        if sampling not in {"logit_normal", "uniform"}:
            raise ValueError(
                "Qwen-Image QAD supports full-range timestep_sampling=logit_normal or uniform."
            )
        config.update(
            {
                "timestep_sampling": sampling,
                "logit_mean": float(flow_matching.get("logit_mean", 0.0)),
                "logit_std": float(flow_matching.get("logit_std", 1.0)),
                "flow_shift": float(flow_matching.get("flow_shift", 3.0)),
                "mix_uniform_ratio": float(flow_matching.get("mix_uniform_ratio", 0.1)),
                "use_sigma_noise": bool(flow_matching.get("use_sigma_noise", True)),
                "sigma_min": float(flow_matching.get("sigma_min", 0.0)),
                "sigma_max": float(flow_matching.get("sigma_max", 1.0)),
                "num_train_timesteps": int(flow_matching.get("num_train_timesteps", 1000)),
            }
        )

        range_keys = {"inference_step_start", "inference_step_end"}
        provided_range_keys = range_keys & set(timestep)
        if provided_range_keys:
            if provided_range_keys != range_keys:
                missing = range_keys - provided_range_keys
                raise ValueError(
                    "Qwen-Image timestep range is missing: " + ", ".join(sorted(missing))
                )
            if sampling != "uniform":
                raise ValueError(
                    "Qwen-Image inference-step ranges require "
                    "flow_matching.timestep_sampling=uniform."
                )
            num_inference_steps = int(timestep.get("num_inference_steps", 50))
            start_step = int(timestep["inference_step_start"])
            end_step = int(timestep["inference_step_end"])
            image_seq_len = int(timestep.get("image_seq_len", 4096))
            if num_inference_steps <= 0:
                raise ValueError("qad.timestep.num_inference_steps must be positive.")
            if not 0 <= start_step < end_step <= num_inference_steps:
                raise ValueError(
                    "Qwen-Image inference-step range must satisfy "
                    "0 <= start < end <= num_inference_steps."
                )
            if image_seq_len <= 0:
                raise ValueError("qad.timestep.image_seq_len must be positive.")
            config["inference_step_range"] = {
                "num_inference_steps": num_inference_steps,
                "start": start_step,
                "end": end_step,
                "image_seq_len": image_seq_len,
            }
        elif {"num_inference_steps", "image_seq_len"} & set(timestep):
            raise ValueError(
                "qad.timestep.num_inference_steps/image_seq_len require both "
                "inference_step_start and inference_step_end."
            )
        return config

    @staticmethod
    def _resolve_attention_grill_config(
        qad: dict[str, Any],
        timestep_config: dict[str, Any],
    ) -> AttentionGrillSettings:
        config = _as_dict(qad.get("attention_grill"))
        supported_keys = {
            "enabled",
            "recipe_path",
            "calibration_path",
            "ignore",
            "expected_replaced",
            "expected_ignored",
        }
        unknown_keys = set(config) - supported_keys
        if unknown_keys:
            raise ValueError(
                "Unsupported qad.attention_grill field(s): " + ", ".join(sorted(unknown_keys))
            )

        raw_ignore = config.get("ignore", ())
        if isinstance(raw_ignore, str):
            ignore = (raw_ignore,)
        else:
            ignore = tuple(str(pattern) for pattern in raw_ignore)
        settings = AttentionGrillSettings(
            enabled=bool(config.get("enabled", False)),
            recipe_path=(
                str(config["recipe_path"]) if config.get("recipe_path") is not None else None
            ),
            calibration_path=(
                str(config["calibration_path"])
                if config.get("calibration_path") is not None
                else None
            ),
            ignore=ignore,
            expected_replaced=int(config.get("expected_replaced", 56)),
            expected_ignored=int(config.get("expected_ignored", 4)),
            inference_profile=str(timestep_config["schedule"]),
        )
        settings.validate()
        return settings

    def _resolve_qad_config(
        self,
    ) -> tuple[StudentSettings, AttentionGrillSettings, dict[str, Any]]:
        qad = _as_dict(self.cfg.get("qad", None))
        if not qad:
            raise ValueError("Missing required qad configuration block.")

        student_cfg = _as_dict(qad.get("student"))
        secondary_artifact_fields = sorted(
            field
            for field in ("quant_state_path", "modelopt_state_path")
            if student_cfg.get(field) is not None
        )
        if secondary_artifact_fields:
            raise ValueError(
                "QAD accepts one complete Diffusers student bundle through "
                "model.pretrained_model_name_or_path; remove unsupported secondary "
                "artifact field(s): "
                + ", ".join(f"qad.student.{field}" for field in secondary_artifact_fields)
            )
        model_name_or_path = self.cfg.get("model.pretrained_model_name_or_path", None)
        if not model_name_or_path:
            raise ValueError(
                "model.pretrained_model_name_or_path is required and is the canonical "
                "student source recorded in checkpoints."
            )
        duplicate_student_path = student_cfg.get("model_name_or_path")
        if duplicate_student_path is not None and str(duplicate_student_path) != str(
            model_name_or_path
        ):
            raise ValueError(
                "qad.student.model_name_or_path conflicts with the canonical "
                "model.pretrained_model_name_or_path. Remove the duplicate QAD field."
            )
        mode = str(student_cfg.get("mode", "nvfp4")).lower()
        # Accept the early design spelling while emitting one canonical name.
        if mode == "svdquant_nvfp4":
            mode = "nvfp4_svdquant"
        settings = StudentSettings(
            mode=mode,
            model_name_or_path=str(model_name_or_path),
            train_scope=str(student_cfg.get("train_scope", "all")).lower(),
        )
        settings.validate()

        teacher_model_name_or_path = qad.get("teacher_model_name_or_path")
        if not teacher_model_name_or_path:
            raise ValueError("qad.teacher_model_name_or_path is required.")

        timestep_config = self._resolve_timestep_config(
            qad,
            _as_dict(self.cfg.get("flow_matching", {})),
        )
        attention_grill = self._resolve_attention_grill_config(qad, timestep_config)

        output_cfg = _as_dict(qad.get("output_loss"))
        if str(output_cfg.get("type", "mse")).lower() != "mse":
            raise ValueError("QAD currently supports only output_loss.type=mse.")
        output_weight = float(output_cfg.get("weight", 1.0))

        task_cfg = _as_dict(qad.get("task_loss"))
        task_weight = float(task_cfg.get("weight", 0.0))

        layerwise_config = _resolve_layerwise_config(_as_dict(qad.get("layerwise")))
        layerwise_weights = [float(pair["weight"]) for pair in layerwise_config["pairs"]]
        if layerwise_config["selection"] == "quantized_blocks":
            layerwise_weights.append(float(layerwise_config["weight"]))

        all_weights = [output_weight, task_weight, *layerwise_weights]
        if any(not math.isfinite(weight) or weight < 0.0 for weight in all_weights):
            raise ValueError("QAD loss weights must be finite and non-negative.")
        if not any(weight > 0.0 for weight in all_weights):
            raise ValueError("At least one QAD loss weight must be positive.")

        return (
            settings,
            attention_grill,
            {
                "teacher_model_name_or_path": str(teacher_model_name_or_path),
                "output_weight": output_weight,
                "task_weight": task_weight,
                "layerwise_config": layerwise_config,
                "timestep_config": timestep_config,
            },
        )

    @staticmethod
    def _layerwise_resume_signature(layerwise: dict[str, Any]) -> tuple[Any, ...]:
        if not layerwise["enabled"]:
            return ("disabled",)
        if layerwise["selection"] == "quantized_blocks":
            return (
                "quantized_blocks",
                str(layerwise["type"]),
                float(layerwise["weight"]),
                str(layerwise["reduction"]),
                tuple(
                    (str(stream["selector"]), float(stream["weight"]))
                    for stream in layerwise["streams"]
                ),
            )
        return (
            "pairs",
            str(layerwise["type"]),
            tuple(
                (
                    str(pair["student_layer"]),
                    str(pair["teacher_layer"]),
                    str(pair["selector"]),
                    float(pair["weight"]),
                )
                for pair in layerwise["pairs"]
            ),
        )

    @staticmethod
    def _resume_signature(
        settings: StudentSettings,
        attention_grill: AttentionGrillSettings,
        loss_config: dict[str, Any],
    ) -> dict[str, Any]:
        if attention_grill.enabled:
            attention_grill_signature: tuple[Any, ...] = (
                "enabled",
                os.path.realpath(str(attention_grill.recipe_path)),
                os.path.realpath(str(attention_grill.calibration_path)),
                attention_grill.ignore,
                attention_grill.expected_replaced,
                attention_grill.expected_ignored,
            )
        else:
            attention_grill_signature = ("disabled",)
        return {
            "student_source": settings.model_name_or_path,
            "student_mode": settings.mode,
            "train_scope": settings.train_scope,
            "teacher_source": loss_config["teacher_model_name_or_path"],
            "output_weight": float(loss_config["output_weight"]),
            "task_weight": float(loss_config["task_weight"]),
            "timestep_config": loss_config["timestep_config"],
            "attention_grill": attention_grill_signature,
            "layerwise": QADDiffusionRecipe._layerwise_resume_signature(
                loss_config["layerwise_config"]
            ),
        }

    @classmethod
    def _resume_signature_from_saved_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        model_cfg = _as_dict(config.get("model"))
        qad_cfg = _as_dict(config.get("qad"))
        student_cfg = _as_dict(qad_cfg.get("student"))
        secondary_artifact_fields = sorted(
            field
            for field in ("quant_state_path", "modelopt_state_path")
            if student_cfg.get(field) is not None
        )
        if secondary_artifact_fields:
            raise RuntimeError(
                "The saved QAD checkpoint uses unsupported secondary student artifact "
                "field(s): "
                + ", ".join(f"qad.student.{field}" for field in secondary_artifact_fields)
            )
        output_cfg = _as_dict(qad_cfg.get("output_loss"))
        task_cfg = _as_dict(qad_cfg.get("task_loss"))
        layerwise_cfg = _as_dict(qad_cfg.get("layerwise"))
        timestep_config = cls._resolve_timestep_config(
            qad_cfg,
            _as_dict(config.get("flow_matching")),
        )
        attention_grill = cls._resolve_attention_grill_config(qad_cfg, timestep_config)

        mode = str(student_cfg.get("mode", "nvfp4")).lower()
        if mode == "svdquant_nvfp4":
            mode = "nvfp4_svdquant"
        loss_config = {
            "teacher_model_name_or_path": str(qad_cfg.get("teacher_model_name_or_path", "")),
            "output_weight": float(output_cfg.get("weight", 1.0)),
            "task_weight": float(task_cfg.get("weight", 0.0)),
            "layerwise_config": _resolve_layerwise_config(layerwise_cfg),
            "timestep_config": timestep_config,
        }
        settings = StudentSettings(
            mode=mode,
            model_name_or_path=str(model_cfg.get("pretrained_model_name_or_path", "")),
            train_scope=str(student_cfg.get("train_scope", "all")).lower(),
        )
        return cls._resume_signature(settings, attention_grill, loss_config)

    def _resolved_checkpoint_dir(self, restore_from: str | None) -> str | None:
        if not self.checkpointer.config.enabled:
            return None
        if restore_from:
            resolved = _resolve_restore_from_to_ckpt_dir(
                self.checkpointer.config.checkpoint_dir,
                restore_from,
            )
        else:
            resolved = _find_latest_checkpoint(self.checkpointer.config.checkpoint_dir)
        if resolved is None:
            return None
        return os.fspath(resolved)

    def _validate_qad_checkpoint_signature(self, checkpoint_dir: str) -> None:
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if not os.path.isfile(config_path):
            raise RuntimeError(
                "QAD cannot safely restore optimizer shards from a checkpoint without "
                f"config.yaml: {checkpoint_dir}"
            )
        with open(config_path) as config_file:
            saved_config = yaml.safe_load(config_file) or {}
        saved_signature = self._resume_signature_from_saved_config(saved_config)
        current_signature = self.__dict__["_qad_resume_signature"]
        if saved_signature != current_signature:
            changed = [
                key
                for key in current_signature
                if saved_signature.get(key) != current_signature[key]
            ]
            raise RuntimeError(
                "QAD resume contract changed for "
                + ", ".join(changed)
                + ". Use the same student bundle, quantization mode, train scope, "
                "teacher, and loss configuration as the saved run."
            )

    def load_checkpoint(self, restore_from: str | None = None) -> None:
        """Validate QAD topology before enabling FSDP2 partial-shard optimizer load."""
        checkpoint_dir = self._resolved_checkpoint_dir(restore_from)
        if checkpoint_dir is not None and os.path.isdir(checkpoint_dir):
            self._validate_qad_checkpoint_signature(checkpoint_dir)
            make_optimizer_partial_load_tolerant(self.checkpointer)
        super().load_checkpoint(restore_from)

    def _remove_incomplete_checkpoint_target(self, epoch: int, global_step: int) -> None:
        """Remove a directory left behind by an interrupted checkpoint save."""
        if not self.checkpointer.config.enabled:
            return

        checkpoint_root = os.fspath(self.checkpointer.config.checkpoint_dir)
        checkpoint_name = f"epoch_{epoch}_step_{global_step}"
        checkpoint_path = os.path.join(checkpoint_root, checkpoint_name)
        cleanup_outcome: tuple[str, str] | None = None

        if is_main_process() and os.path.lexists(checkpoint_path):
            latest_path = os.path.join(checkpoint_root, "LATEST")
            latest_target = os.path.realpath(latest_path) if os.path.lexists(latest_path) else None
            if latest_target == os.path.realpath(checkpoint_path):
                cleanup_outcome = (
                    "refuse",
                    f"Refusing to overwrite the current QAD checkpoint: {checkpoint_path}",
                )
            else:
                # AutoModel writes config.yaml and optimizer DCP metadata before
                # it advances LATEST. Their absence identifies the half-written
                # directory produced when a distributed save is interrupted.
                complete_markers = (
                    os.path.join(checkpoint_path, "config.yaml"),
                    os.path.join(checkpoint_path, "optim", ".metadata"),
                )
                if all(os.path.isfile(marker) for marker in complete_markers):
                    cleanup_outcome = (
                        "refuse",
                        "Refusing to remove an apparently complete QAD checkpoint that is "
                        f"not LATEST: {checkpoint_path}",
                    )
                else:
                    logging.warning(
                        "[QAD][checkpoint] removing incomplete checkpoint left by an "
                        "interrupted save: %s",
                        checkpoint_path,
                    )
                    try:
                        shutil.rmtree(checkpoint_path)
                    except Exception as exc:
                        cleanup_outcome = (
                            "cleanup_failed",
                            f"{type(exc).__name__}: {exc}",
                        )

        if torch.distributed.is_initialized():
            payload = [cleanup_outcome]
            torch.distributed.broadcast_object_list(payload, src=0)
            cleanup_outcome = payload[0]

        if cleanup_outcome is not None:
            outcome, message = cleanup_outcome
            if outcome == "refuse":
                raise FileExistsError(message)
            raise RuntimeError(
                f"Rank 0 failed to remove incomplete checkpoint {checkpoint_path}: {message}"
            )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _rebuild_dataloader_for_resume(self, global_step: int) -> None:
        """Rebuild the loader and deterministically skip to the restored data position."""
        epoch_len = int(getattr(self.step_scheduler, "epoch_len", 0) or 0)
        grad_acc = int(getattr(self.step_scheduler, "grad_acc_steps", 1) or 1)
        if epoch_len <= 0 or self.sampler is None or global_step <= 0:
            return

        current_epoch = global_step // epoch_len
        skip_batches = (global_step % epoch_len) * grad_acc
        old_dataloader = self.dataloader
        dataloader_kwargs = {
            "collate_fn": getattr(old_dataloader, "collate_fn", None),
            "num_workers": int(getattr(old_dataloader, "num_workers", 0) or 0),
            "pin_memory": bool(getattr(old_dataloader, "pin_memory", False)),
        }
        if dataloader_kwargs["num_workers"] > 0:
            dataloader_kwargs["prefetch_factor"] = getattr(
                old_dataloader,
                "prefetch_factor",
                2,
            )
            dataloader_kwargs["persistent_workers"] = bool(
                getattr(old_dataloader, "persistent_workers", False)
            )

        # Keep the parent's existing tracked state key while replacing the
        # StatefulDataLoader object whose restored cursor is known to stick.
        self.__dict__["dataloader"] = StatefulDataLoader(
            old_dataloader.dataset,
            batch_sampler=self.sampler,
            **dataloader_kwargs,
        )
        self.step_scheduler.epoch = current_epoch
        self.sampler.set_epoch(current_epoch)
        self.sampler._batches_to_skip = skip_batches
        if is_main_process():
            logging.info(
                "[QAD][resume] rebuilt dataloader at epoch=%d skip_batches=%d "
                "(global_step=%d epoch_len=%d grad_acc=%d)",
                current_epoch,
                skip_batches,
                global_step,
                epoch_len,
                grad_acc,
            )

    def _load_frozen_teacher(
        self,
        model_name_or_path: str,
        parallel_scheme: dict[str, dict[str, Any]],
    ) -> nn.Module:
        pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=self.bf16,
            device=self.device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
            load_for_training=False,
            low_cpu_mem_usage=True,
        )
        teacher = pipe.transformer
        if mto.ModeloptStateManager.is_converted(teacher):
            raise RuntimeError(
                "QAD teacher must be a plain BF16 Diffusers checkpoint without ModelOpt modes."
            )
        if any(isinstance(module, TensorQuantizer) for module in teacher.modules()):
            raise RuntimeError("QAD teacher must be an unquantized BF16 Diffusers checkpoint.")
        teacher.eval()
        teacher.requires_grad_(False)
        return teacher

    def _validate_state_ownership(self) -> None:
        optimizer_parameters = {
            id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]
        }
        student_parameters = {
            id(parameter) for parameter in self.model.parameters() if parameter.requires_grad
        }
        teacher_parameters = {id(parameter) for parameter in self._qad_teacher.parameters()}
        if optimizer_parameters != student_parameters:
            raise RuntimeError("QAD optimizer does not exactly own the trainable student state.")
        if optimizer_parameters & teacher_parameters:
            raise RuntimeError("Frozen teacher parameters leaked into the student optimizer.")
        if any(parameter.requires_grad for parameter in self._qad_teacher.parameters()):
            raise RuntimeError("QAD teacher must be completely frozen.")

    def run_train_validation_loop(self) -> None:
        """Run a conventional optimizer loop using the QAD objective."""
        self.model.train()
        logging.info(
            "[QAD] starting training: global_batch_size=%s local_batch_size=%s dp_size=%s",
            self.global_batch_size,
            self.local_batch_size,
            self.dp_size,
        )
        global_step = int(self.step_scheduler.step)
        self._rebuild_dataloader_for_resume(global_step)

        try:
            for epoch in self.step_scheduler.epochs:
                if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
                    self.sampler.set_epoch(epoch)

                tqdm_initial = int(getattr(self.sampler, "_batches_to_skip", 0) or 0)
                if is_main_process():
                    from tqdm import tqdm

                    self.step_scheduler.dataloader = tqdm(
                        self.dataloader,
                        desc=f"Epoch {epoch + 1}/{self.num_epochs} (global step {global_step})",
                        initial=tqdm_initial,
                    )
                else:
                    self.step_scheduler.dataloader = self.dataloader

                epoch_loss = 0.0
                num_steps = 0
                for batch_group in self.step_scheduler:
                    # StepScheduler increments only after control returns to its
                    # generator, so refresh at the top of every yielded group.
                    global_step = int(self.step_scheduler.step)
                    self.optimizer.zero_grad(set_to_none=True)
                    prepare_for_grad_accumulation([self.model], pp_enabled=False)
                    num_microbatches = len(batch_group)
                    micro_metrics: list[dict[str, torch.Tensor]] = []

                    for microbatch_index, micro_batch in enumerate(batch_group):
                        if microbatch_index == num_microbatches - 1:
                            prepare_for_final_backward([self.model], pp_enabled=False)
                        try:
                            total_loss, metrics = self._qad_pipeline.step(
                                batch=micro_batch,
                                device=self.device,
                                dtype=self.bf16,
                                global_step=global_step,
                                check_loss=self.check_loss,
                            )
                            (total_loss / num_microbatches).backward()
                            micro_metrics.append(metrics)
                        finally:
                            # Full-block NO_REENTRANT checkpoint wrappers avoid hook
                            # repopulation during recompute; this final cleanup is also
                            # safe when activation checkpointing is disabled.
                            self._qad_pipeline.clear()

                        if microbatch_index == 0:
                            prepare_after_first_microbatch()

                    self._validate_first_step_gradients(global_step)
                    grad_norm = clip_grad_norm(
                        self.clip_grad_max_norm,
                        [self.model],
                        foreach=self.grad_clip_foreach,
                    )
                    grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler[0].step(1)

                    reduced_metrics = {
                        name: float(
                            torch.stack([metrics[name] for metrics in micro_metrics]).mean().item()
                        )
                        for name in micro_metrics[0]
                    }
                    group_loss = reduced_metrics["total_loss"]
                    epoch_loss += group_loss
                    num_steps += 1

                    if self.log_every and global_step % self.log_every == 0 and is_main_process():
                        log_dict = {
                            "train_loss": group_loss,
                            "train_avg_loss": epoch_loss / num_steps,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "grad_norm": grad_norm,
                            "epoch": epoch,
                            "global_step": global_step,
                            **{f"qad/{name}": value for name, value in reduced_metrics.items()},
                        }
                        if wandb.run is not None:
                            wandb.log(log_dict, step=global_step)
                        component_text = " ".join(
                            f"{name}={value:.6f}" for name, value in reduced_metrics.items()
                        )
                        logging.info(
                            "[QAD][TRAIN] step=%d epoch=%d %s lr=%.3e grad_norm=%.3f",
                            global_step,
                            epoch,
                            component_text,
                            self.optimizer.param_groups[0]["lr"],
                            grad_norm,
                        )
                        if hasattr(self.step_scheduler.dataloader, "set_postfix"):
                            self.step_scheduler.dataloader.set_postfix(
                                {
                                    "loss": f"{group_loss:.4f}",
                                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                                    "gn": f"{grad_norm:.2f}",
                                }
                            )

                    if self.step_scheduler.is_ckpt_step:
                        self._remove_incomplete_checkpoint_target(epoch, global_step)
                        self.save_checkpoint(epoch, global_step, epoch_loss / num_steps)

                if num_steps == 0:
                    logging.info(
                        "[QAD] epoch %d skipped (already completed in previous run)", epoch + 1
                    )
                    continue
                logging.info(
                    "[QAD] epoch %d complete: avg_loss=%.6f",
                    epoch + 1,
                    epoch_loss / num_steps,
                )

            if is_main_process() and wandb.run is not None:
                wandb.finish()
            logging.info("[QAD] training complete at step %d", global_step)
        finally:
            self._release_distillation_controller()

    def _validate_first_step_gradients(self, global_step: int) -> None:
        if global_step != 0:
            return
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not any(parameter.grad is not None for parameter in trainable):
            raise RuntimeError("QAD produced no gradients for any trainable student parameter.")

        attention_grill = self._qad_attention_grill_settings
        if attention_grill.enabled:
            missing_projection_gradients: list[str] = []
            bound_attention_count = 0
            projection_grad_sq = torch.zeros(3, device=self.device, dtype=torch.float32)
            for module_name, module in self.model.named_modules():
                processor = getattr(module, "processor", None)
                if processor is None or not hasattr(processor, "_grill_bound_backend_handle"):
                    continue
                bound_attention_count += 1
                for projection_index, projection_name in enumerate(("to_q", "to_k", "to_v")):
                    projection = getattr(module, projection_name, None)
                    projection_parameters = (
                        tuple(
                            parameter
                            for parameter in projection.parameters()
                            if parameter.requires_grad
                        )
                        if projection is not None
                        else ()
                    )
                    if not projection_parameters or not any(
                        parameter.grad is not None for parameter in projection_parameters
                    ):
                        missing_projection_gradients.append(f"{module_name}.{projection_name}")
                    for parameter in projection_parameters:
                        gradient = parameter.grad
                        if gradient is None:
                            continue
                        if hasattr(gradient, "to_local"):
                            gradient = gradient.to_local()
                        projection_grad_sq[projection_index] += (
                            gradient.detach().float().square().sum()
                        )

            local_ok = (
                bound_attention_count == attention_grill.expected_replaced
                and not missing_projection_gradients
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    projection_grad_sq,
                    op=torch.distributed.ReduceOp.SUM,
                )
                global_ok = torch.tensor(int(local_ok), device=self.device)
                torch.distributed.all_reduce(
                    global_ok,
                    op=torch.distributed.ReduceOp.MIN,
                )
                local_ok = bool(global_ok.item())
            if not local_ok:
                details = ", ".join(missing_projection_gradients[:5]) or "failure on a peer rank"
                raise RuntimeError(
                    "Calibrated Attention Grill did not propagate first-step gradients "
                    f"through every Q/K/V projection ({details})."
                )
            if is_main_process():
                projection_grad_norms = projection_grad_sq.sqrt().tolist()
                logging.info(
                    "[QAD] verified Attention Grill STE gradients through Q/K/V in %d "
                    "blocks: to_q=%.6e to_k=%.6e to_v=%.6e",
                    bound_attention_count,
                    *projection_grad_norms,
                )

        if self._qad_student_settings.train_scope == "lora_only":
            missing = [
                name
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            if missing:
                raise RuntimeError(
                    "SVDQuant lora_only parameters missing gradients on the first step: "
                    + ", ".join(missing[:5])
                )

    def _release_distillation_controller(self) -> None:
        controller = getattr(self, "_qad_controller", None)
        if controller is None or not hasattr(controller, "_layers_to_loss"):
            return
        layer_pairs = tuple(controller._layers_to_loss)
        clear_captured_outputs(controller)
        mtd.export(controller)
        for student_layer, teacher_layer in layer_pairs:
            for layer in (student_layer, teacher_layer):
                if hasattr(layer, "_intermediate_output"):
                    delattr(layer, "_intermediate_output")
