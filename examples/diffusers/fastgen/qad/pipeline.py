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

"""QAD loss pipeline layered on AutoModel's flow-matching input preparation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from torch import nn

from .modeling import DistillationLossLayout, clear_captured_outputs

__all__ = ["QADPipeline", "configure_qad_timestep_sampling"]


_QWEN_IMAGE_FLASH_TIMESTEPS = torch.tensor([1000.0, 900.0, 750.0, 500.0])
_QWEN_IMAGE_FLASH_SIGMAS = torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0])


class _DiscreteTimestepSampler:
    def __init__(
        self,
        timesteps: torch.Tensor,
        sigmas: torch.Tensor,
        default_device: torch.device,
        sampling_method: str,
    ) -> None:
        self._cpu_timesteps = timesteps.detach().to(device="cpu", dtype=torch.float32)
        self._cpu_sigmas = sigmas.detach().to(device="cpu", dtype=torch.float32)
        self._default_device = default_device
        self._sampling_method = sampling_method
        self._device_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        target_device = torch.device(device) if device is not None else self._default_device
        if target_device not in self._device_cache:
            self._device_cache[target_device] = (
                self._cpu_sigmas.to(target_device),
                self._cpu_timesteps.to(target_device),
            )
        sigmas, timesteps = self._device_cache[target_device]
        indices = torch.randint(len(timesteps), (batch_size,), device=target_device)
        return sigmas[indices], timesteps[indices], self._sampling_method


class _ContinuousTimestepRangeSampler:
    def __init__(
        self,
        *,
        sigma_min: float,
        sigma_max: float,
        num_train_timesteps: int,
        default_device: torch.device,
        sampling_method: str,
    ) -> None:
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._num_train_timesteps = num_train_timesteps
        self._default_device = default_device
        self._sampling_method = sampling_method

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        target_device = torch.device(device) if device is not None else self._default_device
        sigmas = torch.rand(batch_size, device=target_device, dtype=torch.float32)
        sigmas = self._sigma_min + sigmas * (self._sigma_max - self._sigma_min)
        timesteps = sigmas * self._num_train_timesteps
        return sigmas, timesteps, self._sampling_method


def _validate_scheduler_config(
    config: dict[str, Any],
    *,
    schedule: str,
) -> None:
    if config.get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError(
            f"{schedule} requires a FlowMatchEulerDiscreteScheduler, "
            f"found {config.get('_class_name')}."
        )
    if int(config["num_train_timesteps"]) != 1000:
        raise ValueError(
            f"{schedule} requires scheduler.num_train_timesteps=1000, "
            f"found {config['num_train_timesteps']}."
        )

    if schedule == "qwen_image_flash":
        actual = {
            "shift": float(config["shift"]),
            "shift_terminal": config["shift_terminal"],
            "use_dynamic_shifting": bool(config["use_dynamic_shifting"]),
            "invert_sigmas": bool(config["invert_sigmas"]),
        }
        expected = {
            "shift": 3.0,
            "shift_terminal": None,
            "use_dynamic_shifting": False,
            "invert_sigmas": False,
        }
    else:
        shift_terminal = config["shift_terminal"]
        actual = {
            "shift": float(config["shift"]),
            "shift_terminal": (None if shift_terminal is None else float(shift_terminal)),
            "use_dynamic_shifting": bool(config["use_dynamic_shifting"]),
            "invert_sigmas": bool(config["invert_sigmas"]),
        }
        expected = {
            "shift": 1.0,
            "shift_terminal": 0.02,
            "use_dynamic_shifting": True,
            "invert_sigmas": False,
        }
    if actual != expected:
        raise ValueError(
            f"qad.timestep.schedule={schedule} does not match the student scheduler: "
            f"expected {expected}, found {actual}."
        )


def configure_qad_timestep_sampling(
    flow_matching_pipeline,
    *,
    student_model_name_or_path: str,
    timestep_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the student scheduler and configure QAD timestep sampling."""
    schedule = timestep_config["schedule"]
    pipeline_config = DiffusionPipeline.load_config(student_model_name_or_path)
    expected_component = ["diffusers", "FlowMatchEulerDiscreteScheduler"]
    if pipeline_config.get("scheduler") != expected_component:
        raise ValueError(
            "The student pipeline must map its scheduler component to "
            f"{expected_component}, found {pipeline_config.get('scheduler')}."
        )
    scheduler_config = FlowMatchEulerDiscreteScheduler.load_config(
        student_model_name_or_path,
        subfolder="scheduler",
    )
    _validate_scheduler_config(scheduler_config, schedule=schedule)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    if int(flow_matching_pipeline.num_train_timesteps) != int(scheduler.config.num_train_timesteps):
        raise ValueError(
            "flow_matching.num_train_timesteps must match the student scheduler: "
            f"{flow_matching_pipeline.num_train_timesteps} != "
            f"{scheduler.config.num_train_timesteps}."
        )

    if schedule == "qwen_image":
        inference_step_range = timestep_config.get("inference_step_range")
        if inference_step_range is not None:
            num_inference_steps = int(inference_step_range["num_inference_steps"])
            start_step = int(inference_step_range["start"])
            end_step = int(inference_step_range["end"])
            image_seq_len = int(inference_step_range["image_seq_len"])

            base_seq_len = int(scheduler.config.base_image_seq_len)
            max_seq_len = int(scheduler.config.max_image_seq_len)
            base_shift = float(scheduler.config.base_shift)
            max_shift = float(scheduler.config.max_shift)
            shift_slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            shift_intercept = base_shift - shift_slope * base_seq_len
            mu = image_seq_len * shift_slope + shift_intercept

            # Match QwenImagePipeline: N evaluation sigmas from 1 to 1/N,
            # followed by the scheduler's terminal sigma. A [start, end)
            # denoising-step slice is therefore bounded by sigmas[start] and
            # sigmas[end].
            raw_sigmas = np.linspace(
                1.0,
                1.0 / num_inference_steps,
                num_inference_steps,
            ).tolist()
            scheduler.set_timesteps(sigmas=raw_sigmas, mu=mu)
            inference_sigmas = scheduler.sigmas.detach().to(device="cpu", dtype=torch.float32)
            sigma_max = float(inference_sigmas[start_step].item())
            sigma_min = float(inference_sigmas[end_step].item())
            if not 0.0 <= sigma_min < sigma_max <= 1.0:
                raise ValueError(
                    "Qwen-Image inference-step range produced invalid sigma bounds: "
                    f"[{sigma_min}, {sigma_max}]."
                )

            sampling_method = f"qwen_image_inference_steps_{start_step}_to_{end_step}_uniform"
            sampler = _ContinuousTimestepRangeSampler(
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                num_train_timesteps=int(scheduler.config.num_train_timesteps),
                default_device=flow_matching_pipeline.device,
                sampling_method=sampling_method,
            )
            flow_matching_pipeline.sample_timesteps = sampler.sample_timesteps
            return {
                **timestep_config,
                "scheduler_class": type(scheduler).__name__,
                "dynamic_shift_mu": mu,
                "sampled_sigma_min": sigma_min,
                "sampled_sigma_max": sigma_max,
                "sampled_timestep_min": sigma_min * scheduler.config.num_train_timesteps,
                "sampled_timestep_max": sigma_max * scheduler.config.num_train_timesteps,
                "sampling_method": sampling_method,
            }
        return {
            **timestep_config,
            "scheduler_class": type(scheduler).__name__,
        }

    # QwenImagePipeline passes these raw sigmas to the scheduler instead of
    # calling scheduler.set_timesteps(4). The static shift-3 scheduler then
    # produces the four timesteps at which the Flash transformer is evaluated.
    raw_sigmas = np.linspace(1.0, 1.0 / 4, 4).tolist()
    scheduler.set_timesteps(sigmas=raw_sigmas)
    timesteps = scheduler.timesteps.detach().to(device="cpu", dtype=torch.float32)
    sigmas = scheduler.sigmas.detach().to(device="cpu", dtype=torch.float32)
    if not torch.allclose(
        timesteps,
        _QWEN_IMAGE_FLASH_TIMESTEPS,
        rtol=0.0,
        atol=1e-4,
    ) or not torch.allclose(
        sigmas,
        _QWEN_IMAGE_FLASH_SIGMAS,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "The student scheduler did not reproduce the Qwen-Image-Flash 4-step "
            f"schedule: timesteps={timesteps.tolist()} sigmas={sigmas.tolist()}."
        )

    inference_step_range = timestep_config.get("inference_step_range")
    if inference_step_range is None:
        start_step, end_step = 0, len(timesteps)
    else:
        start_step = int(inference_step_range["start"])
        end_step = int(inference_step_range["end"])
    selected_timesteps = timesteps[start_step:end_step]
    selected_sigmas = sigmas[start_step:end_step]
    sampling_method = (
        "qwen_image_flash_inference_rungs"
        if (start_step, end_step) == (0, len(timesteps))
        else f"qwen_image_flash_inference_steps_{start_step}_to_{end_step}_discrete"
    )
    sampler = _DiscreteTimestepSampler(
        timesteps=selected_timesteps,
        sigmas=selected_sigmas,
        default_device=flow_matching_pipeline.device,
        sampling_method=sampling_method,
    )
    flow_matching_pipeline.sample_timesteps = sampler.sample_timesteps
    return {
        **timestep_config,
        "scheduler_class": type(scheduler).__name__,
        "timesteps": timesteps.tolist(),
        "sigmas": sigmas.tolist(),
        "sampled_step_indices": list(range(start_step, end_step)),
        "sampled_timesteps": selected_timesteps.tolist(),
        "sampled_sigmas": selected_sigmas.tolist(),
        "sampling_method": sampling_method,
    }


class QADPipeline:
    """Run teacher/student on identical inputs and aggregate ModelOpt KD losses."""

    def __init__(
        self,
        flow_matching_pipeline,
        controller: nn.Module,
        loss_layout: DistillationLossLayout,
    ):
        self.flow_matching_pipeline = flow_matching_pipeline
        self.controller = controller
        self.loss_layout = loss_layout

    def step(
        self,
        *,
        batch: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        global_step: int,
        check_loss: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clear_captured_outputs(self.controller)
        _, task_loss, _, _ = self.flow_matching_pipeline.step(
            model=self.controller,
            batch=batch,
            device=device,
            dtype=dtype,
            global_step=global_step,
            collect_metrics=False,
            # The flow target is optional in QAD. Validate the actual combined loss below.
            check_loss=False,
        )
        losses = self.controller.compute_kd_loss(
            student_loss=task_loss,
            skip_balancer=True,
        )
        total = self.controller.loss_balancer(losses)
        if check_loss:
            finite = torch.isfinite(total.detach()).all().to(dtype=torch.int32)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
            if not bool(finite.item()):
                raise FloatingPointError(
                    f"Non-finite QAD loss on at least one rank at step {global_step}."
                )

        kd_values = [value for key, value in losses.items() if key != "student_loss"]
        if len(kd_values) != len(self.loss_layout.names):
            raise RuntimeError(
                "QAD loss-name mapping is out of sync with ModelOpt's returned losses."
            )
        metrics = {"task_loss": task_loss.detach(), "total_loss": total.detach()}
        blockwise_positions = set(self.loss_layout.blockwise_positions)
        for position, (name, value) in enumerate(zip(self.loss_layout.names, kd_values)):
            if position not in blockwise_positions or self.loss_layout.log_per_block:
                metrics[name] = value.detach()

        if blockwise_positions:
            metric_name = self.loss_layout.blockwise_metric_name
            if metric_name is None:
                raise RuntimeError("QAD blockwise losses have no aggregate metric name.")
            metrics[metric_name] = torch.stack(
                [kd_values[position].detach() for position in self.loss_layout.blockwise_positions]
            ).mean()
        return total, metrics

    def clear(self) -> None:
        clear_captured_outputs(self.controller)
