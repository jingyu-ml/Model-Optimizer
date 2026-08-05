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

from .modeling import clear_captured_outputs

__all__ = ["QADPipeline", "configure_qad_timestep_sampling"]


_QWEN_IMAGE_FLASH_TIMESTEPS = torch.tensor([1000.0, 900.0, 750.0, 500.0])
_QWEN_IMAGE_FLASH_SIGMAS = torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0])


class _DiscreteTimestepSampler:
    def __init__(
        self,
        timesteps: torch.Tensor,
        sigmas: torch.Tensor,
        default_device: torch.device,
    ) -> None:
        self._cpu_timesteps = timesteps.detach().to(device="cpu", dtype=torch.float32)
        self._cpu_sigmas = sigmas.detach().to(device="cpu", dtype=torch.float32)
        self._default_device = default_device
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
        return sigmas[indices], timesteps[indices], "qwen_image_flash_inference_rungs"


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

    sampler = _DiscreteTimestepSampler(
        timesteps=timesteps,
        sigmas=sigmas[:-1],
        default_device=flow_matching_pipeline.device,
    )
    flow_matching_pipeline.sample_timesteps = sampler.sample_timesteps
    return {
        **timestep_config,
        "scheduler_class": type(scheduler).__name__,
        "timesteps": timesteps.tolist(),
        "sigmas": sigmas.tolist(),
    }


class QADPipeline:
    """Run teacher/student on identical inputs and aggregate ModelOpt KD losses."""

    def __init__(self, flow_matching_pipeline, controller: nn.Module, loss_names: tuple[str, ...]):
        self.flow_matching_pipeline = flow_matching_pipeline
        self.controller = controller
        self.loss_names = loss_names

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
        if check_loss and not bool(torch.isfinite(total.detach()).all()):
            raise FloatingPointError(f"Non-finite QAD loss at step {global_step}.")

        kd_values = [value for key, value in losses.items() if key != "student_loss"]
        if len(kd_values) != len(self.loss_names):
            raise RuntimeError(
                "QAD loss-name mapping is out of sync with ModelOpt's returned losses."
            )
        metrics = {"task_loss": task_loss.detach(), "total_loss": total.detach()}
        metrics.update({name: value.detach() for name, value in zip(self.loss_names, kd_values)})
        return total, metrics

    def clear(self) -> None:
        clear_captured_outputs(self.controller)
