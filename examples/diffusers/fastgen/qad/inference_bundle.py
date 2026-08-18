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

"""Assemble and load inference bundles from FastGen QAD checkpoints.

QAD's consolidated checkpoint contains the complete transformer state, but it
does not contain the pipeline's static components or ``modelopt_state.pth``.
This utility combines it with the exact pre-QAD student bundle recorded in the
checkpoint config. It also records whether the matching inference path is:

* ModelOpt GEMM quantization only; or
* ModelOpt GEMM quantization plus calibrated Attention Grill attention.

By default assembly uses absolute symlinks so every checkpoint looks like a
complete Diffusers pipeline without duplicating tens of gigabytes. Pass
``--materialize copy`` when a physically independent bundle is required.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

_MANIFEST_NAME = "qad_bundle.json"
_CONSOLIDATED_INDEX = "diffusion_pytorch_model.safetensors.index.json"
_CHECKPOINT_PATTERN = re.compile(r"epoch_(\d+)_step_(\d+)$")
_QWEN_IMAGE_FLASH_TIMESTEPS = (1000.0, 900.0, 750.0, 500.0)


def _load_checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"checkpoint has no config.yaml: {checkpoint}")
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint config must be a mapping: {config_path}")
    return config


def _student_bundle(config: dict[str, Any], checkpoint: Path) -> Path:
    model = config.get("model") or {}
    student = Path(str(model.get("pretrained_model_name_or_path", ""))).resolve()
    required = (student / "model_index.json", student / "transformer/modelopt_state.pth")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            f"checkpoint {checkpoint.name} records an incomplete pre-QAD student bundle: "
            + ", ".join(os.fspath(path) for path in missing)
        )
    return student


def _consolidated_files(checkpoint: Path) -> tuple[Path, ...]:
    consolidated = checkpoint / "model/consolidated"
    config_path = consolidated / "config.json"
    index_path = consolidated / _CONSOLIDATED_INDEX
    if not config_path.is_file() or not index_path.is_file():
        raise ValueError(f"checkpoint has no complete consolidated transformer: {checkpoint}")
    try:
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"could not read consolidated index {index_path}: {exc}") from exc
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"consolidated index has no weight map: {index_path}")
    weight_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and name.endswith(".safetensors") for name in weight_names):
        raise ValueError(f"consolidated index references non-safetensors weights: {index_path}")
    weight_paths = tuple(consolidated / name for name in weight_names)
    missing = [path for path in weight_paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(
            "consolidated index references missing/empty weights: "
            + ", ".join(os.fspath(path) for path in missing)
        )
    return (config_path, index_path, *weight_paths)


def _checkpoint_key(path: Path) -> tuple[int, int]:
    match = _CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        return (-1, -1)
    return (int(match.group(1)), int(match.group(2)))


def discover_checkpoints(run_dir: str | Path) -> list[Path]:
    """Return complete consolidated checkpoints in training order."""
    run_dir = Path(run_dir).resolve()
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"run has no checkpoints directory: {checkpoint_root}")
    complete: list[Path] = []
    for checkpoint in checkpoint_root.iterdir():
        if not checkpoint.is_dir() or _CHECKPOINT_PATTERN.fullmatch(checkpoint.name) is None:
            continue
        try:
            _load_checkpoint_config(checkpoint)
            _consolidated_files(checkpoint)
        except ValueError:
            continue
        complete.append(checkpoint)
    return sorted(complete, key=_checkpoint_key)


def _place(source: Path, destination: Path, materialize: str) -> None:
    if materialize == "symlink":
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=False)
    else:
        shutil.copy2(source, destination)


def _attention_metadata(config: dict[str, Any], staging: Path, materialize: str) -> dict[str, Any]:
    qad = config.get("qad") or {}
    attention = qad.get("attention_grill") or {}
    enabled = bool(attention.get("enabled", False))
    if not enabled:
        return {"enabled": False, "provider": "none"}

    recipe = Path(str(attention.get("recipe_path", ""))).resolve()
    calibration = Path(str(attention.get("calibration_path", ""))).resolve()
    required = (recipe, calibration / "manifest.json", calibration / "anchors.safetensors")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise ValueError(
            "Attention Grill checkpoint records missing inference artifacts: "
            + ", ".join(os.fspath(path) for path in missing)
        )

    attention_dir = staging / "qad_attention_grill"
    attention_dir.mkdir()
    _place(recipe, attention_dir / "recipe.yaml", materialize)
    _place(calibration, attention_dir / "calibration", materialize)
    return {
        "enabled": True,
        "provider": "attention_grill",
        "recipe": "qad_attention_grill/recipe.yaml",
        "calibration": "qad_attention_grill/calibration",
        "source_recipe": os.fspath(recipe),
        "source_calibration": os.fspath(calibration),
        "ignore": list(attention.get("ignore") or ()),
        "expected_replaced": int(attention.get("expected_replaced", 0)),
        "expected_ignored": int(attention.get("expected_ignored", 0)),
        "inference_profile": str((qad.get("timestep") or {}).get("schedule", "")),
    }


def assemble_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    materialize: str = "symlink",
) -> Path:
    """Assemble one QAD checkpoint into a complete Diffusers pipeline bundle."""
    if materialize not in {"symlink", "copy"}:
        raise ValueError("materialize must be 'symlink' or 'copy'")
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite inference bundle: {output_dir}")

    config = _load_checkpoint_config(checkpoint)
    student = _student_bundle(config, checkpoint)
    consolidated_files = _consolidated_files(checkpoint)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for source in sorted(student.iterdir()):
            if source.name == "transformer":
                continue
            _place(source, staging / source.name, materialize)

        transformer = staging / "transformer"
        transformer.mkdir()
        replaced_names = {path.name for path in consolidated_files}
        for source in sorted((student / "transformer").iterdir()):
            if source.name in replaced_names or source.suffix == ".safetensors":
                continue
            if source.name == _CONSOLIDATED_INDEX:
                continue
            _place(source, transformer / source.name, materialize)
        for source in consolidated_files:
            _place(source, transformer / source.name, materialize)

        qad = config.get("qad") or {}
        student_config = qad.get("student") or {}
        attention = _attention_metadata(config, staging, materialize)
        manifest = {
            "format": "fastgen_qad_diffusers_bundle",
            "source_run": os.fspath(checkpoint.parent.parent),
            "source_checkpoint": os.fspath(checkpoint),
            "source_student_bundle": os.fspath(student),
            "materialize": materialize,
            "student": {
                "mode": str(student_config.get("mode", "")),
                "train_scope": str(student_config.get("train_scope", "")),
            },
            "gemm_quantization": {
                "enabled": True,
                "provider": "modelopt",
                "state": "transformer/modelopt_state.pth",
            },
            "attention_quantization": attention,
            "consolidated_safetensors": [
                f"transformer/{path.name}"
                for path in consolidated_files
                if path.suffix == ".safetensors"
            ],
        }
        (staging / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def assemble_run(
    run_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    materialize: str = "symlink",
) -> tuple[list[Path], list[Path]]:
    """Assemble every newly discovered complete checkpoint under one QAD run."""
    run_dir = Path(run_dir).resolve()
    output_root = Path(output_root).resolve() if output_root else run_dir / "inference"
    checkpoints = discover_checkpoints(run_dir)
    assembled: list[Path] = []
    skipped: list[Path] = []
    for checkpoint in checkpoints:
        destination = output_root / checkpoint.name
        if destination.exists():
            if (destination / _MANIFEST_NAME).is_file():
                skipped.append(destination)
                continue
            raise FileExistsError(f"existing path is not a QAD inference bundle: {destination}")
        assembled.append(assemble_checkpoint(checkpoint, destination, materialize=materialize))
    return assembled, skipped


def load_qad_pipeline(
    bundle_dir: str | Path,
    *,
    torch_dtype: Any | None = None,
    device: str | Any = "cuda",
    local_files_only: bool = True,
) -> Any:
    """Load ModelOpt GEMM state and optionally install Attention Grill."""
    import torch
    from diffusers import QwenImagePipeline

    import modelopt.torch.opt as mto

    bundle_dir = Path(bundle_dir).resolve()
    manifest_path = bundle_dir / _MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read QAD bundle manifest {manifest_path}: {exc}") from exc
    if manifest.get("format") != "fastgen_qad_diffusers_bundle":
        raise ValueError(f"unsupported QAD bundle format in {manifest_path}")

    mto.enable_huggingface_checkpointing()
    pipe = QwenImagePipeline.from_pretrained(
        bundle_dir,
        torch_dtype=torch_dtype or torch.bfloat16,
        local_files_only=local_files_only,
    ).to(device)
    if not mto.ModeloptStateManager.is_converted(pipe.transformer):
        raise RuntimeError("assembled QAD pipeline did not restore ModelOpt GEMM state")

    attention = manifest["attention_quantization"]
    report = None
    if attention["enabled"]:
        try:
            import attention_grill
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "this QAD bundle requires Attention Grill; add its source_root/src to PYTHONPATH"
            ) from exc
        report = attention_grill.replace(
            pipe.transformer,
            recipe=bundle_dir / attention["recipe"],
            calibration=bundle_dir / attention["calibration"],
            ignore=attention["ignore"],
        )
        if len(report.replaced) != attention["expected_replaced"]:
            raise RuntimeError(
                f"Attention Grill replaced {len(report.replaced)} modules; "
                f"expected {attention['expected_replaced']}"
            )
        if len(report.ignored) != attention["expected_ignored"]:
            raise RuntimeError(
                f"Attention Grill ignored {len(report.ignored)} modules; "
                f"expected {attention['expected_ignored']}"
            )

    pipe.transformer.eval()
    pipe._qad_bundle_manifest = manifest
    pipe._qad_attention_grill_report = report
    return pipe


def _load_bundle_manifest(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / _MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read QAD bundle manifest {manifest_path}: {exc}") from exc
    if manifest.get("format") != "fastgen_qad_diffusers_bundle":
        raise ValueError(f"unsupported QAD bundle format in {manifest_path}")
    return manifest


def _bundle_timestep_range(bundle_dir: Path, manifest: dict[str, Any]) -> tuple[str, int, int, int]:
    checkpoint = Path(str(manifest["source_checkpoint"])).resolve()
    timestep = (_load_checkpoint_config(checkpoint).get("qad") or {}).get("timestep") or {}
    try:
        return (
            str(timestep["schedule"]),
            int(timestep["num_inference_steps"]),
            int(timestep["inference_step_start"]),
            int(timestep["inference_step_end"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bundle does not record a complete timestep range: {bundle_dir}") from exc


def validate_timestep_composition(
    first_bundle: Path,
    second_bundle: Path,
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    first_manifest = _load_bundle_manifest(first_bundle)
    second_manifest = _load_bundle_manifest(second_bundle)
    first_range = _bundle_timestep_range(first_bundle, first_manifest)
    second_range = _bundle_timestep_range(second_bundle, second_manifest)
    first_profile, first_steps, first_start, first_end = first_range
    second_profile, second_steps, second_start, second_end = second_range

    if first_profile != second_profile or first_steps != second_steps:
        raise ValueError(
            "timestep bundles use different inference schedules: "
            f"first={first_range}, second={second_range}"
        )
    if first_start != 0 or first_end != second_start or second_end != first_steps:
        raise ValueError(
            "timestep bundles must form contiguous [0, split) and [split, num_steps) ranges: "
            f"first={first_range}, second={second_range}"
        )
    if first_profile != "qwen_image_flash" or first_steps != 4:
        raise ValueError(
            "timestep-composed QAD inference currently supports only the exact "
            "Qwen-Image-Flash four-step schedule"
        )

    if first_manifest.get("student", {}).get("mode") != second_manifest.get("student", {}).get(
        "mode"
    ):
        raise ValueError("timestep bundles use different student quantization modes")

    attention_fields = (
        "enabled",
        "provider",
        "source_recipe",
        "source_calibration",
        "ignore",
        "expected_replaced",
        "expected_ignored",
    )
    first_attention = first_manifest.get("attention_quantization") or {}
    second_attention = second_manifest.get("attention_quantization") or {}
    if any(first_attention.get(key) != second_attention.get(key) for key in attention_fields):
        raise ValueError("timestep bundles use different attention inference configurations")
    return first_manifest, second_manifest, first_end, first_steps


class TimestepComposedQADPipeline:
    """Route contiguous inference-step ranges to two complete QAD transformers."""

    def __init__(
        self,
        pipe: Any,
        *,
        first_transformer: Any,
        second_transformer: Any,
        switch_after_step: int,
        num_inference_steps: int,
        manifest: dict[str, Any],
    ) -> None:
        self._pipe = pipe
        self._first_transformer = first_transformer
        self._second_transformer = second_transformer
        self._switch_after_step = switch_after_step
        self._num_inference_steps = num_inference_steps
        self._qad_bundle_manifest = manifest
        self._last_timestep_trace: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipe, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        requested_steps = int(kwargs.get("num_inference_steps", 50))
        if requested_steps != self._num_inference_steps or kwargs.get("sigmas") is not None:
            raise ValueError(
                "timestep-composed QAD pipeline requires the native "
                f"{self._num_inference_steps}-step schedule without custom sigmas"
            )
        if float(kwargs.get("true_cfg_scale", 4.0)) != 1.0:
            raise ValueError(
                "timestep-composed Qwen-Image-Flash inference requires true_cfg_scale=1.0"
            )

        user_callback = kwargs.pop("callback_on_step_end", None)
        self._pipe.transformer = self._first_transformer
        trace: list[dict[str, Any]] = []

        def trace_transformer_call(segment: str) -> Any:
            def hook(_module: Any, _inputs: Any) -> None:
                timestep = self._pipe.current_timestep
                if timestep is None:
                    raise RuntimeError("timestep-composed transformer called without a timestep")
                trace.append(
                    {
                        "segment": segment,
                        "timestep": float(timestep.detach().float().cpu().item()),
                    }
                )

            return hook

        def route_after_step(
            diffusion_pipe: Any,
            step_index: int,
            timestep: Any,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            if step_index + 1 == self._switch_after_step:
                diffusion_pipe.transformer = self._second_transformer
            if user_callback is not None:
                return user_callback(diffusion_pipe, step_index, timestep, callback_kwargs)
            return callback_kwargs

        kwargs["callback_on_step_end"] = route_after_step
        first_hook = self._first_transformer.register_forward_pre_hook(
            trace_transformer_call("first")
        )
        second_hook = self._second_transformer.register_forward_pre_hook(
            trace_transformer_call("second")
        )
        try:
            result = self._pipe(*args, **kwargs)
            expected_trace = [
                {
                    "segment": "first" if index < self._switch_after_step else "second",
                    "timestep": timestep,
                }
                for index, timestep in enumerate(_QWEN_IMAGE_FLASH_TIMESTEPS)
            ]
            if trace != expected_trace:
                raise RuntimeError(
                    f"timestep-composed transformer routing mismatch: {trace} != {expected_trace}"
                )
            self._last_timestep_trace = list(trace)
            return result
        finally:
            first_hook.remove()
            second_hook.remove()
            self._pipe.transformer = self._first_transformer


def load_timestep_composed_qad_pipeline(
    first_bundle_dir: str | Path,
    second_bundle_dir: str | Path,
    *,
    torch_dtype: Any | None = None,
    device: str | Any = "cuda",
    local_files_only: bool = True,
) -> TimestepComposedQADPipeline:
    """Load and route two complete QAD transformers by inference step."""
    import torch

    first_bundle = Path(first_bundle_dir).resolve()
    second_bundle = Path(second_bundle_dir).resolve()
    first_manifest, second_manifest, switch_after_step, num_steps = validate_timestep_composition(
        first_bundle, second_bundle
    )
    first_pipe = load_qad_pipeline(
        first_bundle,
        torch_dtype=torch_dtype,
        device=device,
        local_files_only=local_files_only,
    )
    second_pipe = load_qad_pipeline(
        second_bundle,
        torch_dtype=torch_dtype,
        device=device,
        local_files_only=local_files_only,
    )
    first_transformer = first_pipe.transformer
    second_transformer = second_pipe.transformer
    second_pipe.transformer = None
    del second_pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    composite_manifest = {
        "format": "fastgen_qad_timestep_composition",
        "inference_profile": "qwen_image_flash",
        "num_inference_steps": num_steps,
        "switch_after_step": switch_after_step,
        "segments": [
            {
                "start": 0,
                "end": switch_after_step,
                "bundle": os.fspath(first_bundle),
                "qad_bundle": first_manifest,
            },
            {
                "start": switch_after_step,
                "end": num_steps,
                "bundle": os.fspath(second_bundle),
                "qad_bundle": second_manifest,
            },
        ],
        "resident_transformers": 2,
    }
    return TimestepComposedQADPipeline(
        first_pipe,
        first_transformer=first_transformer,
        second_transformer=second_transformer,
        switch_after_step=switch_after_step,
        num_inference_steps=num_steps,
        manifest=composite_manifest,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="QAD run directory containing checkpoints/")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="output root; defaults to RUN_DIR/inference",
    )
    parser.add_argument(
        "--materialize",
        choices=("symlink", "copy"),
        default="symlink",
        help="reuse source files with symlinks (default) or physically copy them",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoints = discover_checkpoints(args.run_dir)
    print(f"Found {len(checkpoints)} complete consolidated checkpoints in {args.run_dir}")
    for checkpoint in checkpoints:
        safetensors = [
            path for path in _consolidated_files(checkpoint) if path.suffix == ".safetensors"
        ]
        config = _load_checkpoint_config(checkpoint)
        attention = bool(((config.get("qad") or {}).get("attention_grill") or {}).get("enabled"))
        kind = "ModelOpt GEMM + Attention Grill" if attention else "ModelOpt GEMM only"
        print(f"  {checkpoint.name}: {len(safetensors)} safetensors, {kind}")
    assembled, skipped = assemble_run(
        args.run_dir,
        output_root=args.output_root,
        materialize=args.materialize,
    )
    for path in assembled:
        print(f"Assembled {path}")
    for path in skipped:
        print(f"Skipped existing {path}")
    print(f"Done: assembled={len(assembled)} skipped={len(skipped)}")


if __name__ == "__main__":
    main()
