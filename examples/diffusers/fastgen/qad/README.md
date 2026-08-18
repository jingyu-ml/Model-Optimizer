# FastGen Quantization-Aware Distillation

This example trains a quantized diffusion student against a frozen, BF16
Diffusers teacher with ModelOpt's distillation API. It is a standalone FastGen
recipe: it does not run DMD2, create a fake-score model, or add a GAN/EMA
training phase.

The initial Qwen-Image recipe uses the official `Qwen/Qwen-Image` Diffusers
checkpoint as the teacher. Set `qad.teacher_model_name_or_path` to
`nvidia/Qwen-Image-Flash` when the four-step DMD2-trained Qwen-Image checkpoint
should be the teacher instead. Both follow the standard Diffusers checkpoint
interface. QAD intentionally does not interpret FastGen/DMD2 intermediate
checkpoint sidecars or standalone transformer safetensors as teacher inputs.

Every training micro-batch sends the same noisy latent, timestep, prompt
conditioning, and guidance inputs to the teacher and student. Timestep sampling
follows the quantized student's deployment schedule, independently of which BF16
teacher is selected.

## Timestep schedules

`qad.timestep.schedule` is separate from the student's quantization mode:

- `qwen_image_flash` is the default in the provided configs. The recipe reads the
  scheduler from the complete student bundle and reproduces QwenImagePipeline's
  four-step construction. It uniformly samples only model timesteps
  `[1000, 900, 750, 500]`, corresponding to shifted sigmas
  `[1.0, 0.9, 0.75, 0.5]`. The terminal sigma `0.0` is validated but never
  sampled because the transformer is not evaluated there. A half-open
  `[inference_step_start, inference_step_end)` range can select a strict subset
  of these discrete calls: `[0,1)` selects only `t=1000`, while `[1,4)` samples
  only `t=[900,750,500]`. A dynamic-shift original Qwen-Image scheduler is
  rejected in this mode.
- `qwen_image` preserves full-range flow-matching sampling for an original
  Qwen-Image student. Set `flow_matching.timestep_sampling` to `logit_normal` or
  `uniform`. Both use AutoModel's configured flow shift; set
  `flow_matching.use_sigma_noise=false` for uniform sampling directly in sigma
  space. A static shift-3 Flash scheduler is rejected in this mode.

  It can also uniformly sample a continuous slice of the model's official
  inference trajectory. Configure the half-open denoising-step range
  `[inference_step_start, inference_step_end)`, the inference step count, and
  the packed image sequence length. The recipe reads the dynamic-shift settings
  from the student bundle and converts those step boundaries to model sigma and
  timestep bounds; it does not clamp samples from a full-range distribution.
  This range mode requires `flow_matching.timestep_sampling=uniform`.

  For 1024x1024 Qwen-Image, `image_seq_len=4096`. With 50 inference steps,
  `[0, 5)` resolves to approximately `t=[946.335, 1000]`, and `[5, 50)` to
  `t=[0, 946.335]` on the `[0, 1000]` training scale:

  ```yaml
  qad:
    timestep:
      schedule: qwen_image
      num_inference_steps: 50
      inference_step_start: 0
      inference_step_end: 5
      image_seq_len: 4096
  flow_matching:
    timestep_sampling: uniform
  ```

The Flash path intentionally mirrors QwenImagePipeline's raw-sigma construction;
calling `scheduler.set_timesteps(4)` directly produces a different schedule.
Both paths currently form inputs by forward-noising real latents at the sampled
timestep. Trajectory-state rollout is a separate extension.

## Supported students

The `qad.student.mode` field selects one of two bundle validation contracts. In
both cases, `model.pretrained_model_name_or_path` is the only student artifact
path: it points to a complete, calibrated Diffusers pipeline written by
`quantize.py --output-bundle`. QAD restores the pipeline's weights and
component-local ModelOpt state together before FSDP; it does not accept a second
quantizer-state or transformer-checkpoint path.

### Regular NVFP4

Set `qad.student.mode=nvfp4` and point
`model.pretrained_model_name_or_path` at a regular NVFP4 training bundle. The
bundle includes the calibrated weights and ModelOpt quantizer topology/state.
This mode trains all student parameters, so its only valid `train_scope` is
`all`.

Use [`configs/qwen_image_nvfp4.yaml`](configs/qwen_image_nvfp4.yaml) as the
starting configuration.

### NVFP4 SVDQuant with Hugging Face PEFT

Set `qad.student.mode=nvfp4_svdquant` and point
`model.pretrained_model_name_or_path` at a user-prepared, ModelOpt-enabled Diffusers
training bundle. The bundle must contain the complete SVDQuant student:

- a DiffusionPipeline root with `model_index.json` (not only a standalone
  transformer `save_pretrained` directory);
- the ModelOpt topology and quantizer state;
- the residual weights produced by SVDQuant calibration;
- the Hugging Face PEFT A/B factors for the SVDQuant low-rank branch; and
- for magnitude-enabled bundles, the zero-initialized per-output-channel magnitude delta.

For the standard Diffusers layout, the transformer files and ModelOpt sidecar
are under `transformer/`, including `transformer/modelopt_state.pth`. The path
given to QAD is the parent DiffusionPipeline directory.

A standalone weight-free NVFP4 quantizer-state file is not a QAD student bundle.
This is especially important for SVDQuant: calibration subtracts the low-rank
branch from the original weight, so both the resulting residual weight and the
PEFT factors are required. Deployment artifacts are not training bundles and
must not be used here.

The SVDQuant topology is restored before FSDP and before optimizer construction.
`qad.student.train_scope=all` is the default and trains both the residual/base
parameters and the PEFT factors. Set it to `lora_only` to freeze every student
parameter except the SVDQuant PEFT A/B factors and, when present, its magnitude
delta. In both scopes, `pre_quant_scale` remains ModelOpt buffer state and is
never placed in the optimizer.

Use [`configs/qwen_image_svdquant_nvfp4.yaml`](configs/qwen_image_svdquant_nvfp4.yaml)
as the starting configuration.

QAD is restore-only in both modes. It does not calibrate a student during
distributed training.

## Assemble inference bundles

Each training checkpoint keeps FSDP/DCP shards for resume and writes an
inference-ready transformer under `model/consolidated/`. The consolidated
directory is not a complete pipeline and does not contain
`modelopt_state.pth`. Assemble every complete checkpoint with the exact pre-QAD
student bundle recorded in its `config.yaml`:

```bash
python examples/diffusers/fastgen/qad/inference_bundle.py \
  /path/to/qad_run
```

This creates `/path/to/qad_run/inference/epoch_N_step_M`. The default
`--materialize symlink` mode links the static pipeline components, the original
student's `transformer/modelopt_state.pth`, and the checkpoint's consolidated
weights/index/config instead of duplicating them. Use `--materialize copy` only
when a physically independent bundle is required. Re-running the command skips
already assembled bundles and adds newly completed checkpoints.

To update every run immediately below one training output root, use:

```bash
bash examples/diffusers/fastgen/qad/assemble_all_inference_bundles.sh \
  /path/to/qad_training/output
```

Every output contains `qad_bundle.json`. GEMM-only experiments record
`attention_quantization.provider=none`. Experiments trained with quantized
attention record the exact Attention Grill recipe, calibration artifact, ignore
scope, and expected replacement counts. Use the QAD loader so this distinction
is honored automatically:

```python
from qad.inference_bundle import load_qad_pipeline

pipe = load_qad_pipeline("/path/to/qad_run/inference/epoch_0_step_1499")
```

The loader first restores the ModelOpt GEMM/SVDQuant topology and QAD weights.
For an Attention Grill bundle it then installs the recorded calibrated attention
replacement. A plain `QwenImagePipeline.from_pretrained()` call restores only
the ModelOpt portion and is therefore not the complete inference path for those
experiments.

## Prepare a student bundle

Patch Diffusers ModelMixin support and save the complete pipeline through the
quantization entry point. `quantize.py` does this automatically before model
load and calls `pipe.save_pretrained(output_bundle)` after calibration. For
example:

```bash
# Regular NVFP4
python examples/diffusers/quantization/quantize.py \
  --model qwen-image \
  --override-model-path /path/to/Qwen-Image \
  --model-dtype BFloat16 \
  --format fp4 \
  --quant-algo max \
  --block-size 16 \
  --batch-size 1 \
  --calib-size 32 \
  --n-steps 50 \
  --extra-param true_cfg_scale=4.0 \
  --extra-param "negative_prompt= " \
  --output-bundle /path/to/Qwen-Image-NVFP4-Calib32

# NVFP4 SVDQuant, rank 32
python examples/diffusers/quantization/quantize.py \
  --model qwen-image \
  --override-model-path /path/to/Qwen-Image \
  --model-dtype BFloat16 \
  --format fp4 \
  --quant-algo svdquant \
  --lowrank 32 \
  --block-size 16 \
  --batch-size 1 \
  --calib-size 32 \
  --n-steps 50 \
  --extra-param true_cfg_scale=4.0 \
  --extra-param "negative_prompt= " \
  --output-bundle /path/to/Qwen-Image-NVFP4-SVDQuant-Calib32
```

The saved root must contain `model_index.json`; the converted transformer must
contain `transformer/modelopt_state.pth`. For Qwen-Image-Flash, use `--n-steps 4`
and `--extra-param true_cfg_scale=1.0`; omit `negative_prompt`. Standard output
includes ModelOpt's full quantizer summary; capture it with `tee` and retain that
log with the bundle.

## Distillation losses

Output distillation is always MSE. The canonical setting is:

```yaml
qad:
  output_loss:
    type: mse
    weight: 1.0
  task_loss:
    weight: 0.0
```

With layerwise distillation disabled, `output_loss.weight: 1.0` and
`task_loss.weight: 0.0` optimize pure teacher-output MSE. All top-level
coefficients are independent and additive; enabling blockwise loss does not
silently renormalize the final-output coefficient.

The provided configs automatically target every student transformer block that
contains at least one enabled ModelOpt weight quantizer:

```yaml
qad:
  layerwise:
    enabled: true
    selection: quantized_blocks
    type: cosine              # cosine is the cross-model default; mse remains available
    weight: 3.0
    reduction: mean
    streams:
      - selector: encoder_hidden_states
        weight: 0.2
      - selector: hidden_states
        weight: 0.8
    log_per_block: false
```

For every selected block `i`, the Qwen text and image outputs are reduced
independently before weighting:

```text
L_block_i = 0.2 * L_text_i + 0.8 * L_image_i
L_block = mean_i(L_block_i)
L_total = output_weight * L_output + layerwise_weight * L_block
          + task_weight * L_task
```

MSE averages all elements in each stream. Cosine computes `1 - cosine` along
the hidden dimension and then averages batch/token positions. Because each
stream is reduced independently, the larger image tensor does not implicitly
receive more weight; `0.8` is an explicit image-priority coefficient. Stream
weights must sum to `1.0`.

For an original Qwen-Image teacher and a DMD2-trained Flash student, intermediate
residual magnitudes are not aligned: a smoke run measured plain block MSE around
`1.49e13` while final-output MSE was `0.052`. The provided configs therefore use
scale-invariant cosine loss. With the measured cosine block mean of `0.0308`,
`output_loss.weight=1.0` and `layerwise.weight=3.0` make blockwise supervision
slightly dominant (roughly 55% of the scalar objective in that sample). Plain
MSE remains useful when teacher and student representations are already aligned,
such as a BF16 Flash teacher and its directly quantized Flash student.

Block selection is resolved from the restored live student before FSDP and is
not hard-coded to a Qwen layer range. For the current Qwen-Image NVFP4 recipes,
this selects blocks 2 through 57. The student and teacher must expose the same
number of transformer blocks. The layerwise group weight is divided by the
number of selected blocks, so adding 56 hooks does not multiply its objective
scale by 56.

Set `log_per_block: true` for a short verification run; normal training logs
only the block mean. The older exact-module `pairs` form remains supported and
is mutually exclusive with `selection: quantized_blocks`; each explicit pair
uses its own `weight` rather than the block-group weight shown above.

The recipe logs flow-matching loss, output MSE, aggregate blockwise loss, and
the final combined loss separately. Layer hooks retain activations and increase
memory use, especially when activation checkpointing is enabled.

## Configuration and launch

The entry point is `examples/diffusers/fastgen/qad/finetune.py`. It uses the
same YAML plus dotted-command-line override convention as the other FastGen
recipes:

```bash
torchrun --nproc-per-node=4 \
  examples/diffusers/fastgen/qad/finetune.py \
  --config examples/diffusers/fastgen/qad/configs/qwen_image_svdquant_nvfp4.yaml \
  --fsdp.dp_size=4 \
  --model.pretrained_model_name_or_path=/path/to/qwen-image-nvfp4-svdquant-training-bundle \
  --data.dataloader.cache_dir=/path/to/qwen_image_1024p \
  --checkpoint.checkpoint_dir=/path/to/qad/checkpoints
```

Cluster launchers can keep the established `CONFIG`, `RUN_ID`, and
`EXTRA_ARGS` interface. For example:

```bash
EXTRA_ARGS="--step_scheduler.max_steps=50000 \
--step_scheduler.ckpt_every_steps=1000 \
--step_scheduler.num_epochs=200 \
--step_scheduler.global_batch_size=64 \
--optim.learning_rate=2e-6 \
--lr_scheduler.min_lr=2e-6 \
--fsdp.dp_size=64 \
--qad.teacher_model_name_or_path=Qwen/Qwen-Image \
--qad.timestep.schedule=qwen_image_flash \
--qad.output_loss.weight=1.0 \
--qad.task_loss.weight=0.0 \
--qad.layerwise.type=cosine \
--qad.layerwise.weight=3.0 \
--qad.student.mode=nvfp4_svdquant \
--model.pretrained_model_name_or_path=/path/to/qwen-image-nvfp4-svdquant-training-bundle \
--qad.student.train_scope=all \
--data.dataloader.cache_dir=/path/to/qwen_image_1024p" \
CONFIG=examples/diffusers/fastgen/qad/configs/qwen_image_svdquant_nvfp4.yaml \
RUN_ID=qad_qwen_image_svdquant_nvfp4_16n \
NODES=16 \
GPUS_PER_NODE=4 \
TIME=05:00:00 \
PARTITION=batch \
bash /path/to/experiments/qad_qwen_image/launch.sh
```

The launcher must invoke `examples/diffusers/fastgen/qad/finetune.py`.
Pointing the existing DMD2 launcher at a QAD YAML is not sufficient when that
launcher still hard-codes `dmd2_finetune.py`.

Attention Grill is optional and is enabled only when QAD is given an external
recipe and its matching calibrated artifact. The launch environment contains
no DMD2 timestep, fake-score, discriminator, negative-prompt, GAN, or EMA
settings. QAD currently requires `fsdp.tp_size=1`, `fsdp.cp_size=1`, and
`fsdp.pp_size=1`; data parallelism is controlled through `fsdp.dp_size`.

## Restore and checkpoint invariants

On a fresh run the recipe restores the complete student first, constructs its
final ModelOpt/PEFT topology, applies FSDP, builds the optimizer from the selected
training scope, and only then creates the frozen teacher and distillation
controller. On resume, the same immutable student source reconstructs the
topology before the QAD checkpoint is loaded.

The teacher and the transient ModelOpt distillation controller are not training
checkpoint payloads. Checkpoints contain the student state required by the
selected training scope together with optimizer, scheduler, dataloader, RNG, and
global-step state. Resolved dotted CLI overrides are materialized into the saved
`config.yaml`. Resume validates the student bundle, quantization mode, train
scope, teacher, and loss configuration before loading optimizer shards; do not
change them while resuming an existing run.

The provided YAML files now enable blockwise loss by default. To resume an older
output-only checkpoint with the same objective, explicitly pass
`--qad.layerwise.enabled=false`; otherwise the resume signature intentionally
rejects the changed loss contract.
