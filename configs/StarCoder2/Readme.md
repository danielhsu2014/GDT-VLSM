# TinyLlama-GDT Checkpoint

This directory contains the TinyLlama checkpoint fine-tuned on the GDT-VLSM
FlexScript dataset.

The following files are already tracked in the repository:

- `config.json`
- `generation_config.json`
- `tokenizer.model`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `trainer_state.json`
- `training_args.bin`
- `scheduler.pt` (only needed if you want to resume training)

The actual model weights are **not** included in the repository due to size
and double-blind constraints.

After downloading the TinyLlama checkpoint, please place the weight file in
this directory with the following name:

- `model.safetensors`

With `model.safetensors` in place, the following scripts will load this
checkpoint automatically:

- `Code/LMM/OpenCLIP_Linear_Projection_Tinyllama.py`
- any TinyLlama inference or training script that uses
  `configs/TinyLlama-1.1b` as `model_name_or_path`.

Optional training state files (not required for inference):

- `optimizer.pt`
- `rng_state_0.pth` ... `rng_state_7.pth`
