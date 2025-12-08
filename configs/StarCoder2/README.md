# StarCoder2-GDT LoRA Adapter

This directory stores the LoRA adapter and tokenizer files for the
`bigcode/starcoder2-7b` model fine-tuned on the GDT-VLSM FlexScript dataset.

The following files are already tracked in the repository:

- `adapter_config.json`
- `merges.txt`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.json`
- `trainer_state.json`
- `training_args.bin`

The LoRA weight file is **not** included in the repository due to size and
double-blind constraints.

After downloading the adapter checkpoint, please place the weight file in this
directory with the following name:

- `adapter_model.safetensors`

At run time, the base model is loaded from Hugging Face Hub:

- `bigcode/starcoder2-7b`

The following scripts expect this directory structure:

- `Code/LMM/OpenCLIP_Two_MLP_StarCoder2.py`
- any other script that calls
  `PeftModel.from_pretrained(base_model, "configs/StarCoder2")`.

Optional training state files (not required for inference):

- `optimizer.pt`
- `scheduler.pt`
- `rng_state_0.pth` ... `rng_state_7.pth`
