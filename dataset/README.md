The training / validation files are not included in this repository due to size
and double-blind constraints.

After downloading, please place them as:

- `dataset/train.xlsx`
- `dataset/val.xlsx`

These files are used by:

- `Code/LLM/starcoder2_7b_finetuning.py`
- `Code/LLM/tinylama_fully_retrain.py`

For multimodal training, please additionally prepare:

- `dataset/train.jsonl`
- `dataset/images/`  (all images referenced in `train.jsonl`)

These files are used by:

- `Code/LMM/OpenCLIP_Linear_Projection_Tinyllama.py`
- `Code/LMM/OpenCLIP_Two_MLP_StarCoder2.py`
