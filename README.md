# Generative Digital Twins: Vision-Language Simulation Models for Executable Industrial Systems

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://danielhsu2014.github.io/GDT-VLSM-project/)
[![Paper](https://img.shields.io/badge/Paper-PDF-4a4a4a)](https://arxiv.org/pdf/2512.20387)
[![arXiv](https://img.shields.io/badge/arXiv-2512.20387-b31b1b.svg)](https://arxiv.org/abs/2512.20387)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2512.20387)
[![Demo](https://img.shields.io/badge/Gradio-Demo-orange)](https://baker-organizations-bool-sheffield.trycloudflare.com/)

</div>

The goal of this codebase is to reproduce the Vision-Language Simulation Models
(VLSM) used for generating executable FlexScript from text and layout sketches.

---

## Repository Structure

```text
GDT-VLSM/
├── Code/
│   ├── LLM/
│   │   ├── starcoder2_7b_finetuning.py          # QLoRA fine-tuning for StarCoder2-7B
│   │   └── tinylama_fully_retrain.py            # Full-parameter retraining for TinyLLaMA-1.1B
│   └── LMM/
│       ├── OpenCLIP_Linear_Projection_Tinyllama.py   # VLSM-1.1B (OpenCLIP + Linear + TinyLLaMA)
│       └── OpenCLIP_Two_MLP_StarCoder2.py            # VLSM-7B   (OpenCLIP + 2-layer MLP + StarCoder2)
├── configs/
│   ├── StarCoder2/          # LoRA adapter config + tokenizer for StarCoder2-7B
│   └── TinyLlama-1.1b/      # TinyLLaMA-1.1B config + tokenizer
├── dataset/                 # Dataset placeholders (see dataset/README.md)
├── dockerfiles/
│   ├── Dockerfile_StarCoder2    # CUDA 11.8 environment for StarCoder2-7B
│   └── Dockerfile_TinyLlama     # CUDA 11.8 environment for TinyLLaMA-1.1B
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt         # Python dependencies
```

---

## Dependencies & Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

(Optional but recommended) log in to Hugging Face Hub to avoid anonymous
rate limiting:

```bash
huggingface-cli login
```

---

## Datasets

Raw data is **not included** in this repository due to size and double-blind
constraints.

After downloading, place the files under `dataset/` as:

```text
dataset/
  train.xlsx
  val.xlsx
  train.jsonl
  images/        # all images referenced in train.jsonl
```

A short summary is given in `dataset/README.md`.

---

## Training Scripts

| Script | Path | Description |
|--------|------|-------------|
| `starcoder2_7b_finetuning.py` | `Code/LLM/` | QLoRA fine-tuning of **StarCoder2-7B** on text-only GDT-120K. Uses 4-bit quantization and PEFT adapters. |
| `tinylama_fully_retrain.py` | `Code/LLM/` | Full-parameter retraining of **TinyLLaMA-1.1B** on text-only GDT-120K. |
| `OpenCLIP_Linear_Projection_Tinyllama.py` | `Code/LMM/` | Multimodal training of **VLSM-1.1B** (OpenCLIP vision encoder + linear projection connector + TinyLLaMA-1.1B). |
| `OpenCLIP_Two_MLP_StarCoder2.py` | `Code/LMM/` | Multimodal training of **VLSM-7B** (OpenCLIP vision encoder + two-layer MLP connector + StarCoder2-7B + LoRA). |

All scripts follow the standard Hugging Face `from_pretrained` interface and
can be launched via `python`, `torchrun`, or `accelerate` depending on your
hardware.

---

## 1. Text-only Fine-tuning

### StarCoder2-7B (QLoRA)

- Base model: `bigcode/starcoder2-7b` (downloaded automatically from HF).
- Required files:
  - `dataset/train.xlsx`
  - `dataset/val.xlsx`

Run:

```bash
python Code/LLM/starcoder2_7b_finetuning.py
```

This script:

- Loads `bigcode/starcoder2-7b` in 4-bit using `BitsAndBytesConfig`.
- Applies QLoRA to a subset of attention / MLP layers.
- Writes checkpoints under `results/` (created automatically, Git-ignored).

### TinyLLaMA-1.1B (full retrain)

- Base model: `mesolitica/tinyllama-1.1b-4096-fpf` (downloaded from HF).
- Required files:
  - `dataset/train.xlsx`
  - `dataset/val.xlsx`

Run:

```bash
python Code/LLM/tinylama_fully_retrain.py
```

This script:

- Loads the TinyLLaMA config and initializes a fresh `LlamaForCausalLM`.
- Trains all parameters on the FlexScript corpus.
- Writes checkpoints under `results/` (created automatically, Git-ignored).

---

## 2. Multimodal Training (VLSM)

### VLSM-1.1B (OpenCLIP + TinyLLaMA)

Script: `Code/LMM/OpenCLIP_Linear_Projection_Tinyllama.py`

- Vision encoder: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- LLM path: `configs/TinyLlama-1.1b`
  - Expects a fine-tuned TinyLLaMA checkpoint to be placed here as
    `model.safetensors` after download (see **Model Weights** section).
- Required files:
  - `dataset/train.jsonl`
  - `dataset/images/`

Run:

```bash
python Code/LMM/OpenCLIP_Linear_Projection_Tinyllama.py
```

This script:

- Encodes images with OpenCLIP.
- Projects visual tokens into the TinyLLaMA hidden space via a linear layer.
- Concatenates visual embeddings and token embeddings along the sequence
  dimension.
- Optimizes the vision encoder + projection connector, keeping TinyLLaMA
  frozen.
- Saves the best model to `OpenCLIP_Linear_Projection_Tinyllama/`
  (Git-ignored).

### VLSM-7B (OpenCLIP + StarCoder2 + LoRA)

Script: `Code/LMM/OpenCLIP_Two_MLP_StarCoder2.py`

- Vision encoder: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- Base LLM: `bigcode/starcoder2-7b`
- LoRA adapter path: `configs/StarCoder2`
  - Expects `adapter_model.safetensors` after download (see **Model Weights**).
- Required files:
  - `dataset/train.jsonl`
  - `dataset/images/`

Run:

```bash
python Code/LMM/OpenCLIP_Two_MLP_StarCoder2.py
```

This script:

- Loads StarCoder2-7B in 4-bit and attaches a LoRA adapter from
  `configs/StarCoder2/`.
- Encodes images with OpenCLIP.
- Uses a two-layer MLP connector to map visual features into the LLM space.
- Concatenates visual and textual embeddings and trains only the vision
  encoder + connector, keeping the LoRA-augmented StarCoder2 frozen.
- Saves the best model to `OpenCLIP_Two_MLP_StarCoder2/`
  (Git-ignored).

---

## Configuration Files

All hyperparameters and tokenizer settings needed for loading the fine-tuned
models are stored under `configs/`.

| Directory | Contents |
|----------|----------|
| `configs/StarCoder2/` | `adapter_config.json`, tokenizer files, LoRA metadata, training arguments, trainer state. Expects `adapter_model.safetensors` to be placed here after downloading the fine-tuned adapter. |
| `configs/TinyLlama-1.1b/` | `config.json`, `generation_config.json`, tokenizer files, scheduler state, training arguments, trainer state. Expects `model.safetensors` to be placed here after downloading the fine-tuned TinyLLaMA checkpoint. |

Optional training state files that may appear in your local runs but are
**not required** for inference include:

- `optimizer.pt`
- `scheduler.pt`
- `rng_state_0.pth` … `rng_state_7.pth`

These are only needed if you want to resume training with identical optimizer
and RNG state.

---

## Docker Environments

Two Dockerfiles are provided to reproduce the training environments:

| Model | Dockerfile | PyTorch / CUDA |
|-------|-----------|----------------|
| StarCoder2-7B (QLoRA) | `dockerfiles/Dockerfile_StarCoder2` | PyTorch 2.3.0 / CUDA 11.8 |
| TinyLLaMA-1.1B (full retrain) | `dockerfiles/Dockerfile_TinyLlama` | PyTorch 2.1.0 / CUDA 11.8 |

Example build and run:

```bash
docker build -t gdt-starcoder2 -f dockerfiles/Dockerfile_StarCoder2 .
docker run --gpus all -it --rm gdt-starcoder2
```

You can adapt the same pattern to build and run the TinyLLaMA environment.

---

## Model Weights (to be released after review)

To comply with the double-blind policy, model weights are **not**
included in this repository.

After the review phase, we plan to release:

| Model | Target path in this repo | Required files |
|-------|--------------------------|----------------|
| StarCoder2-7B (QLoRA fine-tuned) | `configs/StarCoder2/` | `adapter_model.safetensors` (+ optional optimizer / trainer state) |
| TinyLLaMA-1.1B (fully retrained) | `configs/TinyLlama-1.1b/` | `model.safetensors` (+ optional optimizer / trainer state) |

A `checkpoints_link.txt` file will be added with direct download links once
the review process is complete.

---
## License

This repository is released under the MIT License (see LICENSE).

---

## Model Usage Notice

The models, model weights, checkpoints, and any generated outputs
associated with this project are released for academic research
and educational purposes only.

Commercial use, including but not limited to use in for-profit
products, services, internal industrial deployment, or technology
transfer, is strictly prohibited without prior written permission
from the authors.

Please contact the authors for commercial licensing inquiries.

---

## Contact

If you have any questions, feedback, or are interested in collaboration, feel free to reach out through the following channels:
- 🌐 Project Page: https://danielhsu2014.github.io/GDT-VLSM-project/
- 📧 Email: danielhsu.ii13@nycu.edu.tw
- 💼 LinkedIn: https://www.linkedin.com/in/yu-che-hsu-83048b316
