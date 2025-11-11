# Generative Digital Twins — Vision-Language Simulation Models for Executable Industrial Systems

> **CVPR 2026 Submission (Anonymous)**  
> Repository for the paper *“Generative Digital Twins: Vision-Language Simulation Models for Executable Industrial Systems.”*  


---

## 📖 Overview

<p align="center">
  <img src="figures/VLSM_overview.png" width="85%">
</p>

This work introduces **Vision-Language Simulation Models (VLSM)** that unify visual and textual understanding to generate **executable FlexScript** for industrial simulation systems.  
Two model families are trained on the **GDT-120K** dataset:

| Model | Type | Purpose |
|--------|------|----------|
| **StarCoder2-7B (QLoRA fine-tuned)** | 4-bit quantized fine-tune | Lightweight inference with code-specialized backbone |
| **TinyLLaMA-1.1B (fully retrained)** | Full-precision retraining | Compact model for domain specialization |

All configuration and tokenizer files are included for reproducibility.  
Full checkpoints and adapter weights will be released **after the CVPR 2026 review period** to comply with double-blind submission policy.

---

## 🏗️ Dataset Construction

<p align="center">
  <img src="figures/dataset_pipeline.png" width="85%">
</p>

The **GDT-120K** dataset contains prompt–sketch–code triplets designed for industrial simulation.  
Each sample includes:
- A natural-language description of production logic  
- A layout sketch describing workstation topology  
- Corresponding executable FlexScript code  

Dataset covers **13 industrial categories**, from manual assembly to fully automated AGV systems.  
All parameters (e.g., `InterArrivalTime`, `ProcessTime`, `maxcontent`) follow realistic timing distributions ensuring FlexSim executability.

---

## 🧠 Architecture Overview

<p align="center">
  <img src="figures/VLSM_pipeline.png" width="80%">
</p>

The **Vision-Language Simulation Model (VLSM)** integrates:
- **Visual Encoder:** CLIP / OpenCLIP (ViT-g/14)
- **Connector:** Linear, MLP, or Q-Former for cross-modal fusion
- **Language Backbone:** TinyLLaMA-1.1B or StarCoder2-7B

Two optimized configurations:
| Model | Vision Encoder | Connector | Backbone |
|--------|----------------|-----------|-----------|
| **VLSM-1.1B** | OpenCLIP | Linear Projection | TinyLLaMA-1.1B |
| **VLSM-7B**   | OpenCLIP | Two-Layer MLP | StarCoder2-7B |

---

## ⚙️ Training Scripts
| Script | Description |
|---------|-------------|
| `starcoder2_7b_finetuning.py` | Fine-tuning StarCoder2-7B using QLoRA (PEFT). Supports 4-bit quantization and adapter training. |
| `tinyllama_fully_retrain.py` | End-to-end retraining of TinyLLaMA-1.1B on GDT-120K with full precision. |

Each script expects a standard Hugging Face model interface and can be launched with `torchrun` or `accelerate`.

---

## 🧩 Configuration Files
All hyperparameters and tokenizer settings are preserved under `configs/`.

| Directory | Description |
|------------|-------------|
| `configs/StarCoder2/` | Adapter configuration, tokenizer, and LoRA metadata. |
| `configs/TinyLlama-1.1b/` | Full retraining setup including scheduler state and generation configuration. |

These files ensure full reproducibility without exposing model weights.

---

## 🐳 Docker Environments

| Model | Dockerfile | PyTorch / CUDA | Notes |
|---|---|---|---|
| **StarCoder2-7B (QLoRA)** | `dockerfiles/Dockerfile_StarCoder2` | **PyTorch 2.3.0 / CUDA 11.8** | runtime + cuDNN 8 |
| **TinyLLaMA-1.1B (Full retrain)** | `dockerfiles/Dockerfile_TinyLlama` | **PyTorch 2.1.0 / CUDA 11.8** | runtime + cuDNN 8 |

**Build**
```bash
docker build -t gdt-starcoder2 -f dockerfiles/Dockerfile_StarCoder2 .
docker run --gpus all -it --rm gdt-starcoder2
```

---

## 🔗 Model Weights (to be released after review)
Due to the CVPR 2026 double-blind policy and GitHub size constraints, model checkpoints are temporarily withheld.
After the review process, fine-tuned weights and full retraining checkpoints will be made available via Google Drive.

Planned uploads:

|Model | Files	| Hosting
|--------|-------------|-------|
|StarCoder2-7B (QLoRA fine-tuned, epoch 6 best)	| adapter_model.safetensors, optimizer.pt, trainer_state.json	| Google Drive (to be added)
|TinyLLaMA-1.1B (fully retrained, best checkpoint)	| model.safetensors, optimizer.pt, trainer_state.json	| Google Drive (to be added)

A checkpoints_link.txt file will be added post-review with direct download links.

---

## 📜 License
This repository is released under the MIT License (see LICENSE).

---

## 📬 Contact
Details will be added following the CVPR 2026 review phase.
