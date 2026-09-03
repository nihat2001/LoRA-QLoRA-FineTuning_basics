# 🚀 Fine-Tuning LLMs with LoRA & QLoRA

A comprehensive hands-on guide and codebase for efficient fine-tuning of Large Language Models (LLMs) using **LoRA** (Low-Rank Adaptation) and **QLoRA** (Quantized Low-Rank Adaptation) with Hugging Face's `PEFT`, `TRL`, and `bitsandbytes`.

---

┌─────────────────┐     ┌───────────────────────┐     ┌──────────────────┐
│  Dataset Format │ ──> │ 4-Bit Model + LoRA    │ ──> │ SFT Trainer      │
│  (Chat Template)│     │ Config Initialization │     │ (Fine-Tuning)    │
└─────────────────┘     └───────────────────────┘     └──────────────────┘
                                                               │
                                                               ▼
                                                      ┌──────────────────┐
                                                      │ Inference & Eval │
                                                      └──────────────────┘
                                                      
---
                                                      
## 📊 Quick Overview & Comparison

| Feature / Metric | LoRA (Low-Rank Adaptation) | QLoRA (Quantized LoRA) |
| :--- | :--- | :--- |
| **Base Model Precision** | 16-bit (FP16 / BF16) | 4-bit NormalFloat (NF4) |
| **GPU Memory Requirement** | Moderate | Very Low (~65% reduction) |
| **Quantization Scheme** | None | 4-bit quantization + Double Quantization |
| **Compute Dtype** | FP16 or BF16 | BF16 (Recommended for stability) |
| **Best For** | Medium/Large GPU memory setup | Consumer GPUs / Limited VRAM |

---

## ⚙️ Hyperparameters & Configuration

| Parameter | Recommended Value | Description |
| :--- | :--- | :--- |
| `r` (Rank) | `8` or `16` | Dimension of the low-rank matrices. Higher values increase trainable parameters. |
| `lora_alpha` | `16` or `32` | Scaling factor for LoRA weights (typically set to `2 * r`). |
| `lora_dropout` | `0.05` | Dropout probability for LoRA layers to prevent overfitting. |
| `target_modules` | `["q_proj", "v_proj"]` | Target attention projection matrices to apply LoRA. |
| `bnb_4bit_quant_type` | `"nf4"` | Information-theoretically optimal quantile quantization data type. |
| `bnb_4bit_compute_dtype`| `torch.bfloat16` | Precision used for internal forward/backward computation passes. |
