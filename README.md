## 🚀 Pipeline Workflow

```mermaid
flowchart LR
    A[Dataset Format<br/>Chat Template] --> B[4-Bit Model + LoRA<br/>Config Initialization]
    B --> C[SFT Trainer<br/>Fine-Tuning]
    C --> D[Inference & Eval]
