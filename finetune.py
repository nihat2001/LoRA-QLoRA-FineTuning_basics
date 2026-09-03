# ==== 1. Install Libraries ====
!pip install -q -U transformers datasets peft trl bitsandbytes accelerate

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ==== 2. Dataset ====
examples = [
    {"user": "What is Python?",
     "assistant": "Hello! Python is a programming language."},

    {"user": "What is a GPU?",
     "assistant": "Hello! A GPU is a graphics processor."},

    {"user": "What is AI?",
     "assistant": "Hello! AI means artificial intelligence."},

    {"user": "What is a model?",
     "assistant": "Hello! A model is a trained neural network."},

    {"user": "What is data?",
     "assistant": "Hello! Data is information used for training."},
]

dataset = Dataset.from_list(examples)

# ==== 3. Tokenizer ====
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_prompt(example):
    messages = [
        {"role": "system",    "content": "You are an assistant who provides short answers."},
        {"role": "user",      "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

dataset = dataset.map(format_prompt)
print(dataset[0]["text"])

# ==== 4. Load Model in 4-bit ====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # changed: float16 -> bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

print("Model loaded")

# ==== 5. LoRA Configuration ====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# ==== 6. Training ====
training_args = SFTConfig(
    output_dir="./my-model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=1,
    fp16=False,      # changed
    bf16=True,       # changed
    max_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
)

trainer.train()

# ==== 7. Inference Setup ====
# Switch the model to evaluation mode before inference
trainer.model.eval()                     # disables dropout and other training behaviors
trainer.model.config.use_cache = True    # re-enables KV caching
trainer.model.gradient_checkpointing_disable()  # disables gradient checkpointing

def get_response(model, question):
    """Sends a question to the model and returns its generated response."""

    messages = [
        {"role": "system", "content": "You are an assistant who provides one word answers."},
        {"role": "user",   "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


print(get_response(trainer.model, "What is a car?"))
