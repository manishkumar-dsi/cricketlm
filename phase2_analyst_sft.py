import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

# -----------------------------
# Configuration
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PHASE1_ADAPTER = "phase1_cricket_domain_lora/checkpoint-31000"
PHASE2_OUTPUT = "phase2_cricket_analyst_lora"
DATA_FILE = "phase2_analyst_sft.jsonl"

MAX_LENGTH = 512
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EPOCHS = 3
LEARNING_RATE = 2e-5

# -----------------------------
# Load Dataset
# -----------------------------
dataset = load_dataset("json", data_files=DATA_FILE)["train"]

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Prompt Formatting
# -----------------------------
def format_prompt(example):
    instruction = example["instruction"].strip()
    inp = example.get("input", "").strip()
    output = example["output"].strip()

    prompt = f"Instruction:\n{instruction}\n\n"
    if inp:
        prompt += f"Context:\n{inp}\n\n"
    prompt += "Answer:\n"

    return {
        "prompt": prompt,
        "completion": output
    }

dataset = dataset.map(format_prompt)

# -----------------------------
# Tokenization (SFT MODE)
# -----------------------------
def tokenize_fn(example):
    prompt = example["prompt"]
    completion = example["completion"]

    full_text = prompt + completion

    tokenized_prompt = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH
    )

    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    labels = tokenized_full["input_ids"].copy()
    prompt_len = len(tokenized_prompt["input_ids"])

    # 🔥 MASK PROMPT TOKENS
    labels[:prompt_len] = [-100] * prompt_len

    tokenized_full["labels"] = labels
    return tokenized_full

tokenized_dataset = dataset.map(
    tokenize_fn,
    remove_columns=dataset.column_names
)
tokenized_dataset = tokenized_dataset.with_format("torch")

# -----------------------------
# Load Base Model (QLoRA)
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# Attach Phase-1 adapter
model = PeftModel.from_pretrained(model, PHASE1_ADAPTER)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# -----------------------------
# Phase-2 LoRA (BEHAVIOR MODE)
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # narrower
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=PHASE2_OUTPUT,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    report_to="none",
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Save Phase-2 Adapter
# -----------------------------
model.save_pretrained(PHASE2_OUTPUT)
tokenizer.save_pretrained(PHASE2_OUTPUT)

print("Phase 2 analyst fine-tuning complete.")
