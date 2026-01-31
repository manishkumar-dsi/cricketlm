#!/usr/bin/env python

import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

def main():
    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------
    BASE_MODEL = "meta-llama/Llama-3.1-8B"
    DATA_DIR = "output2"
    OUTPUT_DIR = "phase1_cricket_domain_lora_2"
    CACHE_DIR = "./my_local_cache/"

    MAX_LENGTH = 512
    MICRO_BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 8
    EPOCHS = 1
    LEARNING_RATE = 1e-4

    torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------------
    jsonl_files = sorted([str(p) for p in Path(DATA_DIR).glob("*.jsonl")])
    if not jsonl_files:
        raise ValueError("No JSONL files found")

    dataset = load_dataset("json", data_files=jsonl_files)["train"]
    dataset = dataset.filter(lambda x: x["text"] is not None and isinstance(x["text"], str))
    dataset = dataset.shuffle(seed=42)

    # ------------------------------------------------------------------
    # TOKENIZER
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, cache_dir=CACHE_DIR, local_files_only=False)
    
    # Most of the time it works but it could give an issue in some cases.
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1,  # 🔑 Windows-safe
    )

    # ------------------------------------------------------------------
    # DATA COLLATOR
    # ------------------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ------------------------------------------------------------------
    # MODEL (QLoRA)
    # ------------------------------------------------------------------
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
        cache_dir=CACHE_DIR, local_files_only=False
    )

    # model = prepare_model_for_kbit_training(
    #     model,
    #     "phase1_cricket_domain_lora/checkpoint-31000")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # TRAINING ARGS
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,        # every ~1.5 hours in your case
        save_total_limit=3,
        save_safetensors=True,   
        ignore_data_skip=True,   
        optim="paged_adamw_32bit",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # trainer.train(resume_from_checkpoint=True)
    # trainer.train(resume_from_checkpoint="phase1_cricket_domain_lora/checkpoint-4000")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Phase-1 domain pretraining complete")

if __name__ == "__main__":
    main()
