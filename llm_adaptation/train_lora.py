from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from trl import SFTTrainer

from dataset import load_yaml, load_instruction_dataset, to_sft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Belarusian adaptation")
    parser.add_argument("--config", type=str, default="llm_adaptation/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(int(cfg["experiment"]["seed"]))

    model_id = cfg["model"]["base_model"]
    max_seq_length = int(cfg["model"]["max_seq_length"])
    out_dir = Path(cfg["training"]["output_dir"]) 
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_instruction_dataset(cfg)
    train_ds = to_sft_dataset(ds["train"])
    eval_ds = to_sft_dataset(ds["eval"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["lora_alpha"]),
        lora_dropout=float(cfg["lora"]["lora_dropout"]),
        target_modules=list(cfg["lora"]["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        max_steps=int(train_cfg["max_steps"]),
        eval_strategy="steps",
        eval_steps=int(train_cfg["eval_steps"]),
        save_strategy="steps",
        save_steps=int(train_cfg["save_steps"]),
        logging_steps=int(train_cfg["logging_steps"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        run_name=str(cfg["experiment"]["run_name"]),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        max_length=max_seq_length,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(str(out_dir / "final_adapter"))
    tokenizer.save_pretrained(str(out_dir / "final_adapter"))

    metrics = trainer.evaluate()
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Saved LoRA adapter to", out_dir / "final_adapter")
    print("Eval metrics:", metrics)


if __name__ == "__main__":
    main()
