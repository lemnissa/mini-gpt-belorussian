from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for base or LoRA-tuned model")
    parser.add_argument("--config", type=str, default="llm_adaptation/config.yaml")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--use_lora", action="store_true")
    return parser.parse_args()


def build_model_and_tokenizer(config: dict, use_lora: bool):
    model_id = config["model"]["base_model"]
    out_dir = Path(config["training"]["output_dir"]) / "final_adapter"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if use_lora:
        model = PeftModel.from_pretrained(base_model, str(out_dir))
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    model, tokenizer = build_model_and_tokenizer(cfg, use_lora=args.use_lora)

    prompt = (
        "<|system|>\nТы карысны асістэнт, які адказвае па-беларуску.\n"
        "<|user|>\n"
        f"{args.prompt}\n"
        "<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_cfg = cfg["inference"]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=float(gen_cfg["temperature"]),
            top_p=float(gen_cfg["top_p"]),
            do_sample=bool(gen_cfg["do_sample"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
