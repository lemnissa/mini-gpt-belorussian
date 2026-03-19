from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative eval: base vs LoRA")
    parser.add_argument("--config", type=str, default="llm_adaptation/config.yaml")
    return parser.parse_args()


def _distinct_2(text: str) -> float:
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks[:-1], toks[1:]))
    return len(set(bigrams)) / max(len(bigrams), 1)


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    full_prompt = (
        "<|system|>\nТы карысны асістэнт, які адказвае па-беларуску.\n"
        "<|user|>\n"
        f"{prompt}\n"
        "<|assistant|>\n"
    )
    inp = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    model_id = cfg["model"]["base_model"]
    out_dir = Path(cfg["training"]["output_dir"])
    lora_dir = out_dir / "final_adapter"
    report_dir = out_dir / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    prompts = [
        "Патлумач, чаму беларуская мова важная для культуры.",
        "Складзі кароткі абзац пра вясну ў Мінску.",
        "Напішы 5 парад, як вывучаць беларускую мову кожны дзень.",
        "Перакажы сэнс сказа: мова аб'ядноўвае людзей.",
        "Прыдумай невялікі дыялог у краме па-беларуску.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    tuned_model = PeftModel.from_pretrained(base_model, str(lora_dir))

    rows = []
    for i, prompt in enumerate(prompts, start=1):
        base_text = _generate(base_model, tokenizer, prompt, max_new_tokens=160)
        tuned_text = _generate(tuned_model, tokenizer, prompt, max_new_tokens=160)
        rows.append(
            {
                "id": i,
                "prompt": prompt,
                "base_response": base_text,
                "tuned_response": tuned_text,
                "base_len": len(base_text.split()),
                "tuned_len": len(tuned_text.split()),
                "base_distinct2": f"{_distinct_2(base_text):.4f}",
                "tuned_distinct2": f"{_distinct_2(tuned_text):.4f}",
            }
        )

    csv_path = report_dir / "base_vs_lora.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = report_dir / "base_vs_lora.md"
    lines = ["# Base vs LoRA qualitative comparison\n"]
    for r in rows:
        lines.append(f"## Prompt {r['id']}")
        lines.append(f"**Prompt:** {r['prompt']}")
        lines.append(f"**Base:** {r['base_response']}")
        lines.append(f"**LoRA:** {r['tuned_response']}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved report: {csv_path}")
    print(f"Saved report: {md_path}")


if __name__ == "__main__":
    main()
