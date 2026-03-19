from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "instruction": str(obj.get("instruction", "")).strip(),
                    "input": str(obj.get("input", "")).strip(),
                    "output": str(obj.get("output", "")).strip(),
                }
            )
    return rows


def _build_synthetic_from_corpus(
    corpus_path: Path,
    max_samples: int,
    min_text_chars: int,
) -> list[dict[str, str]]:
    text = corpus_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= min_text_chars]

    rows: list[dict[str, str]] = []
    for p in paragraphs:
        split_idx = int(len(p) * 0.6)
        if split_idx < 30 or len(p) - split_idx < 20:
            continue
        rows.append(
            {
                "instruction": "Працягні наступны беларускі тэкст натуральна і звязна.",
                "input": p[:split_idx],
                "output": p[split_idx:],
            }
        )
        if len(rows) >= max_samples:
            break

    if not rows:
        raise ValueError(
            "Could not build instruction dataset from corpus. "
            "Provide llm_adaptation/data/be_instructions.jsonl"
        )
    return rows


def load_instruction_dataset(config: dict[str, Any]) -> DatasetDict:
    data_cfg = config["data"]
    instruction_path = Path(data_cfg["instruction_path"])
    if instruction_path.exists():
        rows = _read_jsonl(instruction_path)
    else:
        corpus_path = Path(data_cfg["fallback_corpus_path"])
        rows = _build_synthetic_from_corpus(
            corpus_path=corpus_path,
            max_samples=int(data_cfg["max_samples"]),
            min_text_chars=int(data_cfg["min_text_chars"]),
        )

    if len(rows) < 20:
        raise ValueError("Instruction dataset is too small. Need at least 20 samples.")

    split = float(data_cfg["train_split"])
    split_idx = int(len(rows) * split)
    train_rows = rows[:split_idx]
    eval_rows = rows[split_idx:]

    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "eval": Dataset.from_list(eval_rows),
        }
    )


def format_sample(instruction: str, input_text: str, output_text: str) -> str:
    if input_text:
        user_part = f"{instruction}\n\nУвод:\n{input_text}"
    else:
        user_part = instruction

    return (
        "<|system|>\n"
        "Ты карысны асістэнт, які адказвае па-беларуску.\n"
        "<|user|>\n"
        f"{user_part}\n"
        "<|assistant|>\n"
        f"{output_text}"
    )


def to_sft_dataset(ds: Dataset) -> Dataset:
    def _map_fn(x: dict[str, str]) -> dict[str, str]:
        return {
            "text": format_sample(
                instruction=x.get("instruction", ""),
                input_text=x.get("input", ""),
                output_text=x.get("output", ""),
            )
        }

    return ds.map(_map_fn, remove_columns=ds.column_names)
