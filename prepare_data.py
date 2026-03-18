import argparse
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


CYRILLIC_RE = re.compile(r"[А-Яа-яЁёІіЎўЭэЫы]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_good_line(text: str, min_chars: int = 60) -> bool:
    if len(text) < min_chars:
        return False
    if not CYRILLIC_RE.search(text):
        return False
    return True


def stream_wikipedia(max_docs: int) -> Iterable[str]:
    ds = load_dataset("wikimedia/wikipedia", "20231101.be", split="train", streaming=True)
    count = 0
    for row in ds:
        text = normalize_text(row.get("text", ""))
        if is_good_line(text):
            yield text
            count += 1
            if count >= max_docs:
                break


def stream_oscar(max_docs: int) -> Iterable[str]:
    candidates = [
        ("oscar", "unshuffled_deduplicated_be"),
        ("oscar-corpus/OSCAR-2301", "be"),
        ("oscar-corpus/OSCAR-2201", "be"),
    ]
    ds = None
    last_error = None
    for ds_name, ds_config in candidates:
        try:
            ds = load_dataset(ds_name, ds_config, split="train", streaming=True)
            print(f"  using {ds_name}:{ds_config}")
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    count = 0
    for row in ds:
        text = normalize_text(row.get("text", ""))
        if is_good_line(text):
            yield text
            count += 1
            if count >= max_docs:
                break


def stream_mc4(max_docs: int) -> Iterable[str]:
    ds = load_dataset("allenai/c4", "be", split="train", streaming=True)
    count = 0
    for row in ds:
        text = normalize_text(row.get("text", ""))
        if is_good_line(text):
            yield text
            count += 1
            if count >= max_docs:
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Belarusian corpus from public datasets")
    parser.add_argument(
        "--output",
        type=str,
        default="data/corpus_be.txt",
        help="Output corpus path",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["wikipedia", "oscar", "mc4"],
        choices=["wikipedia", "oscar", "mc4"],
        help="Data sources to use",
    )
    parser.add_argument("--max_docs_per_source", type=int, default=20000)
    parser.add_argument("--min_chars", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    global is_good_line 
    original_filter = is_good_line

    def patched_filter(text: str, min_chars: int = args.min_chars) -> bool:
        return original_filter(text, min_chars=min_chars)

    is_good_line = patched_filter

    source_to_fn = {
        "wikipedia": stream_wikipedia,
        "oscar": stream_oscar,
        "mc4": stream_mc4,
    }

    total_docs = 0
    with output_path.open("w", encoding="utf-8") as f:
        for source in args.sources:
            print(f"Collecting source: {source}")
            source_docs = 0
            try:
                for text in source_to_fn[source](args.max_docs_per_source):
                    f.write(text + "\n")
                    source_docs += 1
            except Exception as exc:  
                print(f"  skipped {source}: {exc}")
                continue
            total_docs += source_docs
            print(f"  saved {source_docs} documents")

    print(f"Done. Total documents: {total_docs}")
    print(f"Saved corpus to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
