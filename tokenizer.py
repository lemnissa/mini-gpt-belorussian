import json
from pathlib import Path
from typing import List


class CharTokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.fitted: bool = False

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def fit(self, text: str) -> None:
        unique_chars = sorted(set(text))
        tokens = [self.PAD_TOKEN, self.UNK_TOKEN] + unique_chars
        self.stoi = {ch: idx for idx, ch in enumerate(tokens)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}
        self.fitted = True

    def encode(self, text: str) -> List[int]:
        unk_id = self.stoi[self.UNK_TOKEN]
        return [self.stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        chars = []
        for idx in ids:
            token = self.itos.get(int(idx), self.UNK_TOKEN)
            if token == self.PAD_TOKEN:
                continue
            if token == self.UNK_TOKEN:
                chars.append("?")
            else:
                chars.append(token)
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "fitted": self.fitted,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        tokenizer = cls()
        tokenizer.stoi = {k: int(v) for k, v in payload["stoi"].items()}
        tokenizer.itos = {int(k): v for k, v in payload["itos"].items()}
        tokenizer.fitted = bool(payload["fitted"])
        return tokenizer
