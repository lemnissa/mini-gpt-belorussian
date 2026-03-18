from dataclasses import dataclass, asdict
from pathlib import Path
import torch


@dataclass
class Config:
    data_path: str = "data/corpus_be.txt"
    tokenizer_path: str = "artifacts/tokenizer.json"
    checkpoint_path: str = "artifacts/mini_gpt_be.pt"

    batch_size: int = 32
    block_size: int = 128
    vocab_size: int = 0

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 20
    eval_every: int = 200
    num_workers: int = 0

    seed: int = 42
    train_split: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_grad_norm: float = 1.0


def get_default_config() -> Config:
    return Config()


def resolve_paths(config: Config, base_dir: Path) -> Config:
    config.data_path = str((base_dir / config.data_path).resolve())
    config.tokenizer_path = str((base_dir / config.tokenizer_path).resolve())
    config.checkpoint_path = str((base_dir / config.checkpoint_path).resolve())
    return config


def config_to_dict(config: Config) -> dict:
    return asdict(config)
