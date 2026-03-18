from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class LanguageModelingDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int) -> None:
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        return x, y


def load_text(data_path: str) -> str:
    path = Path(data_path)
    return path.read_text(encoding="utf-8")


def split_tokens(token_ids: list[int], train_split: float = 0.9) -> tuple[list[int], list[int]]:
    split_idx = int(len(token_ids) * train_split)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    return train_ids, val_ids


def create_dataloaders(
    train_ids: list[int],
    val_ids: list[int],
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = LanguageModelingDataset(train_ids, block_size)
    val_dataset = LanguageModelingDataset(val_ids, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
