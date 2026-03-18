import math
from pathlib import Path

import torch

from config import get_default_config, resolve_paths
from tokenizer import CharTokenizer
from dataset import load_text, split_tokens, create_dataloaders
from model import MiniGPT
from utils import AverageMeter, read_checkpoint


def evaluate() -> None:
    base_dir = Path(__file__).resolve().parent
    config = resolve_paths(get_default_config(), base_dir)

    tokenizer = CharTokenizer.load(config.tokenizer_path)
    checkpoint = read_checkpoint(config.checkpoint_path, map_location=config.device)
    checkpoint_config = checkpoint.get("config", {})

    model_vocab_size = int(checkpoint_config.get("vocab_size", tokenizer.vocab_size))
    model_block_size = int(checkpoint_config.get("block_size", config.block_size))
    model_d_model = int(checkpoint_config.get("d_model", config.d_model))
    model_n_heads = int(checkpoint_config.get("n_heads", config.n_heads))
    model_n_layers = int(checkpoint_config.get("n_layers", config.n_layers))
    model_dropout = float(checkpoint_config.get("dropout", config.dropout))

    text = load_text(config.data_path)
    token_ids = tokenizer.encode(text)
    _, val_ids = split_tokens(token_ids, train_split=config.train_split)

    _, val_loader = create_dataloaders(
        train_ids=token_ids[: max(model_block_size + 1, 2)],
        val_ids=val_ids,
        block_size=model_block_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = MiniGPT(
        vocab_size=model_vocab_size,
        block_size=model_block_size,
        d_model=model_d_model,
        n_heads=model_n_heads,
        n_layers=model_n_layers,
        dropout=model_dropout,
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    meter = AverageMeter()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            _, loss = model(x, y)
            meter.update(loss.item(), n=x.size(0))

    val_loss = meter.avg
    perplexity = math.exp(val_loss)

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    evaluate()
