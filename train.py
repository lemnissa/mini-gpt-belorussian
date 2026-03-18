from pathlib import Path

import torch
from torch.optim import AdamW

from config import get_default_config, resolve_paths, config_to_dict
from tokenizer import CharTokenizer
from dataset import load_text, split_tokens, create_dataloaders
from model import MiniGPT
from utils import set_seed, count_parameters, save_checkpoint, maybe_make_dir, AverageMeter


def evaluate(model: MiniGPT, val_loader, device: str) -> float:
    model.eval()
    meter = AverageMeter()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            meter.update(loss.item(), n=x.size(0))
    return meter.avg


def train() -> None:
    base_dir = Path(__file__).resolve().parent
    config = resolve_paths(get_default_config(), base_dir)

    set_seed(config.seed)
    maybe_make_dir(Path(config.checkpoint_path).parent)
    maybe_make_dir(Path(config.tokenizer_path).parent)

    text = load_text(config.data_path)

    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    tokenizer.save(config.tokenizer_path)

    token_ids = tokenizer.encode(text)
    train_ids, val_ids = split_tokens(token_ids, train_split=config.train_split)

    max_allowed_block = min(len(train_ids) - 1, len(val_ids) - 1)
    if config.block_size > max_allowed_block:
        print(
            f"Adjusting block_size from {config.block_size} to {max_allowed_block} "
            "to fit train/val splits."
        )
        config.block_size = max_allowed_block

    train_loader, val_loader = create_dataloaders(
        train_ids=train_ids,
        val_ids=val_ids,
        block_size=config.block_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    config.vocab_size = tokenizer.vocab_size

    model = MiniGPT(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print(f"Device: {config.device}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Trainable params: {count_parameters(model):,}")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_meter = AverageMeter()

        for x, y in train_loader:
            x = x.to(config.device)
            y = y.to(config.device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            global_step += 1
            train_meter.update(loss.item(), n=x.size(0))

            if global_step % config.eval_every == 0:
                val_loss = evaluate(model, val_loader, config.device)
                print(
                    f"Epoch {epoch}/{config.max_epochs} | Step {global_step} | "
                    f"Train Loss {train_meter.avg:.4f} | Val Loss {val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        path=config.checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=global_step,
                        best_val_loss=best_val_loss,
                        config=config_to_dict(config),
                    )
                    print(
                        f"Saved best checkpoint to {config.checkpoint_path} "
                        f"(val_loss={best_val_loss:.4f})"
                    )

        val_loss = evaluate(model, val_loader, config.device)
        print(
            f"Epoch {epoch}/{config.max_epochs} completed | "
            f"Train Loss {train_meter.avg:.4f} | Val Loss {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=config.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                best_val_loss=best_val_loss,
                config=config_to_dict(config),
            )
            print(
                f"Saved best checkpoint to {config.checkpoint_path} "
                f"(val_loss={best_val_loss:.4f})"
            )

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
