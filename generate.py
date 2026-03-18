import argparse
from pathlib import Path

import torch

from config import get_default_config, resolve_paths
from tokenizer import CharTokenizer
from model import MiniGPT
from utils import read_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Belarusian text with MiniGPT")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt text in Belarusian",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    model = MiniGPT(
        vocab_size=model_vocab_size,
        block_size=model_block_size,
        d_model=model_d_model,
        n_heads=model_n_heads,
        n_layers=model_n_layers,
        dropout=model_dropout,
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    prompt_ids = tokenizer.encode(args.prompt)

    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=config.device).unsqueeze(0)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()
