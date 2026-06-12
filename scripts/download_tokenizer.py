# scripts/download_tokenizer.py

from __future__ import annotations

import argparse
from pathlib import Path

from nemotron_repro.tokenizer_utils import load_tokenizer, load_model_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/model_config.json",
        help="Path to model config JSON.",
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help="Hugging Face model/tokenizer name. Overrides config.",
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face cache directory. Overrides config.",
    )

    parser.add_argument(
        "--text",
        default="Hello Wonderland.",
        help="Text used for tokenizer smoke test.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_model_config(args.config)

    model_name = args.model_name or config.get("model_name")
    cache_dir = args.cache_dir or config.get("cache_dir")
    trust_remote_code = bool(config.get("trust_remote_code", True))
    use_fast = bool(config.get("use_fast", True))

    print("=" * 80)
    print("Downloading / loading tokenizer")
    print("model_name:", model_name)
    print("cache_dir :", cache_dir)
    print("trust_remote_code:", trust_remote_code)
    print("use_fast:", use_fast)

    tokenizer = load_tokenizer(
        model_name=model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
        config_path=args.config,
    )

    encoded = tokenizer(
        args.text,
        add_special_tokens=False,
    )

    print("=" * 80)
    print("Tokenizer loaded successfully.")
    print("vocab_size:", getattr(tokenizer, "vocab_size", None))
    print("test_text:", args.text)
    print("token_count:", len(encoded["input_ids"]))
    print("input_ids:", encoded["input_ids"][:50])

    cache_path = Path(cache_dir) if cache_dir else None

    if cache_path is not None:
        print("=" * 80)
        print("Cache directory:")
        print(cache_path.resolve())

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()