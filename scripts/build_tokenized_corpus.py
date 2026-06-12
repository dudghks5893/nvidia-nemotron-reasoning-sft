#!/usr/bin/env python3
# scripts/build_tokenized_corpus.py

"""
Build pre-tokenized corpus shards for Kaggle training.

This script does NOT train a model.

It only does:
    SFT CSV
    -> tokenizer.apply_chat_template
    -> input_ids / attention_mask / labels
    -> .pt shard files
    -> token_stats.csv
    -> meta.json

The generated corpus can later be uploaded to a Kaggle Dataset and used
by a Kaggle GPU training notebook/script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from nemotron_repro.tokenizer_utils import (
    load_messages,
    load_model_config,
    load_tokenizer,
    tokenize_messages_assistant_only,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input SFT CSV. Usually data/generated/sft_train.csv or selected/budgeted CSV.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for tokenized corpus shards.",
    )

    parser.add_argument(
        "--model-config",
        default="configs/model_config.json",
        help="Model/tokenizer config JSON.",
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional HF tokenizer name/path. Overrides config.",
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HF cache dir. Overrides config.",
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Max sequence length. If omitted, uses model_config.json.",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of examples per .pt shard.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug row limit.",
    )

    return parser.parse_args()


def save_shard(records, out_dir: Path, shard_idx: int) -> Path:
    shard_path = out_dir / f"train_{shard_idx:05d}.pt"
    torch.save(records, shard_path)
    return shard_path


def main():
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_model_config(args.model_config)
    max_seq_len = args.max_seq_len or int(config.get("max_seq_len", 8192))

    tokenizer = load_tokenizer(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        config_path=args.model_config,
    )

    df = pd.read_csv(input_path)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    if "messages" not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'messages' column. "
            "Run scripts/prepare_sft_dataset.py first."
        )

    records = []
    stats_rows = []

    total_tokens = 0
    total_unmasked_tokens = 0
    total_prompt_tokens = 0
    truncated_count = 0
    answer_missing_after_truncation_count = 0

    shard_idx = 0

    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc="Building tokenized corpus",
    )

    for i, row in iterator:
        messages = load_messages(row["messages"])

        item = tokenize_messages_assistant_only(
            tokenizer=tokenizer,
            messages=messages,
            max_seq_len=max_seq_len,
            answer=row.get("answer", None),
        )

        sample_id = str(row.get("id", i))
        pattern = str(row.get("pattern", row.get("type", "")))
        rule_name = str(row.get("solver_rule_name", ""))

        loss_weight = row.get("loss_weight", 1.0)

        try:
            loss_weight = float(loss_weight)
        except Exception:
            loss_weight = 1.0

        record = {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
            "id": sample_id,
            "pattern": pattern,
            "rule_name": rule_name,
            "loss_weight": loss_weight,
        }

        records.append(record)

        total_tokens += int(item["seq_len"])
        total_prompt_tokens += int(item["prompt_tokens"])
        total_unmasked_tokens += int(item["unmasked_tokens"])
        truncated_count += int(item["truncated"])

        answer_in_truncated_tail = item.get("answer_in_truncated_tail")

        if item["truncated"] and answer_in_truncated_tail is False:
            answer_missing_after_truncation_count += 1

        stats_rows.append(
            {
                "id": sample_id,
                "pattern": pattern,
                "rule_name": rule_name,
                "seq_len": int(item["seq_len"]),
                "original_seq_len": int(item["original_seq_len"]),
                "prompt_tokens": int(item["prompt_tokens"]),
                "unmasked_tokens": int(item["unmasked_tokens"]),
                "truncated": bool(item["truncated"]),
                "answer_in_truncated_tail": answer_in_truncated_tail,
                "loss_weight": loss_weight,
            }
        )

        if len(records) >= args.shard_size:
            save_shard(records, out_dir, shard_idx)
            shard_idx += 1
            records = []

    if records:
        save_shard(records, out_dir, shard_idx)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "token_stats.csv", index=False)

    pattern_stats = (
        stats_df.groupby("pattern")
        .agg(
            row_count=("pattern", "size"),
            mean_seq_len=("seq_len", "mean"),
            max_seq_len=("seq_len", "max"),
            total_seq_len=("seq_len", "sum"),
            mean_unmasked_tokens=("unmasked_tokens", "mean"),
            total_unmasked_tokens=("unmasked_tokens", "sum"),
            truncated_count=("truncated", "sum"),
        )
        .reset_index()
        .sort_values("total_unmasked_tokens", ascending=False)
    )

    pattern_stats["unmasked_token_share"] = (
        pattern_stats["total_unmasked_tokens"]
        / pattern_stats["total_unmasked_tokens"].sum()
        if pattern_stats["total_unmasked_tokens"].sum() > 0
        else 0.0
    )

    pattern_stats.to_csv(out_dir / "pattern_token_stats.csv", index=False)

    meta = {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "model_name": args.model_name or config.get("model_name"),
        "cache_dir": args.cache_dir or config.get("cache_dir"),
        "max_seq_len": int(max_seq_len),
        "num_examples": int(len(df)),
        "total_tokens": int(total_tokens),
        "prompt_tokens": int(total_prompt_tokens),
        "unmasked_tokens": int(total_unmasked_tokens),
        "truncated_count": int(truncated_count),
        "answer_missing_after_truncation_count": int(
            answer_missing_after_truncation_count
        ),
        "shard_size": int(args.shard_size),
        "num_shards": int(len(list(out_dir.glob("train_*.pt")))),
        "files": {
            "token_stats": "token_stats.csv",
            "pattern_token_stats": "pattern_token_stats.csv",
            "meta": "meta.json",
            "shards": "train_*.pt",
        },
    }

    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("[Tokenized corpus meta]")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    print("=" * 100)
    print("[Pattern token stats]")
    print(pattern_stats.to_string(index=False))

    print("=" * 100)
    print(f"Saved tokenized corpus: {out_dir}")


if __name__ == "__main__":
    main()