#!/usr/bin/env python3
"""Build pre-tokenized corpus shards for Kaggle training.

This script keeps assistant-only labels by masking the prompt prefix tokens with -100.
It saves .pt shards plus meta.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def load_messages(value):
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception as e:
        raise ValueError(f"Could not parse messages JSON: {str(value)[:200]}") from e


def fallback_format_messages(messages, include_assistant=True):
    chunks = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "user":
            chunks.append(f"<|user|>\n{msg.get('content','')}\n")
        elif role == "assistant" and include_assistant:
            reasoning = msg.get("reasoning_content", "")
            content = msg.get("content", "")
            if reasoning:
                chunks.append(f"<|assistant|>\n<think>\n{reasoning}\n</think>\n{content}\n")
            else:
                chunks.append(f"<|assistant|>\n{content}\n")
    return "".join(chunks)


def render_chat(tokenizer, messages, include_assistant=True, add_generation_prompt=False):
    msgs = messages if include_assistant else [m for m in messages if m.get("role") != "assistant"]
    try:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return fallback_format_messages(msgs, include_assistant=include_assistant)


def tokenize_one(tokenizer, messages, max_seq_len):
    # Prefix is user side with assistant generation prompt. Full contains assistant reasoning/content.
    prefix_text = render_chat(tokenizer, messages, include_assistant=False, add_generation_prompt=True)
    full_text = render_chat(tokenizer, messages, include_assistant=True, add_generation_prompt=False)

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if tokenizer.eos_token_id is not None and (len(full_ids) == 0 or full_ids[-1] != tokenizer.eos_token_id):
        full_ids = full_ids + [tokenizer.eos_token_id]

    truncated = len(full_ids) > max_seq_len
    full_ids = full_ids[:max_seq_len]

    labels = full_ids.copy()
    prompt_len = min(len(prefix_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "truncated": truncated,
        "seq_len": len(full_ids),
        "unmasked_tokens": sum(1 for x in labels if x != -100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--shard-size", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if "messages" not in df.columns:
        raise ValueError("Input CSV must contain a messages column. Run prepare_sft_dataset.py first.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    records = []
    stats_rows = []
    total_tokens = 0
    total_unmasked = 0
    truncated_count = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        messages = load_messages(row["messages"])
        item = tokenize_one(tokenizer, messages, max_seq_len=args.max_seq_len)
        item["id"] = str(row.get("id", i))
        item["pattern"] = str(row.get("pattern", row.get("type", "")))
        item["rule_name"] = str(row.get("solver_rule_name", ""))
        if "loss_weight" in row and pd.notna(row["loss_weight"]):
            item["loss_weight"] = float(row["loss_weight"])
        else:
            item["loss_weight"] = 1.0

        total_tokens += item["seq_len"]
        total_unmasked += item["unmasked_tokens"]
        truncated_count += int(item["truncated"])
        records.append(item)
        stats_rows.append({
            "id": item["id"],
            "pattern": item["pattern"],
            "rule_name": item["rule_name"],
            "seq_len": item["seq_len"],
            "unmasked_tokens": item["unmasked_tokens"],
            "truncated": item["truncated"],
            "loss_weight": item["loss_weight"],
        })

        if len(records) >= args.shard_size:
            shard_idx = len(list(out_dir.glob("train_*.pt")))
            torch.save(records, out_dir / f"train_{shard_idx:05d}.pt")
            records = []

    if records:
        shard_idx = len(list(out_dir.glob("train_*.pt")))
        torch.save(records, out_dir / f"train_{shard_idx:05d}.pt")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "token_stats.csv", index=False)

    meta = {
        "input": str(args.input),
        "model_name": args.model_name,
        "max_seq_len": args.max_seq_len,
        "num_examples": len(df),
        "total_tokens": int(total_tokens),
        "unmasked_tokens": int(total_unmasked),
        "truncated_count": int(truncated_count),
        "shard_size": args.shard_size,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved tokenized corpus: {out_dir}")


if __name__ == "__main__":
    main()
