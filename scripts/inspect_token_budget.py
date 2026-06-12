#!/usr/bin/env python3
# scripts/inspect_token_budget.py

"""
Inspect pattern-wise token budget.

This script is for local CPU usage.

It does NOT train a model.
It only loads the tokenizer and counts tokens.

Main outputs:
    - pattern-level token stats
    - optional row-level CSV with token columns
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nemotron_repro.tokenizer_utils import (
    assistant_text_from_messages,
    count_text_tokens,
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
        help="Input SFT CSV. Usually data/generated/sft_train.csv",
    )

    parser.add_argument(
        "--output",
        default="data/generated/token_budget_stats.csv",
        help="Output pattern-level stats CSV.",
    )

    parser.add_argument(
        "--output-data",
        default="data/generated/sft_train_with_token_stats.csv",
        help="Output row-level CSV with token stats.",
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
        "--pattern-col",
        default="pattern",
        help="Pattern column name.",
    )

    parser.add_argument(
        "--solution-col",
        default="solver_solution",
        help="Fallback solution column if messages column is missing.",
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Max sequence length. If omitted, uses model_config.json.",
    )

    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Use character counts only. Does not load tokenizer.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for debugging.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    if args.pattern_col not in df.columns:
        if "type" in df.columns:
            df[args.pattern_col] = df["type"]
        else:
            raise ValueError(
                f"Pattern column '{args.pattern_col}' not found, "
                "and fallback column 'type' does not exist."
            )

    config = load_model_config(args.model_config)
    max_seq_len = args.max_seq_len or int(config.get("max_seq_len", 8192))

    tokenizer = None

    if not args.no_tokenizer:
        tokenizer = load_tokenizer(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            config_path=args.model_config,
        )

    row_stats = []

    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc="Inspecting token budget",
    )

    for row_index, row in iterator:
        row_id = row.get("id", row_index)
        pattern = row.get(args.pattern_col, "")
        rule_name = row.get("solver_rule_name", "")

        answer = row.get("answer", None)

        if "messages" in df.columns:
            messages = load_messages(row["messages"])
        else:
            messages = []

        if messages:
            assistant_text = assistant_text_from_messages(messages)
        else:
            assistant_text = str(row.get(args.solution_col, ""))

        if tokenizer is None:
            assistant_budget_units = len(assistant_text)
            seq_len = None
            prompt_tokens = None
            unmasked_tokens = assistant_budget_units
            truncated = False
            original_seq_len = None
            answer_in_truncated_tail = None

        else:
            if messages:
                tokenized = tokenize_messages_assistant_only(
                    tokenizer=tokenizer,
                    messages=messages,
                    max_seq_len=max_seq_len,
                    answer=answer,
                )

                seq_len = tokenized["seq_len"]
                prompt_tokens = tokenized["prompt_tokens"]
                unmasked_tokens = tokenized["unmasked_tokens"]
                truncated = tokenized["truncated"]
                original_seq_len = tokenized["original_seq_len"]
                answer_in_truncated_tail = tokenized["answer_in_truncated_tail"]
                assistant_budget_units = unmasked_tokens

            else:
                assistant_budget_units = count_text_tokens(
                    tokenizer=tokenizer,
                    text=assistant_text,
                    add_special_tokens=False,
                )
                seq_len = None
                prompt_tokens = None
                unmasked_tokens = assistant_budget_units
                truncated = False
                original_seq_len = None
                answer_in_truncated_tail = None

        row_stats.append(
            {
                "row_index": row_index,
                "id": row_id,
                "pattern": pattern,
                "solver_rule_name": rule_name,
                "assistant_budget_units": int(assistant_budget_units),
                "seq_len": seq_len,
                "prompt_tokens": prompt_tokens,
                "unmasked_tokens": int(unmasked_tokens),
                "original_seq_len": original_seq_len,
                "truncated": bool(truncated),
                "answer_in_truncated_tail": answer_in_truncated_tail,
            }
        )

    stats_df = pd.DataFrame(row_stats)

    df_out = df.copy()

    merge_cols = [
        "assistant_budget_units",
        "seq_len",
        "prompt_tokens",
        "unmasked_tokens",
        "original_seq_len",
        "truncated",
        "answer_in_truncated_tail",
    ]

    for col in merge_cols:
        df_out[col] = stats_df[col].values

    pattern_stats = (
        stats_df.groupby("pattern")
        .agg(
            row_count=("pattern", "size"),
            mean_units=("assistant_budget_units", "mean"),
            median_units=("assistant_budget_units", "median"),
            max_units=("assistant_budget_units", "max"),
            total_units=("assistant_budget_units", "sum"),
            mean_seq_len=("seq_len", "mean"),
            max_seq_len=("seq_len", "max"),
            truncated_count=("truncated", "sum"),
        )
        .reset_index()
        .sort_values("total_units", ascending=False)
    )

    total_units = pattern_stats["total_units"].sum()
    pattern_stats["unit_share"] = (
        pattern_stats["total_units"] / total_units
        if total_units > 0
        else 0.0
    )

    row_share = (
        stats_df["pattern"]
        .value_counts(normalize=True)
        .rename("row_share")
        .reset_index()
        .rename(columns={"index": "pattern"})
    )

    pattern_stats = pattern_stats.merge(
        row_share,
        on="pattern",
        how="left",
    )

    pattern_stats["share_gap"] = (
        pattern_stats["unit_share"] - pattern_stats["row_share"]
    )

    print("=" * 100)
    print("[Pattern token budget stats]")
    print(pattern_stats.to_string(index=False))

    print("=" * 100)
    print("Total rows:", len(df_out))
    print("Total assistant budget units:", int(total_units))
    print("Max seq len:", max_seq_len)
    print("Tokenizer used:", tokenizer is not None)

    if tokenizer is not None:
        print("Truncated rows:", int(stats_df["truncated"].sum()))

    print("=" * 100)
    print("[Top 30 longest rows]")
    longest_cols = [
        "id",
        "pattern",
        "solver_rule_name",
        "assistant_budget_units",
        "seq_len",
        "truncated",
        "answer_in_truncated_tail",
    ]

    print(
        stats_df[longest_cols]
        .sort_values("assistant_budget_units", ascending=False)
        .head(30)
        .to_string(index=False)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pattern_stats.to_csv(output_path, index=False)
    print("=" * 100)
    print(f"Saved pattern stats: {output_path}")

    output_data_path = Path(args.output_data)
    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_data_path, index=False)
    print(f"Saved row-level data: {output_data_path}")


if __name__ == "__main__":
    main()