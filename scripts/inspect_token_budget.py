#!/usr/bin/env python3
"""Inspect pattern-wise text/token budget.

Use solution length without tokenizer, or pass --model-name for exact tokenizer counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_messages(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


def assistant_text_from_messages(messages):
    parts = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            if msg.get("reasoning_content"):
                parts.append(str(msg.get("reasoning_content", "")))
            if msg.get("content"):
                parts.append(str(msg.get("content", "")))
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None, help="Pattern-level stats CSV")
    parser.add_argument("--output-data", default=None, help="Optional copy of input CSV with assistant_budget_units column")
    parser.add_argument("--model-name", default=None, help="Optional HF tokenizer name/path")
    parser.add_argument("--solution-col", default="solver_solution")
    parser.add_argument("--pattern-col", default="pattern")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.pattern_col not in df.columns and "type" in df.columns:
        df[args.pattern_col] = df["type"]

    if args.model_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        def count_tokens(text):
            return len(tokenizer(str(text), add_special_tokens=False)["input_ids"])
    else:
        def count_tokens(text):
            return len(str(text))

    if "messages" in df.columns:
        assistant_texts = df["messages"].apply(lambda x: assistant_text_from_messages(load_messages(x)))
    else:
        assistant_texts = df[args.solution_col].fillna("").astype(str)

    df["assistant_budget_units"] = assistant_texts.apply(count_tokens)

    stats = (
        df.groupby(args.pattern_col)
        .agg(
            row_count=(args.pattern_col, "size"),
            mean_units=("assistant_budget_units", "mean"),
            median_units=("assistant_budget_units", "median"),
            total_units=("assistant_budget_units", "sum"),
        )
        .reset_index()
        .sort_values("total_units", ascending=False)
    )
    stats["unit_share"] = stats["total_units"] / stats["total_units"].sum()

    print(stats.to_string(index=False))
    print("Total rows:", len(df))
    print("Total units:", int(stats["total_units"].sum()))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.output, index=False)
        print(f"Saved stats: {args.output}")

    if args.output_data:
        Path(args.output_data).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_data, index=False)
        print(f"Saved data with budget units: {args.output_data}")


if __name__ == "__main__":
    main()
