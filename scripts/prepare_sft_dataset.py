#!/usr/bin/env python3
"""Build solved SFT dataframe from raw Kaggle train.csv or matched CSV.

Example:
  python scripts/prepare_sft_dataset.py \
    --input data/raw/train.csv \
    --output data/generated/sft_train.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from nemotron_repro.core import (
    add_training_messages,
    initialize_cipher_vocab_from_dataframe,
    metric_like_match,
    parse_dataframe,
    print_distribution,
    run_solver_by_pattern,
)


def serialize_messages(value):
    return json.dumps(value, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path. Usually train.csv or problem_ids_matched.csv")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--use-reasoning-content", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug row limit")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    if "pattern" not in df.columns and "type" in df.columns:
        df["pattern"] = df["type"]

    df = parse_dataframe(df)
    initialize_cipher_vocab_from_dataframe(df)

    print("Running solvers...")
    df["solver_result"] = df.progress_apply(run_solver_by_pattern, axis=1)
    df["solver_solved"] = df["solver_result"].apply(lambda x: bool(x.get("solved", False)))
    df["solver_answer"] = df["solver_result"].apply(lambda x: x.get("answer"))
    df["solver_solution"] = df["solver_result"].apply(lambda x: x.get("solution"))
    df["solver_rule_name"] = df["solver_result"].apply(lambda x: x.get("rule_name"))
    df["solver_name"] = df["solver_result"].apply(lambda x: x.get("solver_name"))
    df["solver_correct"] = df.progress_apply(
        lambda row: metric_like_match(row.get("answer"), row.get("solver_answer")),
        axis=1,
    )

    df = add_training_messages(df, use_reasoning_content=args.use_reasoning_content)
    df["messages_json"] = df["messages"].apply(serialize_messages)

    print_distribution(df, "Solved SFT distribution")

    # Store solver_result/messages as JSON-friendly strings for CSV portability.
    save_df = df.copy()
    save_df["solver_result"] = save_df["solver_result"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    save_df["messages"] = save_df["messages_json"]
    save_df = save_df.drop(columns=["messages_json"])

    save_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Rows: {len(save_df)}")


if __name__ == "__main__":
    main()
