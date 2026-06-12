#!/usr/bin/env python3
"""Select train rows by pattern/mode/rule config and optionally split val/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nemotron_repro.core import (
    build_selected_dataset_by_config,
    print_distribution,
    split_train_val_test_from_selection_rule_aware,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/default_pattern_configs.json")
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-val", default=None)
    parser.add_argument("--output-test", default=None)
    parser.add_argument("--split-val-test", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "row_id" not in df.columns:
        df["row_id"] = df.index

    with open(args.config, "r", encoding="utf-8") as f:
        pattern_configs = json.load(f)

    selected_df = build_selected_dataset_by_config(
        df,
        pattern_configs=pattern_configs,
        random_state=args.random_state,
        shuffle=True,
    )
    print_distribution(selected_df, "Selected train distribution")

    out_train = Path(args.output_train)
    out_train.parent.mkdir(parents=True, exist_ok=True)

    if args.split_val_test:
        train_df, val_df, test_df = split_train_val_test_from_selection_rule_aware(
            original_df=df,
            selected_df=selected_df,
            val_ratio=0.5,
            random_state=args.random_state,
        )
        train_df.to_csv(out_train, index=False)
        if args.output_val:
            Path(args.output_val).parent.mkdir(parents=True, exist_ok=True)
            val_df.to_csv(args.output_val, index=False)
        if args.output_test:
            Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)
            test_df.to_csv(args.output_test, index=False)
        print(f"Saved train: {out_train} rows={len(train_df)}")
        if args.output_val:
            print(f"Saved val: {args.output_val} rows={len(val_df)}")
        if args.output_test:
            print(f"Saved test: {args.output_test} rows={len(test_df)}")
    else:
        selected_df.to_csv(out_train, index=False)
        print(f"Saved train: {out_train} rows={len(selected_df)}")


if __name__ == "__main__":
    main()
