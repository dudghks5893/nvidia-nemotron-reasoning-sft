#!/usr/bin/env python3
# scripts/oversample_patterns.py

"""
Oversample or downsample SFT rows by pattern target row counts.

This script does NOT train a model.
It only creates a new CSV with pattern-level row counts adjusted.

Typical usage:
    PYTHONPATH=src python scripts/oversample_patterns.py \
      --input data/generated/sft_correct_only.csv \
      --config configs/pattern_row_targets.json \
      --output data/generated/sft_correct_only_row_balanced.csv \
      --pattern-col pattern
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV. Usually data/generated/sft_correct_only.csv",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="JSON file containing pattern -> target row count.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--pattern-col",
        default="pattern",
        help="Pattern column name.",
    )

    parser.add_argument(
        "--id-col",
        default="id",
        help="ID column name.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--keep-unspecified",
        action="store_true",
        help="Keep patterns that are not specified in the config.",
    )

    return parser.parse_args()


def load_json(path: str | Path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.pattern_col not in df.columns:
        raise ValueError(f"Pattern column not found: {args.pattern_col}")

    targets = load_json(args.config)

    parts = []
    summary_rows = []

    existing_patterns = sorted(df[args.pattern_col].dropna().unique())

    for pattern in existing_patterns:
        group = df[df[args.pattern_col] == pattern].copy()
        current_count = len(group)

        if pattern not in targets:
            if args.keep_unspecified:
                sampled = group.copy()
                target_count = current_count
                replace = False
                action = "keep_unspecified"
            else:
                print(f"Skip unspecified pattern: {pattern}")
                continue
        else:
            target_count = int(targets[pattern])

            if target_count <= 0:
                print(f"Skip pattern with target_count <= 0: {pattern}")
                continue

            replace = target_count > current_count

            sampled = group.sample(
                n=target_count,
                replace=replace,
                random_state=args.random_state,
            ).copy()

            if target_count > current_count:
                action = "oversample"
            elif target_count < current_count:
                action = "downsample"
            else:
                action = "keep"

        sampled["row_balance_pattern"] = pattern
        sampled["row_balance_original_count"] = current_count
        sampled["row_balance_target_count"] = target_count
        sampled["row_balance_replace"] = bool(replace)
        sampled["row_balance_action"] = action

        parts.append(sampled)

        summary_rows.append(
            {
                "pattern": pattern,
                "original_count": current_count,
                "target_count": target_count,
                "output_count": len(sampled),
                "replace": bool(replace),
                "action": action,
            }
        )

    if not parts:
        raise ValueError("No rows selected. Check your config and pattern column.")

    out_df = pd.concat(parts, axis=0).sample(
        frac=1.0,
        random_state=args.random_state,
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("pattern")

    summary_path = output_path.with_suffix(".summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("=" * 100)
    print("[Row balance summary]")
    print(summary_df.to_string(index=False))

    print("=" * 100)
    print("[Output pattern distribution]")
    print(out_df[args.pattern_col].value_counts())

    if args.id_col in out_df.columns:
        id_counts = out_df[args.id_col].value_counts()
        duplicated_ids = int((id_counts > 1).sum())
        extra_duplicate_rows = int((id_counts - 1).clip(lower=0).sum())

        print("=" * 100)
        print("[Duplicate ID stats]")
        print("total rows:", len(out_df))
        print("unique ids:", out_df[args.id_col].nunique())
        print("duplicated ids:", duplicated_ids)
        print("extra duplicate rows:", extra_duplicate_rows)

        print("=" * 100)
        print("[Top duplicated IDs]")
        print(id_counts.head(30).to_string())

    print("=" * 100)
    print(f"Saved: {output_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()