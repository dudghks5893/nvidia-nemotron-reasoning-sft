#!/usr/bin/env python3
"""Validate the high-score-style cipher solver on a train.csv file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nemotron_repro.core import parse_dataframe, solve_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to original Kaggle train.csv")
    parser.add_argument("--output", default="data/generated/cipher_highscore_style_check.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    parsed = parse_dataframe(df)
    cipher = parsed[parsed["pattern"] == "cipher"].copy()
    solved = solve_dataframe(cipher, initialize_cipher=True)
    solved.to_csv(output_path, index=False)

    sol = solved["solver_solution"].fillna("").astype(str)
    rule = solved["solver_rule_name"].fillna("").astype(str)

    print("[Cipher validation]")
    print("rows:", len(solved))
    print("solved:", int(solved["solver_solved"].sum()))
    print("correct:", int(solved["solver_correct"].sum()))
    print("accuracy:", float(solved["solver_correct"].mean()))
    print()
    print("[Rule counts]")
    print(rule.value_counts().sort_index().to_string())
    print()
    print("[Reasoning wording checks]")
    print("contains 'wonderland words':", int(sol.str.contains("wonderland words", regex=False).sum()))
    print("contains 'Added mappings':", int(sol.str.contains("Added mappings", regex=False).sum()))
    print("contains 'untargeted':", int(sol.str.contains("untargeted", regex=False).sum()))
    print("contains temporary no_reverse text:", int(sol.str.contains("reverse mapping is not checked", regex=False).sum()))
    print("contains scan-only text:", int(sol.str.contains("scan-only", regex=False).sum()))
    print()
    print("Saved:", output_path)

    if int(solved["solver_correct"].sum()) != len(solved):
        raise SystemExit("Cipher validation failed: not all cipher rows are correct.")
    if int(sol.str.contains("reverse mapping is not checked", regex=False).sum()) != 0:
        raise SystemExit("Cipher validation failed: temporary no_reverse wording remains.")
    if int(sol.str.contains("scan-only", regex=False).sum()) != 0:
        raise SystemExit("Cipher validation failed: temporary scan-only wording remains.")


if __name__ == "__main__":
    main()
