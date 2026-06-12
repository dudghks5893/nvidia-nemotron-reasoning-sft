#!/usr/bin/env python3
"""Apply pattern token-budget resampling or create loss_weight column."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nemotron_repro.token_budget import (
    add_pattern_loss_weight_column,
    build_pattern_loss_weights,
    load_target_share,
    resample_by_pattern_token_budget,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--stats", required=True, help="CSV from inspect_token_budget.py")
    parser.add_argument("--target-share", default="configs/token_budget.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["resample", "loss_weight", "both"], default="resample")
    parser.add_argument("--pattern-col", default="pattern")
    parser.add_argument("--token-col", default="assistant_budget_units")
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    stats = pd.read_csv(args.stats)
    target = load_target_share(args.target_share)

    out = df.copy()

    if args.mode in {"loss_weight", "both"}:
        weights = build_pattern_loss_weights(
            pattern_stats_df=stats,
            target_share=target,
            pattern_col=args.pattern_col,
            current_share_col="unit_share",
            weight_power=args.weight_power,
        )
        print("Pattern loss weights:")
        for k, v in sorted(weights.items()):
            print(f"  {k}: {v:.4f}")
        out = add_pattern_loss_weight_column(out, weights, pattern_col=args.pattern_col)

    if args.mode in {"resample", "both"}:
        if args.token_col not in out.columns:
            # Join budget units from current dataframe order/statistics is not enough. Use char length fallback.
            if "solver_solution" in out.columns:
                out[args.token_col] = out["solver_solution"].fillna("").astype(str).str.len()
            else:
                raise ValueError(f"{args.token_col} missing. Run inspect_token_budget and keep the column, or add token counts first.")
        out = resample_by_pattern_token_budget(
            out,
            target_share=target,
            pattern_col=args.pattern_col,
            token_col=args.token_col,
            random_state=args.random_state,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output} rows={len(out)}")


if __name__ == "__main__":
    main()
