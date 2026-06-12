"""Pattern token-budget and loss-weight helper functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_target_share(target_share: dict, existing_patterns):
    filtered = {k: v for k, v in target_share.items() if k in set(existing_patterns)}
    if not filtered:
        raise ValueError("No target_share keys match existing patterns.")
    s = sum(filtered.values())
    return {k: v / s for k, v in filtered.items()}


def resample_by_pattern_token_budget(
    df: pd.DataFrame,
    target_share: dict,
    pattern_col: str = "pattern",
    token_col: str = "assistant_budget_units",
    total_token_budget: int | None = None,
    random_state: int = 42,
):
    df = df.copy()
    existing = sorted(df[pattern_col].dropna().unique())
    target = normalize_target_share(target_share, existing)
    if total_token_budget is None:
        total_token_budget = int(df[token_col].sum())
    parts = []
    for pattern, group in df.groupby(pattern_col):
        if pattern not in target:
            parts.append(group)
            continue
        mean_tokens = group[token_col].mean()
        if mean_tokens <= 0:
            parts.append(group)
            continue
        target_tokens = total_token_budget * target[pattern]
        target_rows = max(int(round(target_tokens / mean_tokens)), 1)
        replace = target_rows > len(group)
        part = group.sample(n=target_rows, replace=replace, random_state=random_state).copy()
        part["resample_target_rows"] = target_rows
        part["resample_replace"] = replace
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)


def build_pattern_loss_weights(
    pattern_stats_df: pd.DataFrame,
    target_share: dict,
    pattern_col: str,
    current_share_col: str = "unit_share",
    min_weight: float = 0.30,
    max_weight: float = 2.00,
    weight_power: float = 0.5,
):
    weights = {}
    for _, row in pattern_stats_df.iterrows():
        pattern = row[pattern_col]
        current = float(row[current_share_col])
        if pattern not in target_share or current <= 0:
            weights[pattern] = 1.0
            continue
        raw = (float(target_share[pattern]) / current) ** weight_power
        weights[pattern] = float(np.clip(raw, min_weight, max_weight))
    mean_w = np.mean(list(weights.values()))
    if mean_w > 0:
        weights = {k: v / mean_w for k, v in weights.items()}
    return weights


def add_pattern_loss_weight_column(df: pd.DataFrame, weights: dict, pattern_col: str = "pattern", default_weight: float = 1.0):
    df = df.copy()
    df["loss_weight"] = df[pattern_col].map(weights).fillna(default_weight).astype(float)
    return df


def load_target_share(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
