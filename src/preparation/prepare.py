"""Stage 3 & 4 - Data Preparation and Transformation.

Takes the validated/clean interactions and:

  * removes cold users (fewer than ``min_interactions_per_user`` events) that
    cannot be split or meaningfully evaluated,
  * standardises formats (ids as strings, rating as float, timestamp as
    datetime),
  * performs a **time-based train/test split per user** - the most recent
    ``test_fraction`` of each user's history is held out. This mimics the
    real task of predicting future behaviour from past behaviour and avoids
    the leakage a random split would cause.

It then builds the derived datasets the downstream stages need:

  * user_activity.csv       - per-user activity summary,
  * item_popularity.csv     - per-item popularity / rating summary,
  * user_item_matrix.parquet - the (dense) user x item rating matrix.

All outputs are written under ``data/processed``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config, resolve

LOG = get_logger("preparation.prepare", log_file="reports/pipeline.log")


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _drop_cold_users(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    counts = df.groupby("user_id")["item_id"].transform("count")
    kept = df[counts >= min_n].copy()
    LOG.info("Dropped cold users: %d -> %d rows (min_interactions=%d)",
             len(df), len(kept), min_n)
    return kept


def _time_split(df: pd.DataFrame, test_fraction: float):
    """Per-user time-based split: last `test_fraction` of each user -> test."""
    train_parts, test_parts = [], []
    for _, grp in df.groupby("user_id", sort=False):
        grp = grp.sort_values("timestamp")
        n_test = max(1, int(round(len(grp) * test_fraction)))
        train_parts.append(grp.iloc[:-n_test])
        test_parts.append(grp.iloc[-n_test:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    # A user must still have at least one training row.
    valid_users = set(train["user_id"])
    test = test[test["user_id"].isin(valid_users)].reset_index(drop=True)
    LOG.info("Time-based split: train=%d test=%d", len(train), len(test))
    return train, test


def _user_activity_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("user_id")
    out = pd.DataFrame({
        "interaction_count": g["item_id"].count(),
        "avg_rating": g["rating"].mean().round(4),
        "distinct_items": g["item_id"].nunique(),
        "first_interaction": g["timestamp"].min(),
        "last_interaction": g["timestamp"].max(),
    }).reset_index()
    return out


def _item_popularity(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("item_id")
    out = pd.DataFrame({
        "interaction_count": g["user_id"].count(),
        "distinct_users": g["user_id"].nunique(),
        "avg_rating": g["rating"].mean().round(4),
    }).reset_index()
    out["popularity_rank"] = out["interaction_count"].rank(
        ascending=False, method="first"
    ).astype(int)
    return out.sort_values("popularity_rank").reset_index(drop=True)


def prepare(validated: dict | None = None, config: dict | None = None) -> dict:
    cfg = config or load_config()
    proc_dir = resolve(cfg["paths"]["processed"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    if validated is not None and "clean_interactions" in validated:
        clean = validated["clean_interactions"]
    else:
        clean = pd.read_parquet(proc_dir / "interactions_clean.parquet")

    clean = _standardise(clean)
    clean = _drop_cold_users(clean, cfg["preparation"]["min_interactions_per_user"])

    train, test = _time_split(clean, cfg["preparation"]["test_fraction"])

    user_activity = _user_activity_summary(train)
    item_pop = _item_popularity(train)

    # User-item rating matrix (built from the TRAIN set only -> no leakage).
    matrix = train.pivot_table(
        index="user_id", columns="item_id", values="rating", aggfunc="mean"
    )

    # Persist everything.
    train.to_parquet(proc_dir / "train.parquet", index=False)
    test.to_parquet(proc_dir / "test.parquet", index=False)
    user_activity.to_csv(proc_dir / "user_activity.csv", index=False)
    item_pop.to_csv(proc_dir / "item_popularity.csv", index=False)
    matrix.to_parquet(proc_dir / "user_item_matrix.parquet")

    LOG.info("Preparation complete. users=%d items=%d matrix=%s",
             matrix.shape[0], matrix.shape[1], matrix.shape)

    return {
        "train": train, "test": test,
        "user_activity": user_activity, "item_popularity": item_pop,
        "matrix": matrix,
    }


if __name__ == "__main__":
    res = prepare()
    print("train:", res["train"].shape, "test:", res["test"].shape,
          "matrix:", res["matrix"].shape)
