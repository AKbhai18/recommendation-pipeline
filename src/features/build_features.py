"""Stage 5 - Feature Management.

A small, self-contained feature layer (a lightweight stand-in for Feast). It
computes **user**, **item** and **interaction** features from the training
data and persists them as parquet "feature tables" under ``data/features``.

Design goals:
  * Reusable for both training and prediction - the same ``build_features``
    function produces the stored tables, and ``load_feature_store`` reads them
    back so the serving/prediction code uses identical definitions (no
    train/serve skew).
  * Self-documenting - every feature is declared once in ``FEATURE_CATALOG``
    with its entity, data type and description, and ``write_feature_catalog``
    emits that catalog as a markdown table for the report.

Reference timestamp: features that depend on "now" (e.g. days since last
interaction) are computed relative to the maximum timestamp in the training
data, so the feature values are deterministic and reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config, resolve

LOG = get_logger("features.build", log_file="reports/pipeline.log")


# --------------------------------------------------------------------------- #
# Feature catalog - the single source of truth for what each feature means.   #
# --------------------------------------------------------------------------- #
FEATURE_CATALOG = [
    # entity, feature_name, dtype, description
    ("user", "user_avg_rating", "float", "Mean rating the user has given across all training interactions."),
    ("user", "user_activity_count", "int", "Number of interactions the user has in the training data."),
    ("user", "user_distinct_items", "int", "Number of distinct items the user interacted with."),
    ("user", "user_rating_std", "float", "Standard deviation of the user's ratings (rating consistency)."),
    ("user", "user_days_since_last", "int", "Days between the user's last interaction and the reference date."),
    ("user", "user_preferred_category", "str", "Item category the user interacts with most often."),
    ("item", "item_popularity", "int", "Number of interactions the item received (interaction count)."),
    ("item", "item_distinct_users", "int", "Number of distinct users who interacted with the item."),
    ("item", "item_avg_rating", "float", "Mean rating the item received."),
    ("item", "item_rating_std", "float", "Standard deviation of the item's ratings."),
    ("item", "item_release_year", "int", "Release year of the item (from the item catalog)."),
    ("item", "item_category", "str", "Category of the item (from the item catalog)."),
    ("interaction", "rating", "float", "Explicit rating given in the interaction (target for RMSE/MAE)."),
    ("interaction", "event_type", "str", "Type of interaction: rating / view / click / purchase."),
    ("interaction", "interaction_month", "int", "Calendar month of the interaction (1-12), a seasonality signal."),
    ("interaction", "interaction_dayofweek", "int", "Day of week of the interaction (0=Mon..6=Sun)."),
]


def _user_features(train: pd.DataFrame, items: pd.DataFrame, ref_date) -> pd.DataFrame:
    merged = train.merge(items[["item_id", "category"]], on="item_id", how="left")
    g = train.groupby("user_id")
    feats = pd.DataFrame({
        "user_avg_rating": g["rating"].mean().round(4),
        "user_activity_count": g["rating"].count(),
        "user_distinct_items": g["item_id"].nunique(),
        "user_rating_std": g["rating"].std().fillna(0.0).round(4),
        "user_days_since_last": (ref_date - g["timestamp"].max()).dt.days,
    })
    # preferred category = modal category among the user's interactions
    pref = (merged.groupby(["user_id", "category"]).size()
            .reset_index(name="n")
            .sort_values(["user_id", "n"], ascending=[True, False])
            .drop_duplicates("user_id").set_index("user_id")["category"])
    feats["user_preferred_category"] = pref
    return feats.reset_index()


def _item_features(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id")
    feats = pd.DataFrame({
        "item_popularity": g["user_id"].count(),
        "item_distinct_users": g["user_id"].nunique(),
        "item_avg_rating": g["rating"].mean().round(4),
        "item_rating_std": g["rating"].std().fillna(0.0).round(4),
    }).reset_index()
    feats = feats.merge(
        items[["item_id", "release_year", "category"]], on="item_id", how="left"
    ).rename(columns={"release_year": "item_release_year", "category": "item_category"})
    return feats


def _interaction_features(train: pd.DataFrame) -> pd.DataFrame:
    feats = train.copy()
    feats["interaction_month"] = feats["timestamp"].dt.month
    feats["interaction_dayofweek"] = feats["timestamp"].dt.dayofweek
    return feats[["user_id", "item_id", "rating", "event_type",
                  "interaction_month", "interaction_dayofweek", "timestamp"]]


def build_features(prepared: dict | None = None, config: dict | None = None) -> dict:
    cfg = config or load_config()
    proc_dir = resolve(cfg["paths"]["processed"])
    feat_dir = resolve(cfg["paths"]["features"])
    feat_dir.mkdir(parents=True, exist_ok=True)

    if prepared is not None and "train" in prepared:
        train = prepared["train"]
    else:
        train = pd.read_parquet(proc_dir / "train.parquet")

    # items catalog (needed for category / release_year features)
    items = pd.read_csv(resolve(cfg["paths"]["raw"]) / "items.csv", dtype={"item_id": str})

    ref_date = train["timestamp"].max()
    LOG.info("Building features (reference date = %s) ...", ref_date.date())

    user_feats = _user_features(train, items, ref_date)
    item_feats = _item_features(train, items)
    inter_feats = _interaction_features(train)

    user_feats.to_parquet(feat_dir / "user_features.parquet", index=False)
    item_feats.to_parquet(feat_dir / "item_features.parquet", index=False)
    inter_feats.to_parquet(feat_dir / "interaction_features.parquet", index=False)

    write_feature_catalog(resolve(cfg["paths"]["reports"]) / "feature_catalog.md")

    LOG.info("Feature store written: users=%d items=%d interactions=%d",
             len(user_feats), len(item_feats), len(inter_feats))
    return {"user_features": user_feats, "item_features": item_feats,
            "interaction_features": inter_feats}


def load_feature_store(config: dict | None = None) -> dict:
    """Reload the persisted feature tables (used by training and prediction)."""
    cfg = config or load_config()
    feat_dir = resolve(cfg["paths"]["features"])
    return {
        "user_features": pd.read_parquet(feat_dir / "user_features.parquet"),
        "item_features": pd.read_parquet(feat_dir / "item_features.parquet"),
        "interaction_features": pd.read_parquet(feat_dir / "interaction_features.parquet"),
    }


def write_feature_catalog(path: Path) -> None:
    lines = [
        "# Feature Catalog",
        "",
        "All features are computed from the **training** split only (no leakage) "
        "and persisted under `data/features/` as parquet tables. The same "
        "definitions are reused at prediction time via `load_feature_store()`.",
        "",
        "| Entity | Feature | Type | Description |",
        "|--------|---------|------|-------------|",
    ]
    for entity, name, dtype, desc in FEATURE_CATALOG:
        lines.append(f"| {entity} | `{name}` | {dtype} | {desc} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    res = build_features()
    print({k: v.shape for k, v in res.items()})
