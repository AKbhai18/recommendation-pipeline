"""Stage 0 (data source): generate a documented, MovieLens-style dataset.

The assignment allows a public dataset *or* clearly documented synthetic data.
We generate synthetic data so the repo is fully self-contained and the run is
deterministic (fixed seed), while still mirroring the structure of MovieLens:

  * users.csv         - user_id, age, gender, occupation, signup_date
  * items.csv         - item_id, title, category, release_year
  * interactions.csv  - user_id, item_id, rating, timestamp, event_type

Ratings follow a realistic distribution (users have latent taste, items have
latent quality), and a small, configurable fraction of rows are deliberately
corrupted (missing values, out-of-range ratings, bad timestamps, duplicates,
dangling foreign keys) so the validation stage has genuine issues to detect.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config, resolve, set_seed

LOG = get_logger("ingestion.generate")

OCCUPATIONS = [
    "student", "engineer", "teacher", "artist", "doctor",
    "writer", "retired", "technician", "manager", "other",
]
GENDERS = ["M", "F", "U"]
TITLE_WORDS_A = ["Silent", "Broken", "Golden", "Last", "Hidden", "Crimson",
                 "Electric", "Midnight", "Lost", "Eternal"]
TITLE_WORDS_B = ["Horizon", "Kingdom", "Promise", "Shadow", "Dream",
                 "Voyage", "Empire", "Legacy", "Secret", "Dawn"]


def _generate_users(n_users: int, start, end, rng) -> pd.DataFrame:
    signup = pd.to_datetime(
        rng.integers(start.value, end.value, size=n_users)
    ).normalize()
    return pd.DataFrame({
        "user_id": [f"U{1000 + i}" for i in range(n_users)],
        "age": rng.integers(15, 75, size=n_users),
        "gender": rng.choice(GENDERS, size=n_users, p=[0.48, 0.48, 0.04]),
        "occupation": rng.choice(OCCUPATIONS, size=n_users),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })


def _generate_items(n_items: int, categories, rng) -> pd.DataFrame:
    titles = [
        f"{rng.choice(TITLE_WORDS_A)} {rng.choice(TITLE_WORDS_B)} {rng.integers(1, 4)}"
        for _ in range(n_items)
    ]
    return pd.DataFrame({
        "item_id": [f"I{2000 + i}" for i in range(n_items)],
        "title": titles,
        "category": rng.choice(categories, size=n_items),
        "release_year": rng.integers(1990, 2024, size=n_items),
    })


def _generate_interactions(users, items, n_inter, scale, start, end, rng) -> pd.DataFrame:
    n_users, n_items = len(users), len(items)

    # Latent-factor taste model. Each user and item gets a short latent vector;
    # a user's affinity for an item is their dot product. This injects GENUINE
    # collaborative structure (users with similar taste rate items similarly),
    # which is exactly what collaborative filtering / matrix factorisation are
    # designed to recover - so the personalised models can beat the popularity
    # baseline. Biases add per-user generosity and per-item quality on top.
    n_factors = 8
    user_factors = rng.normal(0, 1.0, size=(n_users, n_factors))
    item_factors = rng.normal(0, 1.0, size=(n_items, n_factors))
    user_bias = rng.normal(0, 0.5, size=n_users)
    item_quality = rng.normal(0, 0.6, size=n_items)

    # Base item popularity: a power-law over a random ordering gives a realistic
    # long tail that still covers essentially every item.
    order = rng.permutation(n_items)
    ranks = np.empty(n_items)
    ranks[order] = np.arange(1, n_items + 1)
    base_pop = ranks ** (-0.6)

    # Each user interacts with items through a mix of global popularity and
    # personal affinity -> the observed items are themselves personalised.
    affinity_all = user_factors @ item_factors.T            # users x items
    u_idx = rng.integers(0, n_users, size=n_inter)

    i_idx = np.empty(n_inter, dtype=int)
    # Draw each user's items from their personalised preference distribution.
    for u in np.unique(u_idx):
        rows = np.where(u_idx == u)[0]
        pref = base_pop * np.exp(0.6 * affinity_all[u])     # blend pop + taste
        pref = pref / pref.sum()
        i_idx[rows] = rng.choice(n_items, size=len(rows), p=pref)

    affinity = (user_factors[u_idx] * item_factors[i_idx]).sum(axis=1)
    raw = (3.2 + 0.45 * affinity + user_bias[u_idx]
           + item_quality[i_idx] + rng.normal(0, 0.4, n_inter))
    rating = np.clip(np.round(raw * 2) / 2, scale[0], scale[1])  # 1..5 in 0.5 steps

    ts = pd.to_datetime(rng.integers(start.value, end.value, size=n_inter))
    event_type = rng.choice(
        ["rating", "view", "click", "purchase"],
        size=n_inter, p=[0.55, 0.25, 0.15, 0.05],
    )

    return pd.DataFrame({
        "user_id": users["user_id"].values[u_idx],
        "item_id": items["item_id"].values[i_idx],
        "rating": rating,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": event_type,
    })


def _inject_dirty_rows(inter: pd.DataFrame, frac: float, rng) -> pd.DataFrame:
    """Deliberately corrupt a small fraction of rows (documented noise).

    This is what the validation stage is expected to catch and quarantine.
    """
    inter = inter.copy()
    n = len(inter)
    k = max(1, int(n * frac))

    def pick():
        return rng.choice(n, size=k, replace=False)

    # 1) missing ratings
    inter.loc[pick(), "rating"] = np.nan
    # 2) out-of-range ratings (invalid)
    inter.loc[pick(), "rating"] = rng.choice([0.0, 6.0, 7.5, -1.0], size=k)
    # 3) invalid / unparseable timestamps
    inter.loc[pick(), "timestamp"] = "not-a-date"
    # 4) dangling foreign keys (referential integrity violations)
    inter.loc[pick(), "item_id"] = "I999999"
    # 5) exact duplicate rows appended at the end
    dup = inter.iloc[rng.choice(n, size=k, replace=False)].copy()
    inter = pd.concat([inter, dup], ignore_index=True)

    return inter.sample(frac=1.0, random_state=1).reset_index(drop=True)


def generate(config: dict | None = None) -> dict:
    """Generate the three raw CSVs and return their paths + row counts."""
    cfg = config or load_config()
    set_seed(cfg["project"]["seed"])
    rng = np.random.default_rng(cfg["project"]["seed"])

    g = cfg["data_generation"]
    start = pd.Timestamp(g["start_date"])
    end = pd.Timestamp(g["end_date"])

    LOG.info("Generating synthetic MovieLens-style dataset ...")
    users = _generate_users(g["n_users"], start, end, rng)
    items = _generate_items(g["n_items"], g["categories"], rng)
    inter = _generate_interactions(
        users, items, g["n_interactions"], g["rating_scale"], start, end, rng
    )
    inter = _inject_dirty_rows(inter, g["dirty_fraction"], rng)

    raw_dir = resolve(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    users.to_csv(raw_dir / "users.csv", index=False)
    items.to_csv(raw_dir / "items.csv", index=False)
    inter.to_csv(raw_dir / "interactions.csv", index=False)

    LOG.info("Wrote users=%d items=%d interactions=%d to %s",
             len(users), len(items), len(inter), raw_dir)
    return {
        "users": len(users),
        "items": len(items),
        "interactions": len(inter),
        "raw_dir": str(raw_dir),
    }


if __name__ == "__main__":
    print(generate())
