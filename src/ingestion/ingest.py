"""Stage 1 - Data Ingestion.

Reads the three raw CSV sources (users, items, interactions) from ``data/raw``
into pandas DataFrames. Ingestion is intentionally *lenient*: it does NOT drop
or coerce bad rows here. Keeping ingestion separate from validation means the
validation stage sees the data exactly as it arrived and can report on it.

If the raw files do not yet exist, the synthetic generator is invoked so the
pipeline is runnable from a clean checkout.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.generate_data import generate
from src.utils.common import get_logger, load_config, resolve

LOG = get_logger("ingestion.ingest")


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)  # read as strings; validation sets types
    LOG.info("Ingested %-16s rows=%d cols=%d", path.name, len(df), df.shape[1])
    return df


def ingest(config: dict | None = None) -> dict:
    """Ingest raw user/item/interaction data. Returns a dict of DataFrames."""
    cfg = config or load_config()
    raw_dir = resolve(cfg["paths"]["raw"])

    required = ["users.csv", "items.csv", "interactions.csv"]
    if not all((raw_dir / f).exists() for f in required):
        LOG.info("Raw files missing -> generating synthetic dataset first.")
        generate(cfg)

    data = {
        "users": _read_csv(raw_dir / "users.csv"),
        "items": _read_csv(raw_dir / "items.csv"),
        "interactions": _read_csv(raw_dir / "interactions.csv"),
    }
    LOG.info("Ingestion complete: %s",
             {k: len(v) for k, v in data.items()})
    return data


if __name__ == "__main__":
    out = ingest()
    for name, df in out.items():
        print(f"\n== {name} ==")
        print(df.head())
