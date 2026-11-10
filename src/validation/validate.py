"""Stage 2 - Data Validation.

Automated data-quality gate. It runs two layers of checks:

  1. Schema checks (Pandera) - column presence, data types, and value ranges
     for users, items and interactions.
  2. Custom business checks - missing values, duplicates, invalid ratings,
     invalid/unparseable timestamps, and referential integrity between the
     interaction foreign keys and the user/item dimension tables.

Every invalid interaction row is tagged with a human-readable reason and
written to ``data/errors/invalid_interactions.csv`` (the error dataset). A
machine-readable summary is written to ``data/errors/validation_report.json``
and a human-readable report to ``reports/validation_report.md``.

The clean interactions (those that pass every check, de-duplicated) are
returned for the downstream preparation stage.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from src.utils.common import get_logger, load_config, resolve

LOG = get_logger("validation.validate", log_file="reports/pipeline.log")


# --------------------------------------------------------------------------- #
# Pandera schemas for the two dimension tables (structural validation).       #
# --------------------------------------------------------------------------- #
def _user_schema() -> DataFrameSchema:
    return DataFrameSchema({
        "user_id": Column(str, nullable=False, unique=True),
        "age": Column(int, Check.in_range(10, 100), coerce=True, nullable=False),
        "gender": Column(str, Check.isin(["M", "F", "U"]), nullable=False),
        "occupation": Column(str, nullable=False),
        "signup_date": Column(str, nullable=False),
    })


def _item_schema() -> DataFrameSchema:
    return DataFrameSchema({
        "item_id": Column(str, nullable=False, unique=True),
        "title": Column(str, nullable=False),
        "category": Column(str, nullable=False),
        "release_year": Column(
            int, Check.in_range(1900, 2030), coerce=True, nullable=False
        ),
    })


def _validate_dimension(df: pd.DataFrame, schema: DataFrameSchema, name: str) -> dict:
    """Validate a dimension table; return a small result record."""
    try:
        schema.validate(df, lazy=True)
        LOG.info("[%s] schema validation PASSED (rows=%d)", name, len(df))
        return {"table": name, "rows": len(df), "schema_ok": True, "errors": 0}
    except pa.errors.SchemaErrors as exc:
        n_err = len(exc.failure_cases)
        LOG.warning("[%s] schema validation found %d issue(s)", name, n_err)
        return {"table": name, "rows": len(df), "schema_ok": False, "errors": n_err}


# --------------------------------------------------------------------------- #
# Custom row-level checks for the interactions fact table.                    #
# --------------------------------------------------------------------------- #
def _validate_interactions(inter: pd.DataFrame, users: pd.DataFrame,
                           items: pd.DataFrame, cfg: dict):
    """Row-level validation. Returns (clean_df, error_df, stats)."""
    v = cfg["validation"]
    df = inter.copy()
    df["_row"] = range(len(df))

    # Coerce types we can reason about. Bad values become NaN/NaT (then flagged).
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    df["ts_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")

    valid_users = set(users["user_id"])
    valid_items = set(items["item_id"])
    ts_lo = pd.Timestamp(v["valid_from"])
    ts_hi = pd.Timestamp(v["valid_to"])

    reasons = pd.Series([""] * len(df), index=df.index)

    def flag(mask: pd.Series, reason: str):
        hit = mask.fillna(False)
        reasons.loc[hit] = (reasons.loc[hit] + ";" + reason).str.lstrip(";")

    # 1) missing critical values
    flag(df["user_id"].isna() | (df["user_id"] == ""), "missing_user_id")
    flag(df["item_id"].isna() | (df["item_id"] == ""), "missing_item_id")
    flag(df["rating_num"].isna(), "missing_or_nonnumeric_rating")
    # 2) invalid rating range
    flag((df["rating_num"] < v["rating_min"]) | (df["rating_num"] > v["rating_max"]),
         "rating_out_of_range")
    # 3) invalid timestamps (unparseable or outside the plausible window)
    flag(df["ts_parsed"].isna(), "invalid_timestamp")
    flag((df["ts_parsed"] < ts_lo) | (df["ts_parsed"] > ts_hi),
         "timestamp_out_of_window")
    # 4) referential integrity
    flag(~df["user_id"].isin(valid_users), "unknown_user_id")
    flag(~df["item_id"].isin(valid_items), "unknown_item_id")

    df["error_reason"] = reasons
    bad_mask = df["error_reason"] != ""

    # 5) duplicates among the otherwise-valid rows (same user+item+timestamp)
    clean = df[~bad_mask].copy()
    dup_mask = clean.duplicated(subset=["user_id", "item_id", "timestamp"], keep="first")
    dup_rows = clean[dup_mask].copy()
    dup_rows["error_reason"] = "duplicate_interaction"
    clean = clean[~dup_mask].copy()

    error_df = pd.concat([df[bad_mask], dup_rows], ignore_index=True)

    stats = {
        "total_rows": int(len(inter)),
        "clean_rows": int(len(clean)),
        "error_rows": int(len(error_df)),
        "missing_rating": int(df["rating_num"].isna().sum()),
        "rating_out_of_range": int(
            ((df["rating_num"] < v["rating_min"]) |
             (df["rating_num"] > v["rating_max"])).fillna(False).sum()
        ),
        "invalid_timestamp": int(df["ts_parsed"].isna().sum()),
        "unknown_user_id": int((~df["user_id"].isin(valid_users)).sum()),
        "unknown_item_id": int((~df["item_id"].isin(valid_items)).sum()),
        "duplicate_interaction": int(len(dup_rows)),
    }

    # Return clean rows with normalised types for downstream stages.
    clean_out = clean[["user_id", "item_id", "event_type"]].copy()
    clean_out["rating"] = clean["rating_num"].values
    clean_out["timestamp"] = clean["ts_parsed"].values
    return clean_out.reset_index(drop=True), error_df, stats


def _write_markdown_report(stats: dict, dim_results: list, path: Path) -> None:
    pct = (stats["error_rows"] / stats["total_rows"] * 100) if stats["total_rows"] else 0
    lines = [
        "# Data Validation Report",
        "",
        f"_Generated: {datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        "## Dimension tables (schema validation)",
        "",
        "| Table | Rows | Schema OK | Issues |",
        "|-------|------|-----------|--------|",
    ]
    for r in dim_results:
        lines.append(
            f"| {r['table']} | {r['rows']} | {'yes' if r['schema_ok'] else 'no'} | {r['errors']} |"
        )
    lines += [
        "",
        "## Interactions (row-level checks)",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total ingested rows | {stats['total_rows']} |",
        f"| Clean rows (passed) | {stats['clean_rows']} |",
        f"| Error rows (quarantined) | {stats['error_rows']} ({pct:.2f}%) |",
        f"| Missing / non-numeric rating | {stats['missing_rating']} |",
        f"| Rating out of range | {stats['rating_out_of_range']} |",
        f"| Invalid timestamp | {stats['invalid_timestamp']} |",
        f"| Unknown user_id (FK) | {stats['unknown_user_id']} |",
        f"| Unknown item_id (FK) | {stats['unknown_item_id']} |",
        f"| Duplicate interactions | {stats['duplicate_interaction']} |",
        "",
        "Invalid rows are quarantined in `data/errors/invalid_interactions.csv` "
        "with a per-row `error_reason`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def validate(data: dict | None = None, config: dict | None = None) -> dict:
    """Run the full validation stage. Returns clean data + report paths."""
    cfg = config or load_config()
    if data is None:
        from src.ingestion.ingest import ingest
        data = ingest(cfg)

    users, items, inter = data["users"], data["items"], data["interactions"]

    LOG.info("Validating dimension tables ...")
    dim_results = [
        _validate_dimension(users, _user_schema(), "users"),
        _validate_dimension(items, _item_schema(), "items"),
    ]

    LOG.info("Validating interactions (row-level) ...")
    clean, errors, stats = _validate_interactions(inter, users, items, cfg)

    err_dir = resolve(cfg["paths"]["errors"])
    err_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(err_dir / "invalid_interactions.csv", index=False)

    report = {"dimensions": dim_results, "interactions": stats}
    (err_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    reports_dir = resolve(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_markdown_report(stats, dim_results, reports_dir / "validation_report.md")

    LOG.info("Validation done: clean=%d quarantined=%d (%.2f%% bad)",
             stats["clean_rows"], stats["error_rows"],
             stats["error_rows"] / stats["total_rows"] * 100)

    # Persist the clean interactions for the next stage.
    proc_dir = resolve(cfg["paths"]["processed"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(proc_dir / "interactions_clean.parquet", index=False)

    return {"clean_interactions": clean, "users": users, "items": items,
            "errors": errors, "report": report}


if __name__ == "__main__":
    res = validate()
    print(json.dumps(res["report"], indent=2))
