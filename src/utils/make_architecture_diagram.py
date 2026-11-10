"""Render the pipeline architecture diagram to reports/architecture_diagram.png.

Pure-matplotlib (no Graphviz dependency) so it runs anywhere. The diagram shows
the eight pipeline stages, the storage layers they read/write, the experiment-
tracking and orchestration layers, and the flow of data between them.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "reports" / "architecture_diagram.png"

STAGE_COLOR = "#2563eb"
STORE_COLOR = "#059669"
OPS_COLOR = "#7c3aed"
BG = "#f8fafc"


def _box(ax, x, y, w, h, text, color, text_color="white", fontsize=10):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor="white", facecolor=color, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, weight="bold", zorder=3)
    return (x + w / 2, y + h / 2)


def _arrow(ax, p1, p2, color="#334155"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.6, color=color, zorder=1,
        shrinkA=6, shrinkB=6,
    ))


def main():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.text(6.5, 7.6, "End-to-End Recommendation Data-Management Pipeline",
            ha="center", va="center", fontsize=15, weight="bold", color="#0f172a")

    # --- Stage row (the 8 pipeline stages) ---------------------------------
    stages = [
        "1. Data\nIngestion", "2. Data\nValidation", "3. Data\nPreparation",
        "4. Data\nTransformation", "5. Feature\nManagement",
        "6. Model\nTraining", "7. Model\nEvaluation",
    ]
    w, h, gap = 1.55, 1.0, 0.22
    x0, y_stage = 0.3, 4.2
    centers = []
    for i, s in enumerate(stages):
        cx = _box(ax, x0 + i * (w + gap), y_stage, w, h, s, STAGE_COLOR)
        centers.append(cx)
    for i in range(len(centers) - 1):
        _arrow(ax, (centers[i][0] + w / 2, y_stage + h / 2),
               (centers[i + 1][0] - w / 2, y_stage + h / 2))

    # --- Storage row (below stages) ----------------------------------------
    stores = [
        ("data/raw\n(CSV)", 0.3),
        ("data/errors\n(invalid rows)", 2.07),
        ("data/processed\n(train/test,\nmatrix, Parquet)", 3.84),
        ("data/features\n(feature store,\nParquet)", 6.5),
        ("models/\n(pickled\nmodels)", 9.0),
        ("reports/\n(metrics,\ncatalog, logs)", 11.0),
    ]
    y_store = 1.9
    for text, sx in stores:
        _box(ax, sx, y_store, 1.75, 1.05, text, STORE_COLOR, fontsize=8.5)

    # vertical links stage -> storage
    links = [
        (centers[0], (1.17, y_store + 1.05)),   # ingestion -> raw
        (centers[1], (2.95, y_store + 1.05)),   # validation -> errors
        (centers[2], (4.7, y_store + 1.05)),    # preparation -> processed
        (centers[4], (7.37, y_store + 1.05)),   # features -> feature store
        (centers[5], (9.87, y_store + 1.05)),   # training -> models
        (centers[6], (11.87, y_store + 1.05)),  # evaluation -> reports
    ]
    for (scx, scy), tgt in links:
        _arrow(ax, (scx, y_stage), tgt, color="#64748b")

    # --- Orchestration band (top) ------------------------------------------
    _box(ax, 0.3, 6.1, 12.4, 0.8,
         "Orchestration: Apache Airflow DAG / structured Python workflow\n"
         "(topological task scheduling, reproducible via fixed seed)",
         OPS_COLOR, fontsize=10)
    _arrow(ax, (6.5, 6.1), (6.5, y_stage + h), color=OPS_COLOR)

    # --- Experiment tracking (lower band, under training/eval) -------------
    mlf = _box(ax, 9.0, 0.45, 3.75, 0.95,
               "MLflow Experiment Tracking\n(params, metrics, model artifacts)",
               OPS_COLOR, fontsize=9)
    # training + evaluation both log to MLflow
    _arrow(ax, (10.0, y_store), (10.3, 1.4), color=OPS_COLOR)
    _arrow(ax, (11.87, y_store), (11.5, 1.4), color=OPS_COLOR)

    # --- Legend ------------------------------------------------------------
    handles = [
        mpatches.Patch(color=STAGE_COLOR, label="Pipeline stage"),
        mpatches.Patch(color=STORE_COLOR, label="Storage layer"),
        mpatches.Patch(color=OPS_COLOR, label="Orchestration / tracking"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
