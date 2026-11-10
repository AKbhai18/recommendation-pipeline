"""Generate report figures: model comparison chart + MLflow runs summary.

* reports/screenshots/07_model_comparison.png - grouped bar chart comparing the
  three models on the ranking metrics plus a rating-error panel.
* reports/screenshots/08_mlflow_runs.png - a readable table of the MLflow runs
  (experiment, run name, key params, key metrics) read back from the MLflow
  tracking store, standing in for the MLflow UI in a headless environment.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.common import load_config, resolve

ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT / "reports" / "screenshots"


def model_comparison_chart():
    df = pd.read_csv(resolve("reports/evaluation_results.csv"))
    df = df.set_index("model")
    ranking = ["precision@10", "recall@10", "ndcg@10", "coverage", "diversity"]
    rating = ["rmse", "mae"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [3, 1.2]}
    )
    colors = ["#94a3b8", "#2563eb", "#7c3aed"]

    x = np.arange(len(ranking))
    width = 0.25
    for i, model in enumerate(df.index):
        ax1.bar(x + (i - 1) * width, df.loc[model, ranking].values,
                width, label=model, color=colors[i % len(colors)])
    ax1.set_xticks(x)
    ax1.set_xticklabels(ranking, rotation=15)
    ax1.set_title("Ranking metrics @K=10  (higher is better)", weight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    x2 = np.arange(len(rating))
    for i, model in enumerate(df.index):
        ax2.bar(x2 + (i - 1) * width, df.loc[model, rating].values,
                width, label=model, color=colors[i % len(colors)])
    ax2.set_xticks(x2)
    ax2.set_xticklabels(rating)
    ax2.set_title("Rating error  (lower is better)", weight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Comparison - Popularity vs Collaborative Filtering vs "
                 "Matrix Factorisation", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    SHOTS.mkdir(parents=True, exist_ok=True)
    out = SHOTS / "07_model_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.name}")


def mlflow_runs_screenshot():
    cfg = load_config()
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        tracking_dir = resolve(cfg["training"]["mlflow_tracking_dir"])
        mlflow.set_tracking_uri(f"file:{tracking_dir}")
        client = MlflowClient()
        exp = client.get_experiment_by_name(cfg["training"]["mlflow_experiment"])
        runs = client.search_runs([exp.experiment_id])
        rows = []
        for r in runs:
            rows.append({
                "run_name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                "precision@10": r.data.metrics.get("precision_at_10"),
                "recall@10": r.data.metrics.get("recall_at_10"),
                "ndcg@10": r.data.metrics.get("ndcg_at_10"),
                "coverage": r.data.metrics.get("coverage"),
                "rmse": r.data.metrics.get("rmse"),
                "mae": r.data.metrics.get("mae"),
            })
        table = pd.DataFrame(rows)
    except Exception as exc:  # pragma: no cover
        table = pd.DataFrame({"note": [f"MLflow read failed: {exc}"]})

    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.set_title(
        f"MLflow Experiment: '{cfg['training']['mlflow_experiment']}'  -  "
        f"{len(table)} tracked runs  (tracking URI: file:{cfg['training']['mlflow_tracking_dir']})",
        fontsize=11, weight="bold", loc="left", pad=16,
    )
    # Round only the numeric columns; keep run_name as text.
    disp = table.copy()
    for c in disp.columns:
        if c != "run_name" and c != "note":
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(4)
    cell_text = disp.astype(object).where(pd.notna(disp), "-").values

    tbl = ax.table(cellText=cell_text,
                   colLabels=disp.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (row, _), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2563eb")
            cell.set_text_props(color="white", weight="bold")
    out = SHOTS / "08_mlflow_runs.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"Wrote {out.name}")


def main():
    model_comparison_chart()
    mlflow_runs_screenshot()


if __name__ == "__main__":
    main()
