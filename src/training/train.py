"""Stage 6 - Model Training (with MLflow experiment tracking).

Trains the three recommenders (popularity baseline, item-based CF, SVD matrix
factorisation), evaluates each on the held-out test split, and logs the
parameters, metrics and the serialised model artifact to MLflow. A consolidated
comparison table is written to ``reports/evaluation_results.csv`` / ``.md`` and
the trained models are pickled under ``models/``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import mlflow
import pandas as pd

from src.evaluation.metrics import evaluate_model
from src.features.build_features import load_feature_store
from src.training.models import (ItemBasedCF, MatrixFactorization,
                                 PopularityRecommender)
from src.utils.common import get_logger, load_config, resolve, set_seed

LOG = get_logger("training.train", log_file="reports/pipeline.log")


def _save_model(model, path: Path) -> None:
    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def train(config: dict | None = None) -> dict:
    cfg = config or load_config()
    set_seed(cfg["project"]["seed"])

    proc_dir = resolve(cfg["paths"]["processed"])
    models_dir = resolve(cfg["paths"]["models"])
    reports_dir = resolve(cfg["paths"]["reports"])
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(proc_dir / "train.parquet")
    test_df = pd.read_parquet(proc_dir / "test.parquet")
    item_features = load_feature_store(cfg)["item_features"]

    k = cfg["evaluation"]["k"]
    rel = cfg["evaluation"]["relevance_threshold"]

    # MLflow tracking -> local directory store (no server needed).
    tracking_dir = resolve(cfg["training"]["mlflow_tracking_dir"])
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_dir}")
    mlflow.set_experiment(cfg["training"]["mlflow_experiment"])

    specs = [
        (PopularityRecommender(), {"type": "popularity_baseline"}),
        (ItemBasedCF(top_k_neighbours=cfg["training"]["cf_top_k_neighbours"]),
         {"type": "item_based_cf",
          "top_k_neighbours": cfg["training"]["cf_top_k_neighbours"]}),
        (MatrixFactorization(n_factors=cfg["training"]["mf_n_factors"],
                             seed=cfg["project"]["seed"]),
         {"type": "matrix_factorization_svd",
          "n_factors": cfg["training"]["mf_n_factors"]}),
    ]

    results = []
    for model, params in specs:
        LOG.info("Training model: %s", model.name)
        with mlflow.start_run(run_name=model.name):
            model.fit(train_df)
            metrics = evaluate_model(model, test_df, item_features, k, rel)

            mlflow.log_param("model_name", model.name)
            mlflow.log_param("k", k)
            mlflow.log_param("relevance_threshold", rel)
            for key, val in params.items():
                mlflow.log_param(key, val)
            for key, val in metrics.items():
                mlflow.log_metric(key.replace("@", "_at_"), float(val))

            model_path = models_dir / f"{model.name}.pkl"
            _save_model(model, model_path)
            mlflow.log_artifact(str(model_path))

            row = {"model": model.name, **metrics}
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(reports_dir / "evaluation_results.csv", index=False)
    _write_results_markdown(results_df, reports_dir / "evaluation_results.md", k)
    (reports_dir / "evaluation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    LOG.info("Training complete. Results:\n%s", results_df.to_string(index=False))
    return {"results": results_df}


def _write_results_markdown(df: pd.DataFrame, path: Path, k: int) -> None:
    lines = [
        "# Model Evaluation Results",
        "",
        f"Metrics computed on the held-out time-based test split (K = {k}).",
        "",
        "| " + " | ".join(df.columns) + " |",
        "|" + "|".join(["---"] * len(df.columns)) + "|",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    lines += [
        "",
        "**Reading the table**: higher Precision@K / Recall@K / NDCG@K / "
        "Coverage / Diversity is better; lower RMSE / MAE is better. The "
        "popularity model is the non-personalised baseline; the collaborative-"
        "filtering and matrix-factorisation models are the personalised "
        "recommenders.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    out = train()
    print(out["results"].to_string(index=False))
