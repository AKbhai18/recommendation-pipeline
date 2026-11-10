"""Stage 7 - Model Evaluation.

Ranking and rating metrics computed on the held-out (time-based) test split:

  * Precision@K / Recall@K - fraction of the top-K recommendations that are
    relevant, and fraction of the relevant items that were retrieved.
  * NDCG@K                 - rank-aware quality (relevant hits near the top
    count more).
  * Coverage               - fraction of the item catalog that appears in at
    least one user's top-K list (catalog coverage).
  * Diversity              - 1 - average intra-list cosine similarity of
    recommended items (based on item category one-hot vectors); higher means
    more varied recommendations.
  * RMSE / MAE             - rating-prediction error on the test interactions.

An item in a user's test set is "relevant" when its rating >= the configured
relevance threshold.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.common import get_logger

LOG = get_logger("evaluation.metrics", log_file="reports/pipeline.log")


def _dcg(relevances: List[int]) -> float:
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))


def _ndcg_at_k(recommended: List[str], relevant: set, k: int) -> float:
    rec_k = recommended[:k]
    gains = [1 if it in relevant else 0 for it in rec_k]
    dcg = _dcg(gains)
    ideal = _dcg(sorted(gains, reverse=True)) if any(gains) else 0.0
    # ideal DCG uses as many 1s as there are relevant items (capped at k)
    n_rel = min(len(relevant), k)
    idcg = _dcg([1] * n_rel) if n_rel > 0 else 0.0
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(model, test: pd.DataFrame, item_features: pd.DataFrame,
                     k: int, relevance_threshold: float) -> Dict[str, float]:
    """Compute Precision@K, Recall@K, NDCG@K, Coverage and Diversity."""
    # relevant items per user in the test set
    rel = test[test["rating"] >= relevance_threshold]
    relevant_by_user = rel.groupby("user_id")["item_id"].agg(set).to_dict()
    eval_users = [u for u, items in relevant_by_user.items() if len(items) > 0]

    # category vectors for diversity
    cat = item_features.set_index("item_id")["item_category"]
    cat_dummies = pd.get_dummies(cat)
    cat_vectors = {it: cat_dummies.loc[it].values.astype(float)
                   for it in cat_dummies.index}

    precisions, recalls, ndcgs, diversities = [], [], [], []
    recommended_catalog = set()
    all_items = set(item_features["item_id"])

    for u in eval_users:
        recs = model.recommend(u, k)
        if not recs:
            precisions.append(0.0); recalls.append(0.0)
            ndcgs.append(0.0)
            continue
        relevant = relevant_by_user[u]
        hits = sum(1 for it in recs if it in relevant)
        precisions.append(hits / len(recs))
        recalls.append(hits / len(relevant))
        ndcgs.append(_ndcg_at_k(recs, relevant, k))
        recommended_catalog.update(recs)

        # intra-list diversity = 1 - mean pairwise cosine similarity
        vecs = [cat_vectors[i] for i in recs if i in cat_vectors]
        if len(vecs) > 1:
            sims = []
            for a in range(len(vecs)):
                for b in range(a + 1, len(vecs)):
                    va, vb = vecs[a], vecs[b]
                    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
                    sims.append(float(np.dot(va, vb) / denom))
            diversities.append(1.0 - (np.mean(sims) if sims else 0.0))

    return {
        f"precision@{k}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f"recall@{k}": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        f"ndcg@{k}": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        "coverage": round(len(recommended_catalog) / len(all_items), 4),
        "diversity": round(float(np.mean(diversities)) if diversities else 0.0, 4),
        "eval_users": len(eval_users),
    }


def evaluate_rating(model, test: pd.DataFrame) -> Dict[str, float]:
    """Compute RMSE and MAE on the test interactions (explicit ratings)."""
    preds, actuals = [], []
    for row in test.itertuples(index=False):
        p = model.predict_rating(row.user_id, row.item_id)
        if p is not None:
            preds.append(p)
            actuals.append(row.rating)
    if not preds:
        return {"rmse": float("nan"), "mae": float("nan")}
    preds = np.array(preds); actuals = np.array(actuals)
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4)}


def evaluate_model(model, test, item_features, k, relevance_threshold) -> Dict[str, float]:
    ranking = evaluate_ranking(model, test, item_features, k, relevance_threshold)
    rating = evaluate_rating(model, test)
    metrics = {**ranking, **rating}
    LOG.info("[%s] %s", getattr(model, "name", "model"), metrics)
    return metrics
