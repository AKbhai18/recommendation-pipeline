"""Recommendation model implementations.

Three models, all exposing the same interface so the evaluation stage can treat
them uniformly:

    model.fit(train_df)
    model.recommend(user_id, k)         -> list[item_id]  (top-k, excl. seen)
    model.predict_rating(user_id, item) -> float | None   (for RMSE / MAE)

Models
------
* PopularityRecommender  - non-personalised baseline. Recommends the globally
  most popular items (by interaction count), same list for everyone. Rating
  prediction falls back to the item's global mean rating.

* ItemBasedCF            - item-item collaborative filtering using cosine
  similarity between item rating vectors. A user's score for an unseen item is
  the similarity-weighted average of their ratings on similar items.

* MatrixFactorization    - truncated-SVD latent-factor model on the mean-
  centred user-item matrix. Reconstructs the full rating matrix from the
  low-rank factors.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class BaseRecommender:
    name = "base"

    def fit(self, train: pd.DataFrame):  # pragma: no cover - interface
        raise NotImplementedError

    def recommend(self, user_id: str, k: int) -> List[str]:  # pragma: no cover
        raise NotImplementedError

    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        return None


class PopularityRecommender(BaseRecommender):
    """Non-personalised popularity baseline."""
    name = "popularity_baseline"

    def fit(self, train: pd.DataFrame):
        self.global_mean_ = float(train["rating"].mean())
        pop = train.groupby("item_id").agg(
            count=("user_id", "count"), mean_rating=("rating", "mean")
        )
        self.ranked_items_ = pop.sort_values("count", ascending=False).index.tolist()
        self.item_mean_ = pop["mean_rating"].to_dict()
        self.user_seen_ = train.groupby("user_id")["item_id"].agg(set).to_dict()
        return self

    def recommend(self, user_id: str, k: int) -> List[str]:
        seen = self.user_seen_.get(user_id, set())
        recs = [i for i in self.ranked_items_ if i not in seen]
        return recs[:k]

    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        return float(self.item_mean_.get(item_id, self.global_mean_))


class ItemBasedCF(BaseRecommender):
    """Item-item collaborative filtering with cosine similarity."""
    name = "item_based_cf"

    def __init__(self, top_k_neighbours: int = 50):
        self.top_k_neighbours = top_k_neighbours

    def fit(self, train: pd.DataFrame):
        self.global_mean_ = float(train["rating"].mean())
        matrix = train.pivot_table(
            index="user_id", columns="item_id", values="rating", aggfunc="mean"
        )
        self.users_ = matrix.index.tolist()
        self.items_ = matrix.columns.tolist()
        self.item_pos_ = {it: j for j, it in enumerate(self.items_)}
        self.user_pos_ = {u: i for i, u in enumerate(self.users_)}

        filled = matrix.fillna(0.0).values
        self.R_ = filled                                  # users x items
        sim = cosine_similarity(filled.T)                 # items x items
        np.fill_diagonal(sim, 0.0)
        self.sim_ = sim
        self.item_mean_ = np.where(
            (filled != 0).sum(axis=0) > 0,
            filled.sum(axis=0) / np.maximum((filled != 0).sum(axis=0), 1),
            self.global_mean_,
        )
        self.user_seen_ = {
            u: set(matrix.columns[np.where(filled[i] != 0)[0]])
            for u, i in self.user_pos_.items()
        }
        return self

    def _scores_for_user(self, user_id: str) -> Optional[np.ndarray]:
        i = self.user_pos_.get(user_id)
        if i is None:
            return None
        user_vec = self.R_[i]                             # ratings over items
        # weighted sum of similarities to the items the user rated
        scores = self.sim_ @ user_vec
        norm = np.abs(self.sim_).sum(axis=1)
        norm[norm == 0] = 1.0
        return scores / norm

    def recommend(self, user_id: str, k: int) -> List[str]:
        scores = self._scores_for_user(user_id)
        if scores is None:
            return []
        seen = self.user_seen_.get(user_id, set())
        order = np.argsort(scores)[::-1]
        recs = [self.items_[j] for j in order if self.items_[j] not in seen]
        return recs[:k]

    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        scores = self._scores_for_user(user_id)
        j = self.item_pos_.get(item_id)
        if scores is None or j is None:
            return float(self.global_mean_)
        # map the neighbourhood score back onto the rating scale via item mean
        base = self.item_mean_[j]
        val = base + scores[j]
        return float(np.clip(val, 1.0, 5.0))


class MatrixFactorization(BaseRecommender):
    """Truncated-SVD latent-factor model on the mean-centred matrix."""
    name = "matrix_factorization_svd"

    def __init__(self, n_factors: int = 50, seed: int = 42):
        self.n_factors = n_factors
        self.seed = seed

    def fit(self, train: pd.DataFrame):
        self.global_mean_ = float(train["rating"].mean())
        matrix = train.pivot_table(
            index="user_id", columns="item_id", values="rating", aggfunc="mean"
        )
        self.users_ = matrix.index.tolist()
        self.items_ = matrix.columns.tolist()
        self.item_pos_ = {it: j for j, it in enumerate(self.items_)}
        self.user_pos_ = {u: i for i, u in enumerate(self.users_)}

        mask = matrix.notna().values
        self.user_mean_ = np.where(
            mask.any(axis=1),
            np.nanmean(matrix.values, axis=1),
            self.global_mean_,
        )
        centred = np.where(mask, matrix.values - self.user_mean_[:, None], 0.0)

        n_factors = min(self.n_factors, min(centred.shape) - 1)
        svd = TruncatedSVD(n_components=n_factors, random_state=self.seed)
        U = svd.fit_transform(centred)                    # users x f
        self.recon_ = U @ svd.components_ + self.user_mean_[:, None]
        self.user_seen_ = {
            u: set(matrix.columns[np.where(mask[i])[0]])
            for u, i in self.user_pos_.items()
        }
        return self

    def recommend(self, user_id: str, k: int) -> List[str]:
        i = self.user_pos_.get(user_id)
        if i is None:
            return []
        scores = self.recon_[i].copy()
        seen = self.user_seen_.get(user_id, set())
        order = np.argsort(scores)[::-1]
        recs = [self.items_[j] for j in order if self.items_[j] not in seen]
        return recs[:k]

    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        i = self.user_pos_.get(user_id)
        j = self.item_pos_.get(item_id)
        if i is None or j is None:
            return float(self.global_mean_)
        return float(np.clip(self.recon_[i, j], 1.0, 5.0))
