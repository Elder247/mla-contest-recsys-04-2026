"""ALS-based collaborative filtering recommender (via implicit library)."""
import logging

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares

from src.data.dataset import positive_listens
from src.data.preprocessing import build_id_maps, build_csr_matrix
from src.models.base import BaseModel

log = logging.getLogger(__name__)


class ALSModel(BaseModel):
    """ALS collaborative filtering.

    recommend() returns (uid, item_id, score, als_rank) — the extra column
    is used as a ranker feature downstream.
    """

    def __init__(
        self,
        name: str = "als",
        factors: int = 128,
        iterations: int = 20,
        regularization: float = 0.01,
        alpha: float = 40.0,
        n_cand: int = 500,
        random_state: int = 42,
    ):
        self.name = name
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.n_cand = n_cand
        self.random_state = random_state

        self._model: AlternatingLeastSquares | None = None
        self._matrix = None
        self._uid_map: dict = {}
        self._item_map: dict = {}
        self._inv_item_map: dict = {}

    def fit(self, train: pl.DataFrame, **kwargs) -> None:
        pos = positive_listens(train)
        uid_map, item_map, _, inv_item_map = build_id_maps(pos)
        matrix = build_csr_matrix(pos, uid_map, item_map)

        log.info(
            "fitting ALS: users=%d items=%d factors=%d iters=%d alpha=%.1f",
            len(uid_map), len(item_map), self.factors, self.iterations, self.alpha,
        )
        model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            alpha=self.alpha,
            random_state=self.random_state,
            num_threads=0,
        )
        model.fit(matrix)

        self._model = model
        self._matrix = matrix
        self._uid_map = uid_map
        self._item_map = item_map
        self._inv_item_map = inv_item_map
        log.info("ALS fitted")

    def recommend(self, users: list[int], n: int = 100, **kwargs) -> pl.DataFrame:
        known = [u for u in users if u in self._uid_map]
        if not known:
            return pl.DataFrame(schema={"uid": pl.Int64, "item_id": pl.Int64, "score": pl.Float32, "als_rank": pl.Int32})

        idxs = [self._uid_map[u] for u in known]
        recs, scores = self._model.recommend(
            userid=idxs,
            user_items=self._matrix[idxs],
            N=n,
            filter_already_liked_items=True,
        )

        uids = np.repeat(known, n)
        item_ids = np.array(
            [self._inv_item_map[j] for row in recs for j in row],
            dtype=np.int64,
        )
        flat_scores = scores.flatten().astype(np.float32)
        ranks = np.tile(np.arange(1, n + 1, dtype=np.int32), len(known))

        return pl.DataFrame({
            "uid": uids,
            "item_id": item_ids,
            "score": flat_scores,
            "als_rank": ranks,
        }).with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
