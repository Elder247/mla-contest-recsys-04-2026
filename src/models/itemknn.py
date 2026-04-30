"""Item-item KNN candidate generator (cosine similarity over user-item matrix).

Per Yambda paper this is the strongest single CF model on 50M Listen+
(Recall@100 ≈ 0.13 plain, ≈ 200 in our x 1000 scale). Backed by
``implicit.nearest_neighbours.CosineRecommender`` which keeps a sparse
item-similarity matrix (items x K).
"""
import logging

import numpy as np
import polars as pl
from implicit.nearest_neighbours import CosineRecommender

from src.data.dataset import positive_listens
from src.data.preprocessing import build_csr_matrix, build_id_maps
from src.models.base import BaseModel

log = logging.getLogger(__name__)


class ItemKNNModel(BaseModel):
    """Cosine item-KNN over the binary positive-listen matrix.

    Output columns: ``uid, item_id, score, itemknn_rank``.
    """

    def __init__(
        self,
        name: str = "itemknn",
        k: int = 200,
        n_cand: int = 200,
    ):
        self.name = name
        self.k = k
        self.n_cand = n_cand
        self._model: CosineRecommender | None = None
        self._matrix = None
        self._uid_map: dict = {}
        self._item_map: dict = {}
        self._inv_item_map: dict = {}

    def fit(self, train: pl.DataFrame, **kwargs) -> None:
        pos = positive_listens(train)
        uid_map, item_map, _, inv_item_map = build_id_maps(pos)
        matrix = build_csr_matrix(pos, uid_map, item_map)

        log.info(
            "fitting ItemKNN: users=%d items=%d K=%d",
            len(uid_map), len(item_map), self.k,
        )
        model = CosineRecommender(K=self.k)
        model.fit(matrix)

        self._model = model
        self._matrix = matrix
        self._uid_map = uid_map
        self._item_map = item_map
        self._inv_item_map = inv_item_map
        log.info("ItemKNN fitted")

    def recommend(self, users: list[int], n: int = 100, **kwargs) -> pl.DataFrame:
        rank_col = f"{self.name}_rank"
        empty = pl.DataFrame(
            schema={
                "uid": pl.Int64,
                "item_id": pl.Int64,
                "score": pl.Float32,
                rank_col: pl.Int32,
            }
        )

        known = [u for u in users if u in self._uid_map]
        if not known:
            return empty

        idxs = [self._uid_map[u] for u in known]
        recs, scores = self._model.recommend(
            userid=idxs,
            user_items=self._matrix[idxs],
            N=n,
            filter_already_liked_items=False,
        )

        # implicit pads short rows with item_id=-1, score=0. Drop padding.
        n_users = len(known)
        uid_grid = np.repeat(np.asarray(known, dtype=np.int64), n)
        rank_grid = np.tile(np.arange(1, n + 1, dtype=np.int32), n_users)
        item_idx_flat = recs.reshape(-1)
        score_flat = scores.reshape(-1).astype(np.float32)

        valid = item_idx_flat >= 0
        if not valid.any():
            return empty

        uid_grid = uid_grid[valid]
        rank_grid = rank_grid[valid]
        item_idx_flat = item_idx_flat[valid]
        score_flat = score_flat[valid]

        # Map internal indices back to original item ids via vectorised lookup.
        max_idx = max(self._inv_item_map.keys()) + 1
        idx_to_item = np.empty(max_idx, dtype=np.int64)
        for k, v in self._inv_item_map.items():
            idx_to_item[k] = v
        item_ids = idx_to_item[item_idx_flat]

        return pl.DataFrame({
            "uid": uid_grid,
            "item_id": item_ids,
            "score": score_flat,
            rank_col: rank_grid,
        }).with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
