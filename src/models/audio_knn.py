"""Audio-embedding KNN candidate generator (FAISS HNSW32).

Cold-track-friendly CG that uses pre-computed audio embeddings (CNN over
spectrograms — see Yambda paper §4 / data-dictionary.md) to find tracks
acoustically similar to a user's recent listens. Targets the 15.9% cold
cohort (per `notebooks/01_eda.ipynb`) that pure CF cannot reach.

User vector = L2-normalised mean of the last ``user_history_k`` positive
listens that have embedding coverage. We use ``normalized_embed`` from
embeddings.parquet directly, so cosine similarity collapses to
``METRIC_INNER_PRODUCT`` over the FAISS HNSW graph.

Index: HNSW32 (M=32) with configurable ``ef_construction`` / ``ef_search``.
On 50m (~630K items, dim=128) construction takes ~10 s; search is <1 ms
per query. The same index scales to millions of items on 500m / 5B; for 5B
we may need to swap to IVFPQ to control memory — see roadmap Phase C.2.
"""
import logging

import faiss
import numpy as np
import polars as pl

from src.data.dataset import positive_listens
from src.models.base import BaseModel

log = logging.getLogger(__name__)


class AudioEmbedKNNModel(BaseModel):
    """Per-user mean-embedding KNN over normalised audio embeddings.

    Output columns: ``uid, item_id, score, audio_knn_rank``.

    Notes:
        - Uses ``normalized_embed`` from embeddings.parquet directly; cosine
          similarity == inner product.
        - Users with 0 covered listens are skipped (no recommendations) —
          DecayPop / ALS cover them.
        - Items without embeddings are simply absent from this CG's output
          but still reachable via the other 6 CGs.
        - The FAISS index pickles cleanly via NumPy serialisation: we save
          the raw item matrix + ids and rebuild the index on load. Avoids
          dragging FAISS pickle compatibility into the cg cache.
    """

    def __init__(
        self,
        name: str = "audio_knn",
        n_cand: int = 100,
        user_history_k: int = 20,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        embeddings_path: str | None = None,
    ):
        self.name = name
        self.n_cand = n_cand
        self.user_history_k = user_history_k
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.embeddings_path = embeddings_path

        self._item_matrix: np.ndarray | None = None  # (n_items, dim) float32, L2-norm
        self._item_ids: np.ndarray | None = None     # (n_items,) int64
        self._user_vec: dict[int, np.ndarray] = {}   # uid -> (dim,) float32
        self._dim: int = 0
        self._index: faiss.Index | None = None       # rebuilt on load (see __setstate__)

    # ── pickle helpers — drop the FAISS index, rebuild on unpickle ──────────
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_index"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._item_matrix is not None:
            self._build_index()

    def _build_index(self) -> None:
        index = faiss.IndexHNSWFlat(self._dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        index.add(self._item_matrix)
        self._index = index
        log.info(
            "AudioEmbedKNN: HNSW32 index built (M=%d efC=%d efS=%d items=%d dim=%d)",
            self.hnsw_m, self.ef_construction, self.ef_search,
            self._item_matrix.shape[0], self._dim,
        )

    # ── fit / recommend ──────────────────────────────────────────────────────
    def fit(self, train: pl.DataFrame, **kwargs) -> None:
        pos = positive_listens(train).select(["uid", "item_id", "timestamp"])
        log.info(
            "AudioEmbedKNN: positive listens=%d users=%d items=%d",
            len(pos), pos["uid"].n_unique(), pos["item_id"].n_unique(),
        )

        train_items = pos["item_id"].unique().to_list()
        emb_path = self.embeddings_path or "data/embeddings.parquet"
        log.info(
            "loading embeddings (filter to %d train items) from %s",
            len(train_items), emb_path,
        )

        emb_df = (
            pl.scan_parquet(emb_path)
            .filter(pl.col("item_id").is_in(train_items))
            .select(["item_id", "normalized_embed"])
            .collect()
        )
        coverage = len(emb_df) / max(len(train_items), 1)
        log.info(
            "AudioEmbedKNN: %d / %d items have embeddings (coverage=%.1f%%)",
            len(emb_df), len(train_items), coverage * 100,
        )

        emb_arr = np.ascontiguousarray(
            np.asarray(emb_df["normalized_embed"].to_list(), dtype=np.float32)
        )
        ids_arr = emb_df["item_id"].cast(pl.Int64).to_numpy()
        self._dim = int(emb_arr.shape[1])
        self._item_matrix = emb_arr
        self._item_ids = ids_arr
        log.info("AudioEmbedKNN: item matrix shape=%s", emb_arr.shape)

        self._build_index()

        # ── per-user mean embeddings from last K covered listens ─────────────
        item_id_to_idx: dict[int, int] = {int(i): k for k, i in enumerate(ids_arr.tolist())}
        last_k_pd = (
            pos
            .sort(["uid", "timestamp"], descending=[False, True])
            .group_by("uid", maintain_order=True)
            .head(self.user_history_k)
            .select(["uid", "item_id"])
            .to_pandas()
        )
        last_k_pd["idx"] = last_k_pd["item_id"].map(item_id_to_idx)
        last_k_pd = last_k_pd.dropna(subset=["idx"])
        last_k_pd["idx"] = last_k_pd["idx"].astype(np.int64)

        unique_uids, inv = np.unique(last_k_pd["uid"].to_numpy(), return_inverse=True)
        sums = np.zeros((len(unique_uids), self._dim), dtype=np.float32)
        counts = np.zeros(len(unique_uids), dtype=np.int32)
        np.add.at(sums, inv, emb_arr[last_k_pd["idx"].to_numpy()])
        np.add.at(counts, inv, 1)
        means = sums / counts[:, None].astype(np.float32)
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        means = (means / norms).astype(np.float32)

        self._user_vec = {int(u): v for u, v in zip(unique_uids.tolist(), means)}
        log.info(
            "AudioEmbedKNN fitted: %d user vectors (out of %d positive-listening users)",
            len(self._user_vec), pos["uid"].n_unique(),
        )

    def recommend(self, users: list[int], n: int = 100, **kwargs) -> pl.DataFrame:
        if self._index is None:
            raise RuntimeError("AudioEmbedKNN not fitted (or index not rebuilt)")
        rank_col = f"{self.name}_rank"

        u_with_vec = [u for u in users if u in self._user_vec]
        log.info(
            "AudioEmbedKNN: recommending for %d / %d users (others lack embed history)",
            len(u_with_vec), len(users),
        )
        if not u_with_vec:
            return pl.DataFrame(schema={
                "uid": pl.Int64, "item_id": pl.Int64,
                "score": pl.Float32, rank_col: pl.Int32,
            })

        # FAISS expects (n, dim) float32 contiguous queries.
        Q = np.ascontiguousarray(
            np.stack([self._user_vec[u] for u in u_with_vec], axis=0).astype(np.float32)
        )
        top_n = min(n, self._item_matrix.shape[0])
        scores, idx = self._index.search(Q, top_n)  # (B, n) each

        B = len(u_with_vec)
        uid_flat = np.repeat(np.asarray(u_with_vec, dtype=np.int64), top_n)
        item_flat = self._item_ids[idx.ravel()]
        score_flat = scores.ravel().astype(np.float32)
        rank_flat = np.tile(np.arange(1, top_n + 1, dtype=np.int32), B)

        return pl.DataFrame({
            "uid": uid_flat,
            "item_id": item_flat,
            "score": score_flat,
            rank_col: rank_flat,
        })
