"""CatBoost-based reranker: takes labeled candidates DataFrame, returns top-n per user."""
import logging

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

log = logging.getLogger(__name__)

_SKIP_COLS = {"uid", "item_id", "label"}

# Default chunk size for ``score`` — keeps the pandas materialisation under
# ~1-2 GB on 70-feature dataframes (500K × 70 × 8 bytes ≈ 280 MB) and lets
# the same code path scale to 5B without OOM.
DEFAULT_SCORE_CHUNK = 500_000


class RankerModel:
    """Thin wrapper around CatBoostRanker.

    Expects a DataFrame with columns: uid, item_id, label (0/1), and feature columns.

    The inference path is split into two stages:

    - ``score(df)`` runs the underlying CatBoost predict in row chunks and
      returns a slim ``(uid, item_id, ranker_score)`` table preserving input
      order. Use this when downstream code wants raw scores (e.g. Optuna
      n_cand allocation needs to retry many top-k cuts on the same scored
      table without re-running predict).
    - ``top_k_per_user(df, k)`` is a stateless staticmethod that takes any
      scored DataFrame and emits the per-user top-k slice.
    - ``predict(df, n)`` is the convenience that composes the two — kept
      with the original signature so train_ranker.py / submit_ranker.py
      don't change.
    """

    def __init__(
        self,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        task_type: str = "CPU",
        devices: str | None = None,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.task_type = task_type
        self.devices = devices
        self._model: CatBoostRanker | None = None
        self._feature_cols: list[str] = []

    def fit(self, df_train: pl.DataFrame, df_val: pl.DataFrame | None = None) -> None:
        self._feature_cols = [c for c in df_train.columns if c not in _SKIP_COLS]
        log.info("fitting RankerModel: %d features, train=%d rows", len(self._feature_cols), len(df_train))

        # GPU YetiRank hard limit: max 1023 candidates per query group.
        # Cap to 1023 keeping all positives (sort label desc before head).
        if self.task_type.upper() == "GPU":
            df_train = (
                df_train
                .sort(["uid", "label"], descending=[False, True])
                .group_by("uid", maintain_order=True)
                .head(1023)
            )
            if df_val is not None:
                df_val = (
                    df_val
                    .sort(["uid", "label"], descending=[False, True])
                    .group_by("uid", maintain_order=True)
                    .head(1023)
                )
            log.info(
                "GPU mode: capped train to %d rows, val to %d rows (max 1023/user)",
                len(df_train), len(df_val) if df_val is not None else 0,
            )

        # CatBoostRanker requires query_ids to be contiguous per group.
        df_train = df_train.sort("uid")
        train_pool = Pool(
            data=df_train[self._feature_cols].to_pandas(),
            label=df_train["label"].to_pandas(),
            group_id=df_train["uid"].to_pandas(),
        )
        # Pool has internalised the data; drop the polars frame so the GC can
        # release ~3 GB on 500m before fit() builds CatBoost's own arena.
        del df_train
        params = dict(
            loss_function="YetiRank",
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state,
            verbose=100,
            nan_mode="Min",
            task_type=self.task_type,
        )
        if self.task_type.upper() == "GPU":
            if self.devices is not None:
                params["devices"] = str(self.devices)
        else:
            params["thread_count"] = -1
        if df_val is not None:
            df_val = df_val.sort("uid")
            val_pool = Pool(
                data=df_val[self._feature_cols].to_pandas(),
                label=df_val["label"].to_pandas(),
                group_id=df_val["uid"].to_pandas(),
            )
            del df_val
            params["early_stopping_rounds"] = self.early_stopping_rounds
            self._model = CatBoostRanker(**params)
            self._model.fit(train_pool, eval_set=val_pool)
        else:
            self._model = CatBoostRanker(**params)
            self._model.fit(train_pool)

        log.info("RankerModel fitted, best_iteration=%s", self._model.get_best_iteration())

    def score(
        self,
        df: pl.DataFrame,
        chunk_size: int = DEFAULT_SCORE_CHUNK,
    ) -> pl.DataFrame:
        """Predict ``ranker_score`` for every row, chunked to bound memory.

        Returns ``(uid, item_id, ranker_score)`` with the same row order as
        the input. Each chunk materialises only its own feature slice into
        pandas — total peak memory ≈ chunk_size × n_features × 8 bytes
        regardless of input size.
        """
        if self._model is None:
            raise RuntimeError("RankerModel.score: model is not fitted yet")

        n_rows = len(df)
        if n_rows == 0:
            return (
                df.select(["uid", "item_id"])
                .with_columns(pl.lit(0.0, dtype=pl.Float32).alias("ranker_score"))
            )

        feat_cols = self._feature_cols
        n_chunks = (n_rows + chunk_size - 1) // chunk_size
        log.info(
            "RankerModel.score: %d rows in %d chunk(s) of <=%d",
            n_rows, n_chunks, chunk_size,
        )

        if n_chunks == 1:
            scores = self._model.predict(df[feat_cols].to_pandas())
        else:
            score_parts = []
            for start in range(0, n_rows, chunk_size):
                chunk_pdf = df.slice(start, chunk_size)[feat_cols].to_pandas()
                score_parts.append(self._model.predict(chunk_pdf))
            scores = np.concatenate(score_parts)

        return (
            df.select(["uid", "item_id"])
            .with_columns(pl.Series("ranker_score", scores))
        )

    @staticmethod
    def top_k_per_user(
        df: pl.DataFrame,
        k: int = 100,
        score_col: str = "ranker_score",
    ) -> pl.DataFrame:
        """Per-user top-k by ``score_col``. Output: ``(uid, item_id, score_col)``."""
        return (
            df.sort(["uid", score_col], descending=[False, True])
            .group_by("uid")
            .head(k)
            .select(["uid", "item_id", score_col])
        )

    def predict(
        self,
        df: pl.DataFrame,
        n: int = 100,
        chunk_size: int = DEFAULT_SCORE_CHUNK,
    ) -> pl.DataFrame:
        """Score every row then keep per-user top-n. Same contract as before."""
        scored = self.score(df, chunk_size=chunk_size)
        return self.top_k_per_user(scored, k=n)

    def feature_importance(self, prettified: bool = True):
        """CatBoost native PredictionValuesChange feature importance.

        For ``YetiRank`` (and other ranking losses) CatBoost defaults to
        ``LossFunctionChange``, which needs a Pool to compute. We force
        ``PredictionValuesChange`` so the call works without refeeding data.

        Returns a pandas DataFrame sorted descending by importance when
        ``prettified=True``; otherwise an array aligned with feature order.
        """
        if self._model is None:
            raise RuntimeError("RankerModel.feature_importance: model is not fitted yet")
        return self._model.get_feature_importance(
            type="PredictionValuesChange",
            prettified=prettified,
        )
