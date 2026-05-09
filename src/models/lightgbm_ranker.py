"""Stage-1 cascade ranker: LightGBM lambdarank.

Used to prune the merged candidate pool to ``n_ranker`` per uid before
the heavier CatBoost YetiRank stage. Fixed hyperparams — tune n_ranker
in optuna instead of LGBM internals (3x cheaper trial cost).

API mirrors :class:`src.models.catboost_ranker.RankerModel`:

    ``fit(df_train, df_val=None)``        — lambdarank fit with optional ES
    ``score(df, chunk_size=...)``        — chunked predict, returns
                                            ``(uid, item_id, lgbm_score)``
    ``top_k_per_user(df, k, score_col)``  — staticmethod cut
    ``predict(df, n=100)``                — composition

The score column is ``lgbm_score``; the cascade in ``train_ranker.py``
also derives a per-user dense ``lgbm_rank`` and surfaces both columns
to the downstream CatBoost ranker.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import polars as pl

log = logging.getLogger(__name__)

_DEFAULT_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    num_leaves=63,
    learning_rate=0.05,
    n_estimators=1500,
    min_child_samples=50,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)
_SKIP_COLS = {"uid", "item_id", "label"}
DEFAULT_SCORE_CHUNK = 500_000


class LightGBMRanker:
    """Thin wrapper around :class:`lightgbm.LGBMRanker` for the cascade pipeline."""

    def __init__(self, **overrides) -> None:
        self.params = {**_DEFAULT_PARAMS, **overrides}
        self._model: lgb.LGBMRanker | None = None
        self._feature_cols: list[str] = []

    def fit(
        self,
        df_train: pl.DataFrame,
        df_val: pl.DataFrame | None = None,
    ) -> None:
        self._feature_cols = [c for c in df_train.columns if c not in _SKIP_COLS]
        df_train = df_train.sort("uid")
        groups_train = (
            df_train.group_by("uid", maintain_order=True)
            .agg(pl.len())["len"]
            .to_list()
        )
        X_train = df_train[self._feature_cols].to_pandas()
        y_train = df_train["label"].to_pandas()

        eval_kwargs: dict = {}
        if df_val is not None:
            df_val = df_val.sort("uid")
            groups_val = (
                df_val.group_by("uid", maintain_order=True)
                .agg(pl.len())["len"]
                .to_list()
            )
            X_val = df_val[self._feature_cols].to_pandas()
            y_val = df_val["label"].to_pandas()
            eval_kwargs = dict(
                eval_set=[(X_val, y_val)],
                eval_group=[groups_val],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
            )

        self._model = lgb.LGBMRanker(**self.params)
        log.info(
            "LightGBMRanker.fit: %d feats, train=%d rows",
            len(self._feature_cols), len(df_train),
        )
        self._model.fit(X_train, y_train, group=groups_train, **eval_kwargs)
        best_iter = self._model.best_iteration_ or self.params["n_estimators"]
        log.info("LightGBMRanker fitted, best_iter=%s", best_iter)

    def score(
        self,
        df: pl.DataFrame,
        chunk_size: int = DEFAULT_SCORE_CHUNK,
    ) -> pl.DataFrame:
        if self._model is None:
            raise RuntimeError("LightGBMRanker.score: not fitted")
        n = len(df)
        if n == 0:
            return (
                df.select(["uid", "item_id"])
                .with_columns(pl.lit(0.0, dtype=pl.Float32).alias("lgbm_score"))
            )
        feats = self._feature_cols
        if n <= chunk_size:
            scores = self._model.predict(df[feats].to_pandas())
        else:
            parts = []
            for s in range(0, n, chunk_size):
                parts.append(
                    self._model.predict(df.slice(s, chunk_size)[feats].to_pandas())
                )
            scores = np.concatenate(parts)
        return (
            df.select(["uid", "item_id"])
            .with_columns(pl.Series("lgbm_score", scores).cast(pl.Float32))
        )

    @staticmethod
    def top_k_per_user(
        df: pl.DataFrame,
        k: int,
        score_col: str = "lgbm_score",
    ) -> pl.DataFrame:
        return (
            df.sort(["uid", score_col], descending=[False, True])
            .group_by("uid", maintain_order=True)
            .head(k)
        )

    def predict(
        self,
        df: pl.DataFrame,
        n: int = 100,
        chunk_size: int = DEFAULT_SCORE_CHUNK,
    ) -> pl.DataFrame:
        return self.top_k_per_user(self.score(df, chunk_size), k=n)
