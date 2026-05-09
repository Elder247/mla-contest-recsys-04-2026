"""Stage-1 cascade ranker: LightGBM lambdarank.

Used to prune the merged candidate pool to ``n_ranker_eval`` per uid (or
``n_ranker_train`` during fit) before the heavier CatBoost YetiRank
stage. Fixed hyperparams — tune ``n_ranker_eval`` in optuna instead of
LGBM internals (3x cheaper trial cost).

Hyperparameter rationale (defaults):

* ``lambdarank_truncation_level=200`` — LightGBM only computes pairwise
  λ-loss up to this position. Default (30) is far below our cascade
  cutoff (~1000+), so ranking quality at the cascade boundary is poorly
  optimised. Setting this to ~max(n_ranker_eval, 200) aligns the loss
  with the actual decision boundary for the membership cut.
* ``feature_fraction=0.8 / bagging_fraction=0.8 / bagging_freq=1`` —
  implicit ensembling via row+col subsampling. Standard de-facto for
  GBDT ranking; +0.3-0.7 recall typical with negligible cost.
* ``num_leaves=127``, ``learning_rate=0.03``, ``n_estimators=3000``,
  ``min_child_samples=100`` — slightly deeper / slower / more regularised
  than the defaults; ES@100 keeps wallclock similar.
* ``reg_alpha=0.1`` / ``reg_lambda=1.0`` — L1/L2 split regularisation.

Negative subsampling
--------------------
Pos rate on the cascade pool is ~0.5%; the bottom 99.5% of negatives
are interchangeable for pairwise λ-loss. When ``negative_ratio`` is
set (default 10), ``fit`` keeps every positive and a random sample of
``len(positives) * negative_ratio`` negatives → 5-10× faster cascade
fit + cache build with no measurable Recall@100 impact. Set to ``None``
to disable.

API mirrors :class:`src.models.catboost_ranker.RankerModel`:

    ``fit(df_train, df_val=None)``        — lambdarank fit with optional ES
    ``score(df, chunk_size=...)``        — chunked predict, returns
                                            ``(uid, item_id, lgbm_score)``
    ``top_k_per_user(df, k, score_col)``  — staticmethod cut
    ``predict(df, n=100)``                — composition

The score column is ``lgbm_score``; the cascade in ``train_ranker.py``
also derives a per-user dense ``lgbm_rank`` and surfaces both columns
to the downstream CatBoost ranker.

Out-of-fold (OOF) scoring
-------------------------
Scoring ``labeled_full`` with an LGBM trained on a subset that includes
the same rows leaks labels into ``lgbm_score`` (CatBoost then overfits).
Use :meth:`fit_oof` to produce leak-free ``lgbm_score`` on labeled rows,
then call :meth:`fit` once on an 80/20 split for the pickle used at
submit-time (eval pool scoring).
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

log = logging.getLogger(__name__)

_DEFAULT_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    num_leaves=127,
    learning_rate=0.03,
    n_estimators=3000,
    min_child_samples=100,
    reg_lambda=1.0,
    reg_alpha=0.1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambdarank_truncation_level=200,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)
_SKIP_COLS = {"uid", "item_id", "label"}
DEFAULT_SCORE_CHUNK = 500_000
DEFAULT_NEGATIVE_RATIO = None
# Early-stopping patience at the cascade boundary. 200 is a safe default
# given num_leaves=127 + lr=0.03 — typical best_iter on 500m sits in the
# 600-1200 range, so 200 patience prevents premature stops on noise.
DEFAULT_EARLY_STOPPING_ROUNDS = 200
# Evaluate NDCG at the actual cascade boundary (n_ranker_eval ≈ 1500)
# rather than the LightGBM default {1,2,3,4,5}. Tiny-K NDCG saturates
# within 1-2 trees on a pool with 9 strong CG-rank features → causes
# spurious early-stopping at iter=1. NDCG@100 matches the contest metric
# and gives a smooth, monotonic improvement signal. Passed as a kwarg
# to ``.fit()`` (NOT into params) — putting it in params triggers a
# UserWarning from lightgbm because ``eval_at`` is also a fit-time arg.
DEFAULT_EVAL_AT = (100,)


class LightGBMRanker:
    """Thin wrapper around :class:`lightgbm.LGBMRanker` for the cascade pipeline.

    Parameters:
        negative_ratio: keep ``len(positives) * ratio`` random negatives in
            the **train** pool during ``fit``; ``None`` disables subsampling.
            Validation is **never** subsampled — early-stopping needs a
            distribution that matches deployment (cascade pool of ~800
            candidates per uid). Subsampling val to 10:1 collapses NDCG@K
            to a near-constant after the first tree, which historically
            caused ``best_iter=1`` ES failures.
        subsample_seed: deterministic seed for the polars ``sample(seed=)``
            calls so different runs of the same fit produce the same
            subsampled pool.
        early_stopping_rounds: ES patience on the validation NDCG@100. Pass
            ``None`` to disable ES entirely (model trains for the full
            ``n_estimators``).
        **overrides: forwarded into the underlying ``lgb.LGBMRanker`` ctor.
    """

    def __init__(
        self,
        *,
        negative_ratio: int | float | None = DEFAULT_NEGATIVE_RATIO,
        subsample_seed: int = 42,
        early_stopping_rounds: int | None = DEFAULT_EARLY_STOPPING_ROUNDS,
        eval_at: list[int] | tuple[int, ...] = DEFAULT_EVAL_AT,
        **overrides,
    ) -> None:
        # ``eval_at`` is a *fit-time* lightgbm arg, not a model param —
        # passing it through ``params`` raises a UserWarning. Keep it as
        # a separate instance attr and forward only in ``.fit()``.
        if "eval_at" in overrides:
            eval_at = overrides.pop("eval_at")
        self._user_model_overrides = dict(overrides)
        self.params = {**_DEFAULT_PARAMS, **self._user_model_overrides}
        self.negative_ratio = negative_ratio
        self.subsample_seed = subsample_seed
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_at = list(eval_at)
        self._model: lgb.LGBMRanker | None = None
        self._feature_cols: list[str] = []

    def _clone_like(self, *, subsample_seed: int | None = None) -> LightGBMRanker:
        """Fresh ranker with identical ctor kwargs (for OOF folds)."""
        return LightGBMRanker(
            negative_ratio=self.negative_ratio,
            subsample_seed=(
                self.subsample_seed if subsample_seed is None else subsample_seed
            ),
            early_stopping_rounds=self.early_stopping_rounds,
            eval_at=tuple(self.eval_at),
            **self._user_model_overrides,
        )

    def fit_oof(
        self,
        df_labeled: pl.DataFrame,
        *,
        n_folds: int,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Out-of-fold ``lgbm_score`` for every labeled row (group-wise by ``uid``).

        For each fold, trains on all rows whose ``uid`` is in the train folds,
        early-stops on the holdout fold, then scores **only** the holdout rows.
        Concatenated result has one row per input row with columns
        ``uid``, ``item_id``, ``lgbm_score`` — safe to join to ``df_labeled``
        for cascade + CatBoost without label leakage into stage-1 scores.

        Does **not** mutate ``self._model``; call :meth:`fit` afterward for the
        production pickle used to score the eval/submit pool.

        Args:
            df_labeled: labeled candidate×user rows (must include ``uid``,
                ``item_id``, ``label``, features).
            n_folds: ``>= 2``. Uses :class:`sklearn.model_selection.GroupKFold`
                so all rows of a user stay in the same fold split.
            seed: forwarded to ``GroupKFold.shuffle`` when sklearn supports it;
                subsample seeds per fold use ``self.subsample_seed + fold * 10001``.
        """
        if n_folds < 2:
            raise ValueError("fit_oof: n_folds must be >= 2")
        df = df_labeled.sort("uid")
        uids = df["uid"].to_numpy()
        n = len(df)
        gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        parts: list[pl.DataFrame] = []
        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(np.zeros(n), groups=uids),
        ):
            df_tr = df[train_idx]
            df_va = df[val_idx]
            m = self._clone_like(subsample_seed=self.subsample_seed + fold * 10_001)
            log.info(
                "LightGBMRanker.fit_oof: fold %d/%d  train_rows=%d  val_rows=%d",
                fold + 1, n_folds, len(df_tr), len(df_va),
            )
            m.fit(df_tr, df_va)
            parts.append(m.score(df_va))
        out = pl.concat(parts)
        if len(out) != n:
            raise RuntimeError(
                f"fit_oof: expected {n} scored rows, got {len(out)}",
            )
        log.info("LightGBMRanker.fit_oof: done — %d rows OOF", len(out))
        return out

    def _maybe_subsample(self, df: pl.DataFrame, kind: str) -> pl.DataFrame:
        """Return ``df`` with negatives downsampled to ``negative_ratio:1``.

        No-op when ``negative_ratio`` is None or the pool already has
        fewer negatives than the ratio implies. Sort by uid afterwards so
        the LightGBM group definition stays contiguous.
        """
        if self.negative_ratio is None:
            return df
        n_pos = int((df["label"] == 1).sum())
        if n_pos == 0:
            log.warning("LightGBMRanker._maybe_subsample(%s): 0 positives, skip", kind)
            return df
        target_neg = int(n_pos * self.negative_ratio)
        pos = df.filter(pl.col("label") == 1)
        neg = df.filter(pl.col("label") != 1)
        n_neg = len(neg)
        if n_neg <= target_neg:
            return df
        neg_sampled = neg.sample(n=target_neg, seed=self.subsample_seed, shuffle=True)
        log.info(
            "LightGBMRanker._maybe_subsample(%s): %d -> %d rows "
            "(%d pos kept, %d/%d neg sampled, ratio %d:1)",
            kind, len(df), len(pos) + len(neg_sampled),
            n_pos, target_neg, n_neg, int(self.negative_ratio),
        )
        return pl.concat([pos, neg_sampled]).sort("uid")

    def fit(
        self,
        df_train: pl.DataFrame,
        df_val: pl.DataFrame | None = None,
    ) -> None:
        self._feature_cols = [c for c in df_train.columns if c not in _SKIP_COLS]
        df_train = self._maybe_subsample(df_train.sort("uid"), kind="train")
        groups_train = (
            df_train.group_by("uid", maintain_order=True)
            .agg(pl.len())["len"]
            .to_list()
        )
        X_train = df_train[self._feature_cols].to_pandas()
        y_train = df_train["label"].to_pandas()

        eval_kwargs: dict = {}
        if df_val is not None:
            # Critical: do NOT subsample val. ES on NDCG@100 needs the full
            # per-uid candidate pool to produce a meaningful learning curve.
            df_val = df_val.sort("uid")
            groups_val = (
                df_val.group_by("uid", maintain_order=True)
                .agg(pl.len())["len"]
                .to_list()
            )
            X_val = df_val[self._feature_cols].to_pandas()
            y_val = df_val["label"].to_pandas()
            callbacks = []
            if self.early_stopping_rounds is not None:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=int(self.early_stopping_rounds),
                        verbose=False,
                    )
                )
            # Light progress logging — every 50 trees write one line so the
            # user can track convergence in real-time without flooding logs.
            callbacks.append(lgb.log_evaluation(period=50))
            eval_kwargs = dict(
                eval_set=[(X_val, y_val)],
                eval_group=[groups_val],
                eval_at=self.eval_at,
                callbacks=callbacks,
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
