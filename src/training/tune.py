"""Optuna-driven hyperparameter search infrastructure (Phase D).

Three independent surfaces, each returning a fitted ``optuna.Study``:

    tune_candidate_generator(...)  — Phase D.1, standalone CG by Recall@N_max
    tune_ranker(...)               — Phase D.2, CatBoost params, Recall@100 after rerank
    tune_n_cand(...)               — Phase D.3, per-CG n_cand allocation under a
                                      total budget, Recall@100 after rerank
                                      (zero re-fit / re-score per trial)

The metric for standalone CG tuning is **Recall@N_max** — coverage of GT,
not internal ordering — because the ranker reranks afterwards. For ranker
and n_cand tuning the metric is the contest's Recall@100 (×1000 scale).

Default search-space helpers (``default_*_space``) match the ranges in
``docs/roadmap.md`` §D.2; pass your own callable to override.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import optuna
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from src.evaluation.metrics import recall_at_k
from src.models.base import BaseModel
from src.models.catboost_ranker import RankerModel

log = logging.getLogger(__name__)

# Optuna's INFO logging is one line per trial — too chatty inside our pipeline.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Storage helper
# ---------------------------------------------------------------------------

def make_storage(path: str | Path) -> str:
    """Build a SQLite storage URL after ensuring its parent dir exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{p.resolve()}"


# ---------------------------------------------------------------------------
# B1 — Candidate generator tuning
# ---------------------------------------------------------------------------

def tune_candidate_generator(
    model_factory: Callable[[optuna.Trial], BaseModel],
    train: pl.DataFrame,
    eval_users: list[int],
    gt_val: pl.DataFrame,
    n_max: int,
    n_trials: int,
    study_name: str | None = None,
    storage: str | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    seed: int = 42,
    show_progress_bar: bool = False,
) -> optuna.Study:
    """Tune a single CG by **Recall@N_max** standalone (no ranker).

    Args:
        model_factory: callable ``(trial) -> BaseModel``. Sample params via
            ``trial.suggest_*`` inside the factory and instantiate the CG
            with both the sampled params and any fixed kwargs (name,
            n_cand=n_max, etc.).
        train: DataFrame to fit on.
        eval_users: subset of users to recommend for (typically those
            present in ``gt_val``).
        gt_val: ground-truth pairs ``(uid, item_id)``.
        n_max: top-N to recommend per user; metric is Recall@n_max.
    """
    def objective(trial: optuna.Trial) -> float:
        model = model_factory(trial)
        model.fit(train)
        recs = model.recommend(eval_users, n=n_max)
        recs_for_metric = (
            recs
            .select(["uid", "item_id", "score"])
            .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
            .unique(["uid", "item_id"])
        )
        return recall_at_k(gt_val, recs_for_metric, k=n_max)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
        sampler=sampler or optuna.samplers.TPESampler(seed=seed),
        direction="maximize",
    )
    log.info(
        "tune_candidate_generator: %d trials, n_max=%d, storage=%s",
        n_trials, n_max, storage,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study


# ---------------------------------------------------------------------------
# B2 — Ranker tuning
# ---------------------------------------------------------------------------

def tune_ranker(
    labeled_df: pl.DataFrame,
    eval_features_df: pl.DataFrame,
    gt_val: pl.DataFrame,
    n_trials: int,
    k: int = 100,
    space: Callable[[optuna.Trial], dict] | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    seed: int = 42,
    show_progress_bar: bool = False,
) -> optuna.Study:
    """Tune CatBoost YetiRank hyperparams on fixed candidates+features.

    Args:
        labeled_df: DataFrame with ``uid, item_id, label`` and feature
            columns — output of ``train_ranker.py`` step 5 (cached parquet
            when ``cache_features=true`` in the ranker config).
        eval_features_df: features for the full eval users (no labels)
            — output of ``add_features`` on the merged candidate pool for
            10k eval users.
        gt_val: ground truth used to compute Recall@k after ranker.predict.
        space: callable ``(trial) -> dict`` for ranker init kwargs.
            Defaults to :func:`default_ranker_space`.
    """
    space = space or default_ranker_space

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        ranker = RankerModel(**params, random_state=seed)
        df_train, df_val = _split_for_ranker(labeled_df, seed=seed)
        ranker.fit(df_train, df_val)
        preds = ranker.predict(eval_features_df, n=k)
        # ``predict`` returns ``ranker_score`` — rename for ``recall_at_k``
        # which sorts by ``score`` if present (no-op on already top-k input,
        # but keeps the assertion happy).
        preds = preds.rename({"ranker_score": "score"})
        return recall_at_k(gt_val, preds, k=k)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
        sampler=sampler or optuna.samplers.TPESampler(seed=seed),
        direction="maximize",
    )
    log.info("tune_ranker: %d trials, k=%d, storage=%s", n_trials, k, storage)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study


def _split_for_ranker(
    df: pl.DataFrame, seed: int, test_size: float = 0.2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    pdf = df.to_pandas()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(pdf, groups=pdf["uid"]))
    return pl.from_pandas(pdf.iloc[train_idx]), pl.from_pandas(pdf.iloc[val_idx])


# ---------------------------------------------------------------------------
# B3 — n_cand allocation tuning
# ---------------------------------------------------------------------------

def tune_n_cand(
    scored_df: pl.DataFrame,
    gt_val: pl.DataFrame,
    cg_names: list[str],
    n_max_per_cg: int,
    total_budget: int,
    n_trials: int,
    k: int = 100,
    step: int = 25,
    score_col: str = "ranker_score",
    study_name: str | None = None,
    storage: str | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    seed: int = 42,
    show_progress_bar: bool = False,
) -> optuna.Study:
    """Tune per-CG ``n_cand`` allocation under a total-budget constraint.

    Operates on a **pre-scored** merged candidates DataFrame with one row
    per ``(uid, item_id)`` and ``{name}_rank`` columns from
    ``merge_candidates`` plus a ranker score column. No CG re-fit, no
    re-score per trial — only a boolean filter, per-user top-k, and Recall@k.

    Args:
        scored_df: merged-and-scored candidates. Must contain ``uid``,
            ``item_id``, ``score_col``, and ``{name}_rank`` for every name
            in ``cg_names``.
        cg_names: CGs whose budget to optimise.
        n_max_per_cg: hard ceiling on per-CG ``n_cand``; must equal the
            top-N used when generating ``scored_df``.
        total_budget: sum-of-n_cand cap. Trials over budget return 0
            (soft penalty so TPE still learns).
        step: ``trial.suggest_int`` step — coarsens the search space; 25
            keeps it tractable at ~20 levels per CG.
    """
    rank_cols = [f"{name}_rank" for name in cg_names]
    required = ["uid", "item_id", score_col, *rank_cols]
    missing = [c for c in required if c not in scored_df.columns]
    if missing:
        raise ValueError(
            f"tune_n_cand: scored_df missing required columns {missing}; "
            f"got {list(scored_df.columns)}"
        )

    def objective(trial: optuna.Trial) -> float:
        n_cands = {
            name: trial.suggest_int(f"n_cand_{name}", 0, n_max_per_cg, step=step)
            for name in cg_names
        }
        if sum(n_cands.values()) > total_budget:
            return 0.0  # soft penalty — over budget

        keep_expr = pl.lit(False)
        any_active = False
        for name, n in n_cands.items():
            if n <= 0:
                continue
            any_active = True
            rc = f"{name}_rank"
            keep_expr = keep_expr | (pl.col(rc).is_not_null() & (pl.col(rc) <= n))
        if not any_active:
            return 0.0

        filtered = scored_df.filter(keep_expr)
        top_k = RankerModel.top_k_per_user(filtered, k=k, score_col=score_col)
        # ``recall_at_k`` sorts by ``score`` if present; rename for compatibility.
        if score_col != "score":
            top_k = top_k.rename({score_col: "score"})
        return recall_at_k(gt_val, top_k, k=k)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
        sampler=sampler or optuna.samplers.TPESampler(seed=seed),
        direction="maximize",
    )
    log.info(
        "tune_n_cand: %d trials, %d CGs, n_max=%d, budget=%d, storage=%s",
        n_trials, len(cg_names), n_max_per_cg, total_budget, storage,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study


# ---------------------------------------------------------------------------
# Default search spaces (per docs/roadmap.md §D.2)
# ---------------------------------------------------------------------------

def default_ranker_space(trial: optuna.Trial) -> dict:
    """CatBoost YetiRank hyperparam space — Phase D.2 ranges."""
    return dict(
        iterations=trial.suggest_int("iterations", 500, 3000, step=250),
        depth=trial.suggest_int("depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
        early_stopping_rounds=trial.suggest_int("early_stopping_rounds", 50, 200, step=50),
    )


def default_decaypop_space(trial: optuna.Trial) -> dict:
    return dict(
        half_life_units=trial.suggest_int("half_life_units", 86_400, 1_036_800, log=True),
    )


# Repeat shares the same single-param space as DecayPop.
default_repeat_space = default_decaypop_space


def default_recent_likes_space(trial: optuna.Trial) -> dict:
    return dict(
        half_life_units=trial.suggest_int("half_life_units", 173_000, 1_728_000, log=True),
    )


def default_als_space(trial: optuna.Trial) -> dict:
    return dict(
        factors=trial.suggest_int("factors", 64, 512, log=True),
        iterations=trial.suggest_int("iterations", 10, 30),
        regularization=trial.suggest_float("regularization", 1e-3, 1.0, log=True),
        alpha=trial.suggest_float("alpha", 1.0, 100.0, log=True),
        low_engagement_weight=trial.suggest_float("low_engagement_weight", 0.0, 0.5),
        high_engagement_weight=trial.suggest_float("high_engagement_weight", 1.5, 5.0),
    )


def default_itemknn_space(trial: optuna.Trial) -> dict:
    return dict(k=trial.suggest_int("k", 50, 500))


def default_artist_album_pop_space(trial: optuna.Trial) -> dict:
    return dict(
        top_entities=trial.suggest_int("top_entities", 5, 30),
        half_life_units=trial.suggest_int("half_life_units", 86_400, 1_036_800, log=True),
    )


def default_audio_knn_space(trial: optuna.Trial) -> dict:
    return dict(
        user_history_k=trial.suggest_int("user_history_k", 5, 50),
        ef_search=trial.suggest_int("ef_search", 16, 256, log=True),
    )
