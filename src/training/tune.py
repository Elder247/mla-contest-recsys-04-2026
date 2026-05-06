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


def _maybe_enqueue_baseline(study: optuna.Study, baseline_params: dict | None) -> None:
    """Enqueue baseline params as the next trial when the study is fresh.

    No-op when ``baseline_params`` is None or the study already has trials
    (avoids re-running baseline on study resume).
    """
    if not baseline_params:
        return
    if len(study.trials) > 0:
        log.info(
            "skipping baseline enqueue: study has %d existing trials",
            len(study.trials),
        )
        return
    study.enqueue_trial(dict(baseline_params))
    log.info("enqueued baseline trial: %s", dict(baseline_params))


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
    baseline_params: dict | None = None,
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
    _maybe_enqueue_baseline(study, baseline_params)
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
    baseline_params: dict | None = None,
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
    _maybe_enqueue_baseline(study, baseline_params)
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
    baseline_params: dict | None = None,
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
    _maybe_enqueue_baseline(study, baseline_params)
    log.info(
        "tune_n_cand: %d trials, %d CGs, n_max=%d, budget=%d, storage=%s",
        n_trials, len(cg_names), n_max_per_cg, total_budget, storage,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study


# ---------------------------------------------------------------------------
# B4 — Joint ranker + n_cand allocation tuning
# ---------------------------------------------------------------------------

def tune_ranker_and_n_cand(
    labeled_df: pl.DataFrame,
    eval_features_df: pl.DataFrame,
    gt_val: pl.DataFrame,
    cg_names: list[str],
    n_max_per_cg: int,
    total_budget: int,
    n_trials: int,
    ranker_space: Callable[[optuna.Trial], dict] | None = None,
    k: int = 100,
    n_cand_step: int = 25,
    n_cand_min: int = 25,
    early_stopping_rounds: int = 150,
    task_type: str = "GPU",
    devices: str | None = "0",
    study_name: str | None = None,
    storage: str | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    seed: int = 42,
    show_progress_bar: bool = False,
    baseline_params: dict | None = None,
) -> optuna.Study:
    """Tune CatBoost YetiRank hyperparams *and* per-CG n_cand allocation jointly.

    Each trial:
      1. Samples ranker params (via ``ranker_space``) and per-CG n_cand
         (uniform int with ``n_cand_step``, range ``[n_cand_min, n_max_per_cg]``).
      2. Skips trial (returns 0.0) if ``sum(n_cand) > total_budget``.
      3. Builds a row-level keep mask: a row survives iff at least one CG has
         ``{name}_rank IS NOT NULL AND {name}_rank <= n_cand_per_cg``.
      4. Filters both ``labeled_df`` and ``eval_features_df`` with the same mask
         — this emulates "what if each CG had been called with n_cand_per_cg".
      5. Refits the ranker on the filtered training pool and computes
         ``Recall@k`` on the filtered eval pool against ``gt_val``.

    The keep-mask correctness assumes each CG's ``{name}_rank`` is the true
    top-by-score order at any N (i.e. CG outputs are deterministic with
    monotonic top-k). All current 8 CGs satisfy this with fixed seeds. If a
    new CG with stochastic top-k truncation is added, this function must be
    revisited.

    Positives (label=1) whose only contributing CG gets a small n_cand will
    be dropped from the training pool — this is the correct emulation of
    inference-time behaviour.

    Args:
        labeled_df: cached training features w/ ``label`` column. Each CG
            must have generated ``n_cand=n_max_per_cg`` candidates.
        eval_features_df: cached eval features (no label). Same n_max.
        cg_names: list of CG names whose budgets to optimise; must match
            ``{name}_rank`` columns in both DataFrames.
        n_max_per_cg: must equal the n_cand used to generate the cached
            features; sets the upper bound on n_cand_per_cg.
        total_budget: soft cap on sum(n_cand). Trials over budget return 0.
        early_stopping_rounds: fixed (not sampled) — passed to RankerModel.
        task_type / devices: passed to RankerModel; default "GPU"/"0".
    """
    ranker_space = ranker_space or default_ranker_space

    # Validate {name}_rank columns up-front so we fail fast, not per-trial.
    rank_cols = [f"{name}_rank" for name in cg_names]
    missing_labeled = [c for c in rank_cols if c not in labeled_df.columns]
    missing_eval = [c for c in rank_cols if c not in eval_features_df.columns]
    if missing_labeled:
        raise ValueError(
            f"tune_ranker_and_n_cand: labeled_df missing {missing_labeled}; "
            f"got {list(labeled_df.columns)}"
        )
    if missing_eval:
        raise ValueError(
            f"tune_ranker_and_n_cand: eval_features_df missing {missing_eval}; "
            f"got {list(eval_features_df.columns)}"
        )

    def objective(trial: optuna.Trial) -> float:
        ranker_params = ranker_space(trial)
        n_cands = {
            name: trial.suggest_int(
                f"n_cand_{name}", n_cand_min, n_max_per_cg, step=n_cand_step,
            )
            for name in cg_names
        }
        if sum(n_cands.values()) > total_budget:
            return 0.0  # soft penalty — over budget

        keep_expr = pl.lit(False)
        for name, c in n_cands.items():
            rc = f"{name}_rank"
            keep_expr = keep_expr | (
                pl.col(rc).is_not_null() & (pl.col(rc) <= c)
            )

        labeled_f = labeled_df.filter(keep_expr)
        eval_f = eval_features_df.filter(keep_expr)
        if len(labeled_f) == 0 or len(eval_f) == 0:
            return 0.0

        df_tr, df_va = _split_for_ranker(labeled_f, seed=seed)
        ranker = RankerModel(
            **ranker_params,
            random_state=seed,
            early_stopping_rounds=early_stopping_rounds,
            task_type=task_type,
            devices=devices,
        )
        ranker.fit(df_tr, df_va)
        preds = ranker.predict(eval_f, n=k).rename({"ranker_score": "score"})
        return recall_at_k(gt_val, preds, k=k)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
        sampler=sampler or optuna.samplers.TPESampler(seed=seed, multivariate=True),
        direction="maximize",
    )
    _maybe_enqueue_baseline(study, baseline_params)
    log.info(
        "tune_ranker_and_n_cand: %d trials, %d CGs, n_max=%d, budget=%d, "
        "task_type=%s, storage=%s",
        n_trials, len(cg_names), n_max_per_cg, total_budget, task_type, storage,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study


# ---------------------------------------------------------------------------
# B5 — Joint ranker + n_cand allocation tuning, v2 (anti-overfit)
# ---------------------------------------------------------------------------

def tune_ranker_and_n_cand_v2(
    labeled_df: pl.DataFrame,
    eval_features_df: pl.DataFrame,
    gt_val: pl.DataFrame,
    cg_names: list[str],
    *,
    gt_test: pl.DataFrame | None = None,
    combined_objective: bool = True,
    n_max_per_cg: int = 500,
    n_trials: int = 60,
    ranker_space: Callable[[optuna.Trial], dict] | None = None,
    k: int = 100,
    n_cand_step: int = 25,
    n_cand_min: int = 0,
    early_stopping_rounds: int = 150,
    task_type: str = "GPU",
    devices: str | None = "0",
    study_name: str | None = None,
    storage: str | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    n_startup_trials: int = 20,
    seed: int = 42,
    show_progress_bar: bool = False,
    baseline_params: dict | None = None,
) -> optuna.Study:
    """Anti-overfit variant of :func:`tune_ranker_and_n_cand`.

    Three changes from the v1 above:

    1. **No total-budget penalty.** Trials whose summed n_cand would have
       exceeded a soft cap are no longer forced to return 0 — TPE got too
       much zero-noise from this in the original 142-trial joint study.
       The downstream :class:`RankerModel.fit` already caps to 1023 rows
       per user on GPU via per-row RRF score, so an arbitrarily large
       merged pool is safe. Optuna learns the true Recall(n_cand) surface.

    2. **n_cand_min defaults to 0.** A CG can be effectively disabled when
       its contribution is dominated by other CGs — useful for newly
       added CGs whose value is uncertain (e.g. a freshly retrained
       eSASRec). The v1 floor of 25 forced every CG into every trial.

    3. **Combined val + test objective** (``combined_objective=True``,
       ``gt_test`` provided): metric is ``0.5 * (recall_val + recall_test)``.
       The temporal_split test window is +1 day from val; averaging the
       two recalls reduces variance ~√2 and shrinks the val→public gap
       observed in the original joint Optuna run (val +13.6 vs public +5.4).

    Args:
        labeled_df / eval_features_df: cached ``train_ranker.py`` outputs
            with ``{name}_rank`` columns at ``n_cand=n_max_per_cg``.
        gt_val: validation ground truth (always required).
        gt_test: test ground truth (required when ``combined_objective``).
        cg_names: CGs whose ``n_cand_*`` to optimise. Each must have
            ``{name}_rank`` in both DataFrames.
        n_max_per_cg: must equal the n_cand used to generate the cached
            features; sets the upper bound on ``n_cand_*`` (default 500).
        n_cand_min: lower bound on each ``n_cand_*`` (default 0).
        n_startup_trials: random exploration before TPE kicks in
            (default 20 — wider than v1 to weaken multi-test bias).
        early_stopping_rounds, task_type, devices: passed to RankerModel.

    Returns:
        Fitted ``optuna.Study``. Direction is "maximize".
    """
    ranker_space = ranker_space or default_ranker_space

    if combined_objective and gt_test is None:
        raise ValueError(
            "tune_ranker_and_n_cand_v2: combined_objective=True requires "
            "gt_test; pass gt_test or set combined_objective=False"
        )

    rank_cols = [f"{name}_rank" for name in cg_names]
    missing_labeled = [c for c in rank_cols if c not in labeled_df.columns]
    missing_eval = [c for c in rank_cols if c not in eval_features_df.columns]
    if missing_labeled:
        raise ValueError(
            f"tune_ranker_and_n_cand_v2: labeled_df missing {missing_labeled}; "
            f"got {list(labeled_df.columns)}"
        )
    if missing_eval:
        raise ValueError(
            f"tune_ranker_and_n_cand_v2: eval_features_df missing {missing_eval}; "
            f"got {list(eval_features_df.columns)}"
        )

    def objective(trial: optuna.Trial) -> float:
        ranker_params = ranker_space(trial)
        n_cands = {
            name: trial.suggest_int(
                f"n_cand_{name}", n_cand_min, n_max_per_cg, step=n_cand_step,
            )
            for name in cg_names
        }

        # No budget penalty — RankerModel caps to 1023 rows/user on GPU.
        keep_expr = pl.lit(False)
        any_active = False
        for name, c in n_cands.items():
            if c <= 0:
                continue
            any_active = True
            rc = f"{name}_rank"
            keep_expr = keep_expr | (
                pl.col(rc).is_not_null() & (pl.col(rc) <= c)
            )
        if not any_active:
            return 0.0

        labeled_f = labeled_df.filter(keep_expr)
        eval_f = eval_features_df.filter(keep_expr)
        if len(labeled_f) == 0 or len(eval_f) == 0:
            return 0.0

        df_tr, df_va = _split_for_ranker(labeled_f, seed=seed)
        ranker = RankerModel(
            **ranker_params,
            random_state=seed,
            early_stopping_rounds=early_stopping_rounds,
            task_type=task_type,
            devices=devices,
        )
        ranker.fit(df_tr, df_va)

        # ``score`` once, then top_k_per_user — letting us compute
        # recall on val *and* test from the same predictions.
        scored = ranker.score(eval_f)
        top_k = (
            RankerModel.top_k_per_user(scored, k=k, score_col="ranker_score")
            .rename({"ranker_score": "score"})
        )

        recall_val = recall_at_k(gt_val, top_k, k=k)
        if combined_objective:
            recall_test = recall_at_k(gt_test, top_k, k=k)
            trial.set_user_attr("recall_val", recall_val)
            trial.set_user_attr("recall_test", recall_test)
            return 0.5 * (recall_val + recall_test)
        return recall_val

    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            seed=seed, multivariate=True, n_startup_trials=n_startup_trials,
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
        sampler=sampler,
        direction="maximize",
    )
    _maybe_enqueue_baseline(study, baseline_params)
    log.info(
        "tune_ranker_and_n_cand_v2: %d trials, %d CGs, n_max=%d, n_min=%d, "
        "combined=%s, task_type=%s, storage=%s",
        n_trials, len(cg_names), n_max_per_cg, n_cand_min, combined_objective,
        task_type, storage,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study


# ---------------------------------------------------------------------------
# Default search spaces (per docs/roadmap.md §D.2)
# ---------------------------------------------------------------------------

def default_ranker_space(trial: optuna.Trial) -> dict:
    """CatBoost YetiRank hyperparam space.

    early_stopping_rounds is **not** sampled here — it is fixed at the call
    site (e.g. ``RankerModel(early_stopping_rounds=150, ...)``) to keep the
    Optuna search dimension lower. Past tuning showed best ES in 100-200
    range with little sensitivity, so fixing 150 is a safe default.
    """
    return dict(
        iterations=trial.suggest_int("iterations", 1500, 4000, step=250),
        depth=trial.suggest_int("depth", 4, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
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
        factors=trial.suggest_int("factors", 128, 1024, log=True),
        iterations=trial.suggest_int("iterations", 10, 30),
        regularization=trial.suggest_float("regularization", 1e-4, 1.0, log=True),
        alpha=trial.suggest_float("alpha", 0.5, 100.0, log=True),
        low_engagement_weight=trial.suggest_float("low_engagement_weight", 0.0, 0.5),
        high_engagement_weight=trial.suggest_float("high_engagement_weight", 1.5, 5.0),
    )


def default_itemknn_space(trial: optuna.Trial) -> dict:
    return dict(k=trial.suggest_int("k", 1, 100))


def default_artist_album_pop_space(trial: optuna.Trial) -> dict:
    return dict(
        top_entities=trial.suggest_int("top_entities", 5, 30),
        half_life_units=trial.suggest_int("half_life_units", 86_400, 1_036_800, log=True),
    )


def default_audio_knn_space(trial: optuna.Trial) -> dict:
    return dict(
        user_history_k=trial.suggest_int("user_history_k", 5, 50),
        hnsw_m=trial.suggest_int("hnsw_m", 16, 64),
        ef_construction=trial.suggest_int("ef_construction", 100, 400),
        ef_search=trial.suggest_int("ef_search", 32, 128),
    )
