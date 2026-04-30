"""Tests for src.training.tune (Phase D / B1-B3).

Smoke-level: 2-5 trials each, tiny synthetic data. Verifies API contract
(study returned, trials recorded, best_value sensible), not search quality.
"""
from __future__ import annotations

import numpy as np
import optuna
import polars as pl
import pytest

from src.models.catboost_ranker import RankerModel
from src.models.pop import DecayPop
from src.training.tune import (
    default_decaypop_space,
    default_ranker_space,
    tune_candidate_generator,
    tune_n_cand,
    tune_ranker,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_listens() -> pl.DataFrame:
    """Synthetic positive listens: 5 users × 20 items × 3 timestamps each."""
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(5):
        for j in range(60):
            item_id = int(rng.integers(0, 20))
            ts = int(rng.integers(0, 100_000))
            rows.append((uid, item_id, ts, 100, 1, 60))
    return pl.DataFrame(
        rows,
        schema=["uid", "item_id", "timestamp", "played_ratio_pct", "is_organic", "track_length_seconds"],
        orient="row",
    ).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("timestamp").cast(pl.Int64),
        pl.col("played_ratio_pct").cast(pl.UInt16),
        pl.col("is_organic").cast(pl.UInt8),
        pl.col("track_length_seconds").cast(pl.UInt32),
    ])


@pytest.fixture
def tiny_gt(tiny_listens: pl.DataFrame) -> pl.DataFrame:
    """Use a subset of listens as held-out ground-truth for tests."""
    return (
        tiny_listens
        .select(["uid", "item_id"])
        .unique()
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
    )


@pytest.fixture
def tiny_labeled() -> pl.DataFrame:
    """Synthetic ranker-input: 30 users × 12 cands × {label, 3 features}."""
    rng = np.random.default_rng(1)
    rows = []
    for uid in range(30):
        for j in range(12):
            f0 = float(rng.normal())
            f1 = float(rng.normal())
            f2 = float(rng.normal())
            label = int((f0 + 0.5 * f1 - 0.2 * f2) > 0)
            rows.append((uid, uid * 1000 + j, label, f0, f1, f2))
    return pl.DataFrame(
        rows,
        schema=["uid", "item_id", "label", "f0", "f1", "f2"],
        orient="row",
    ).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("label").cast(pl.Int8),
    ])


@pytest.fixture
def tiny_gt_for_ranker(tiny_labeled: pl.DataFrame) -> pl.DataFrame:
    """Pretend all label==1 rows are GT for Recall@k computation."""
    return (
        tiny_labeled
        .filter(pl.col("label") == 1)
        .select(["uid", "item_id"])
    )


# ── B1: tune_candidate_generator ─────────────────────────────────────────


def test_tune_cg_runs_and_records_trials(tiny_listens, tiny_gt):
    """3 trials of DecayPop with half_life sweep — verify API contract."""
    eval_users = sorted(tiny_listens["uid"].unique().to_list())

    def factory(trial: optuna.Trial) -> DecayPop:
        params = default_decaypop_space(trial)
        return DecayPop(name="decaypop", n_cand=20, **params)

    study = tune_candidate_generator(
        model_factory=factory,
        train=tiny_listens,
        eval_users=eval_users,
        gt_val=tiny_gt,
        n_max=20,
        n_trials=3,
    )
    assert len(study.trials) == 3
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    # Recall@N_max is in [0, 1000] (×1000 scale)
    assert 0 <= study.best_value <= 1000
    assert "half_life_units" in study.best_params


# ── B2: tune_ranker ───────────────────────────────────────────────────────


def test_tune_ranker_runs_and_uses_custom_space(tiny_labeled, tiny_gt_for_ranker):
    """2 trials with a TINY custom space (iterations=10) for fast test."""
    def fast_space(trial: optuna.Trial) -> dict:
        return dict(
            iterations=10,
            depth=trial.suggest_int("depth", 3, 4),
            learning_rate=0.3,
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 5.0),
            early_stopping_rounds=5,
        )

    eval_features = tiny_labeled.drop("label")
    study = tune_ranker(
        labeled_df=tiny_labeled,
        eval_features_df=eval_features,
        gt_val=tiny_gt_for_ranker,
        n_trials=2,
        k=5,
        space=fast_space,
    )
    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert 0 <= study.best_value <= 1000
    assert "depth" in study.best_params
    assert "l2_leaf_reg" in study.best_params


def test_default_ranker_space_returns_required_keys():
    """The default space must produce the kwargs RankerModel.__init__ expects."""
    study = optuna.create_study()
    trial = study.ask()
    params = default_ranker_space(trial)
    expected = {"iterations", "depth", "learning_rate", "l2_leaf_reg", "early_stopping_rounds"}
    assert expected.issubset(params.keys())


# ── B3: tune_n_cand ───────────────────────────────────────────────────────


def _make_scored_df(rng: np.random.Generator, n_users: int = 8, n_items: int = 30) -> pl.DataFrame:
    """Synthetic 2-CG scored merged DataFrame.

    Each (uid, item_id) row carries one or both of ``cg_a_rank`` /
    ``cg_b_rank`` (NULL where the CG didn't propose) and ``ranker_score``.
    """
    rows = []
    for uid in range(n_users):
        for item_id in range(n_items):
            rank_a = (item_id + 1) if item_id < 10 else None       # cg_a covers items 0..9
            rank_b = (item_id - 5 + 1) if 5 <= item_id < 25 else None  # cg_b covers 5..24
            if rank_a is None and rank_b is None:
                continue  # not in either CG → not a candidate at all
            rows.append((uid, item_id, rank_a, rank_b, float(rng.normal())))
    return pl.DataFrame(
        rows,
        schema=["uid", "item_id", "cg_a_rank", "cg_b_rank", "ranker_score"],
        orient="row",
    ).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("cg_a_rank").cast(pl.Int32),
        pl.col("cg_b_rank").cast(pl.Int32),
        pl.col("ranker_score").cast(pl.Float64),
    ])


def test_tune_n_cand_runs_and_obeys_budget():
    rng = np.random.default_rng(2)
    scored = _make_scored_df(rng, n_users=6, n_items=30)
    # Pretend top-scored items per user are ground truth (pick top-3 by score).
    gt = (
        scored
        .sort(["uid", "ranker_score"], descending=[False, True])
        .group_by("uid", maintain_order=True)
        .head(3)
        .select(["uid", "item_id"])
    )

    study = tune_n_cand(
        scored_df=scored,
        gt_val=gt,
        cg_names=["cg_a", "cg_b"],
        n_max_per_cg=10,
        total_budget=15,
        n_trials=8,
        k=3,
        step=5,
    )

    assert len(study.trials) == 8
    # Best params must respect the budget (sum ≤ 15).
    bp = study.best_params
    assert bp["n_cand_cg_a"] + bp["n_cand_cg_b"] <= 15
    assert 0 <= study.best_value <= 1000


def test_tune_n_cand_rejects_missing_columns():
    scored = pl.DataFrame({
        "uid": [1, 2], "item_id": [10, 20], "cg_a_rank": [1, 1],
        # missing 'ranker_score', missing 'cg_b_rank'
    })
    with pytest.raises(ValueError, match="missing required columns"):
        tune_n_cand(
            scored_df=scored,
            gt_val=pl.DataFrame({"uid": [1], "item_id": [10]}),
            cg_names=["cg_a", "cg_b"],
            n_max_per_cg=10,
            total_budget=15,
            n_trials=2,
        )
