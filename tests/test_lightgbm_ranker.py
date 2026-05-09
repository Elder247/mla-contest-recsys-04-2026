"""Tests for :class:`src.models.lightgbm_ranker.LightGBMRanker`.

Smoke-level: tiny synthetic data, verify fit/score/top_k contract.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.lightgbm_ranker import LightGBMRanker


@pytest.fixture
def tiny_labeled() -> pl.DataFrame:
    """Synthetic ranker-input: 30 users × 12 cands × {label, 3 features}.

    Mirrors the fixture in ``test_tune.py`` so both rankers train on the
    same shape of toy data.
    """
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


def test_fit_score_top_k_smoke(tiny_labeled: pl.DataFrame) -> None:
    ranker = LightGBMRanker(n_estimators=20, num_leaves=7, min_child_samples=2)
    train = tiny_labeled.head(240)
    val = tiny_labeled.tail(120)
    ranker.fit(train, val)

    scored = ranker.score(tiny_labeled)
    assert scored.columns == ["uid", "item_id", "lgbm_score"]
    assert len(scored) == len(tiny_labeled)
    assert scored["lgbm_score"].dtype == pl.Float32

    top_k = LightGBMRanker.top_k_per_user(scored, k=3)
    counts = top_k.group_by("uid").len()["len"].to_list()
    assert all(c <= 3 for c in counts)
    # And ``predict`` is the composition.
    preds = ranker.predict(tiny_labeled, n=3)
    assert preds.columns == ["uid", "item_id", "lgbm_score"]
    assert len(preds) == top_k.shape[0]


def test_score_chunking_matches_single_pass(tiny_labeled: pl.DataFrame) -> None:
    ranker = LightGBMRanker(n_estimators=10, num_leaves=4, min_child_samples=2)
    ranker.fit(tiny_labeled)
    full = ranker.score(tiny_labeled, chunk_size=10_000)
    chunked = ranker.score(tiny_labeled, chunk_size=37)
    np.testing.assert_allclose(
        full["lgbm_score"].to_numpy(),
        chunked["lgbm_score"].to_numpy(),
        atol=1e-6,
    )


def test_score_empty_input(tiny_labeled: pl.DataFrame) -> None:
    ranker = LightGBMRanker(n_estimators=5, num_leaves=4, min_child_samples=2)
    ranker.fit(tiny_labeled)
    empty = tiny_labeled.head(0)
    out = ranker.score(empty)
    assert out.columns == ["uid", "item_id", "lgbm_score"]
    assert len(out) == 0


def test_score_before_fit_raises(tiny_labeled: pl.DataFrame) -> None:
    with pytest.raises(RuntimeError):
        LightGBMRanker().score(tiny_labeled)


def test_negative_subsampling_keeps_all_positives(tiny_labeled: pl.DataFrame) -> None:
    """With ``negative_ratio=2``, target_neg = 2 x n_pos; positives must survive."""
    ranker = LightGBMRanker(
        negative_ratio=2,
        n_estimators=5,
        num_leaves=4,
        min_child_samples=2,
    )
    n_pos = int((tiny_labeled["label"] == 1).sum())
    sub = ranker._maybe_subsample(tiny_labeled.sort("uid"), kind="train")
    assert int((sub["label"] == 1).sum()) == n_pos
    assert int((sub["label"] != 1).sum()) <= 2 * n_pos


def test_negative_subsampling_disabled_when_none() -> None:
    ranker = LightGBMRanker(negative_ratio=None)
    df = pl.DataFrame({
        "uid": [1, 1, 1],
        "item_id": [10, 20, 30],
        "label": [1, 0, 0],
        "f0": [0.1, 0.2, 0.3],
    }).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("label").cast(pl.Int8),
    ])
    out = ranker._maybe_subsample(df, kind="train")
    assert len(out) == len(df)
