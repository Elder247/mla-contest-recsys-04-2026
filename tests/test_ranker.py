"""Tests for src.models.catboost_ranker.RankerModel.

Focus: the new ``score`` / ``top_k_per_user`` split (Phase D1+D2):
  - ``score`` chunked vs non-chunked produces identical scores
  - row order is preserved through ``score``
  - ``predict`` == ``score`` then ``top_k_per_user``
  - ``top_k_per_user`` keeps at most k rows per user, ordered by score desc
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.catboost_ranker import RankerModel


@pytest.fixture(scope="module")
def fitted_ranker() -> tuple[RankerModel, pl.DataFrame]:
    """Tiny ranker fitted on synthetic data — usable across multiple tests."""
    rng = np.random.default_rng(0)
    n_users, n_per_user = 30, 12
    rows = []
    for uid in range(n_users):
        for j in range(n_per_user):
            f0 = float(rng.normal())
            f1 = float(rng.normal())
            f2 = float(rng.normal())
            # label is loosely correlated with features so YetiRank has signal
            label = int((f0 + 0.5 * f1 - 0.2 * f2) > 0)
            rows.append((uid, uid * 1000 + j, label, f0, f1, f2))
    df = pl.DataFrame(
        rows,
        schema=["uid", "item_id", "label", "f0", "f1", "f2"],
        orient="row",
    ).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("label").cast(pl.Int8),
    ])

    ranker = RankerModel(iterations=20, depth=3, learning_rate=0.3)
    # Suppress CatBoost training noise for cleaner test output.
    ranker.fit(df)
    return ranker, df


def test_score_returns_three_cols_and_preserves_row_count(fitted_ranker):
    ranker, df = fitted_ranker
    out = ranker.score(df)
    assert out.columns == ["uid", "item_id", "ranker_score"]
    assert len(out) == len(df)


def test_score_preserves_row_order(fitted_ranker):
    ranker, df = fitted_ranker
    out = ranker.score(df)
    # uid + item_id sequence must match input row-for-row
    assert out["uid"].to_list() == df["uid"].to_list()
    assert out["item_id"].to_list() == df["item_id"].to_list()


def test_score_chunked_equals_unchunked(fitted_ranker):
    """The whole point of D1: chunking must not change the predictions."""
    ranker, df = fitted_ranker
    full = ranker.score(df, chunk_size=10_000_000)
    chunked = ranker.score(df, chunk_size=37)  # forces multiple chunks
    np.testing.assert_allclose(
        full["ranker_score"].to_numpy(),
        chunked["ranker_score"].to_numpy(),
        rtol=0,
        atol=0,
    )


def test_score_handles_empty_df(fitted_ranker):
    ranker, df = fitted_ranker
    empty = df.head(0)
    out = ranker.score(empty)
    assert len(out) == 0
    assert "ranker_score" in out.columns


def test_top_k_per_user_caps_rows_and_orders_by_score():
    df = pl.DataFrame({
        "uid": [1, 1, 1, 2, 2, 2, 2],
        "item_id": [10, 11, 12, 20, 21, 22, 23],
        "ranker_score": [0.1, 0.9, 0.5, 0.2, 0.8, 0.3, 0.7],
    }).with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])

    out = RankerModel.top_k_per_user(df, k=2)
    grouped = (
        out.group_by("uid", maintain_order=True)
        .agg(pl.col("item_id"), pl.col("ranker_score"))
        .sort("uid")
    )

    # uid=1: top-2 by score desc → 11 (0.9), 12 (0.5)
    # uid=2: top-2 by score desc → 21 (0.8), 23 (0.7)
    row_uid1 = grouped.filter(pl.col("uid") == 1).row(0, named=True)
    row_uid2 = grouped.filter(pl.col("uid") == 2).row(0, named=True)
    assert row_uid1["item_id"] == [11, 12]
    assert row_uid2["item_id"] == [21, 23]
    # at most k rows per user
    counts = out.group_by("uid").agg(pl.len().alias("n"))["n"].to_list()
    assert all(c <= 2 for c in counts)


def test_top_k_per_user_with_custom_score_col():
    df = pl.DataFrame({"uid": [1, 1], "item_id": [10, 11], "my_score": [0.1, 0.9]})
    out = RankerModel.top_k_per_user(df, k=1, score_col="my_score")
    assert out.columns == ["uid", "item_id", "my_score"]
    assert out["item_id"].to_list() == [11]


def test_predict_equals_score_then_top_k(fitted_ranker):
    """predict() must be a pure composition of score → top_k_per_user."""
    ranker, df = fitted_ranker
    direct = ranker.predict(df, n=3)
    composed = RankerModel.top_k_per_user(ranker.score(df), k=3)
    # Both sort by (uid asc, score desc); order should match exactly.
    assert direct["uid"].to_list() == composed["uid"].to_list()
    assert direct["item_id"].to_list() == composed["item_id"].to_list()
    np.testing.assert_allclose(
        direct["ranker_score"].to_numpy(),
        composed["ranker_score"].to_numpy(),
    )


def test_score_raises_when_unfitted():
    ranker = RankerModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        ranker.score(pl.DataFrame({"uid": [1], "item_id": [2], "f0": [0.0]}))
