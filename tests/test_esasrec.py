"""Smoke tests for ESASRecModel (Phase C / A1-A2).

Goal: prove that fit / recommend / pickle round-trip on tiny synthetic
data. Real training quality is verified by val Recall@100 on the server
(Phase C), not here.
"""
from __future__ import annotations

import pickle

import numpy as np
import polars as pl
import pytest
import torch

from src.models.esasrec import ESASRecModel


@pytest.fixture(scope="module")
def tiny_listens() -> pl.DataFrame:
    """20 users × 30 items × 15 listens — enough to exercise sequence code."""
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(20):
        for j in range(15):
            rows.append((
                uid,
                int(rng.integers(0, 30)),  # item_id
                uid * 100 + j,              # timestamp (per-user monotonic)
                100,                         # played_ratio_pct (always positive)
                1,                           # is_organic
                60,                          # track_length_seconds
            ))
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
def tiny_model() -> ESASRecModel:
    """Tiny architecture for fast tests — 1 epoch, small dims, CPU."""
    return ESASRecModel(
        name="esasrec",
        n_cand=10,
        emb_dim=32,
        n_blocks=1,
        n_heads=2,
        dropout=0.0,
        sequence_max_len=10,
        n_negatives=8,
        max_epochs=1,
        lr=1e-2,
        batch_size=8,
        device="cpu",
        random_state=42,
    )


def test_fit_runs_and_populates_state(tiny_listens, tiny_model):
    tiny_model.fit(tiny_listens)
    assert tiny_model._model is not None
    # vocab = unique items in tiny_listens (≤30); index 0 reserved for pad
    n_items = len(tiny_model._item_to_idx)
    assert n_items >= 1
    assert n_items <= 30
    # one user history entry per user with ≥2 listens (all 20 here)
    assert 0 < len(tiny_model._user_history) <= 20


def test_recommend_returns_valid_schema(tiny_listens, tiny_model):
    tiny_model.fit(tiny_listens)
    users = sorted(tiny_listens["uid"].unique().to_list())
    out = tiny_model.recommend(users, n=5)

    assert out.columns == ["uid", "item_id", "score", "esasrec_rank"]
    assert out.schema["uid"] == pl.Int64
    assert out.schema["item_id"] == pl.Int64
    assert out.schema["score"] == pl.Float32
    assert out.schema["esasrec_rank"] == pl.Int32

    # at most 5 candidates per user, ranks 1..n
    by_user = out.group_by("uid").agg([
        pl.len().alias("n"),
        pl.col("esasrec_rank").min().alias("min_rank"),
        pl.col("esasrec_rank").max().alias("max_rank"),
    ])
    assert by_user["n"].max() <= 5
    assert by_user["min_rank"].min() == 1
    assert by_user["max_rank"].max() <= 5

    # no padding sentinel (-1) leaked into outputs
    assert (out["item_id"] >= 0).all()


def test_recommend_skips_users_without_history(tiny_listens, tiny_model):
    tiny_model.fit(tiny_listens)
    # one known + one unknown user
    out = tiny_model.recommend([0, 999_999], n=3)
    # Only uid=0 should appear (999_999 has no history)
    uids = set(out["uid"].unique().to_list())
    assert uids == {0}


def test_pickle_roundtrip_preserves_outputs(tiny_listens, tiny_model):
    tiny_model.fit(tiny_listens)
    users = sorted(tiny_listens["uid"].unique().to_list())[:5]
    before = tiny_model.recommend(users, n=3)

    blob = pickle.dumps(tiny_model)
    restored = pickle.loads(blob)
    after = restored.recommend(users, n=3)

    # exact equality of (uid, item_id, rank) + close score equality
    assert before["uid"].to_list() == after["uid"].to_list()
    assert before["item_id"].to_list() == after["item_id"].to_list()
    assert before["esasrec_rank"].to_list() == after["esasrec_rank"].to_list()
    np.testing.assert_allclose(
        before["score"].to_numpy(),
        after["score"].to_numpy(),
        rtol=1e-5, atol=1e-5,
    )


def test_recommend_raises_when_unfitted(tiny_model):
    with pytest.raises(RuntimeError, match="not fitted"):
        tiny_model.recommend([1, 2], n=3)


def test_resolves_explicit_cpu_device(tiny_listens, tiny_model):
    tiny_model.fit(tiny_listens)
    assert next(tiny_model._model.parameters()).device.type == "cpu"
