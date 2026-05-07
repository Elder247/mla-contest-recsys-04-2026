"""Tests for src.inference.merge_candidates."""
from __future__ import annotations

import polars as pl
import pytest

from src.inference.merge_candidates import (
    apply_n_cand_keep,
    cg_recall,
    merge_candidates,
)


def _cg_df(name: str, rows: list[tuple[int, int, float, int]]) -> pl.DataFrame:
    """Build a CG-output DataFrame: (uid, item_id, score, {name}_rank)."""
    return pl.DataFrame(
        rows,
        schema=["uid", "item_id", "score", f"{name}_rank"],
        orient="row",
    ).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("score").cast(pl.Float64),
        pl.col(f"{name}_rank").cast(pl.Int32),
    ])


@pytest.fixture
def merged_two_cgs() -> pl.DataFrame:
    """Outer-joined output of 2 CGs covering overlapping items.

    cg_a covers items 1..5 for uid=100 (rank 1..5).
    cg_b covers items 4..8 for uid=100 (rank 1..5; items 4 and 5 overlap).
    """
    a = _cg_df("cg_a", [(100, i, 1.0 / i, i) for i in range(1, 6)])
    b = _cg_df("cg_b", [(100, i, 0.5 / (i - 3), i - 3) for i in range(4, 9)])
    return merge_candidates({"cg_a": a, "cg_b": b})


# ── apply_n_cand_keep ─────────────────────────────────────────────────────


def test_apply_n_cand_keep_noop_when_no_cg_has_field(merged_two_cgs):
    """No CG block carries 'n_cand_keep' → returns merged unchanged."""
    cg_cfg_list = [
        {"name": "cg_a", "n_cand": 5},
        {"name": "cg_b", "n_cand": 5},
    ]
    out = apply_n_cand_keep(merged_two_cgs, cg_cfg_list)
    assert out.equals(merged_two_cgs)


def test_apply_n_cand_keep_keeps_rows_within_threshold(merged_two_cgs):
    """cg_a keep=2, cg_b keep=2 → only items reachable through one of them."""
    cg_cfg_list = [
        {"name": "cg_a", "n_cand": 5, "n_cand_keep": 2},
        {"name": "cg_b", "n_cand": 5, "n_cand_keep": 2},
    ]
    out = apply_n_cand_keep(merged_two_cgs, cg_cfg_list)
    # cg_a keeps items 1, 2 (rank 1, 2)
    # cg_b keeps items 4, 5 (cg_b_rank 1, 2)
    # Union: {1, 2, 4, 5}
    assert sorted(out["item_id"].to_list()) == [1, 2, 4, 5]


def test_apply_n_cand_keep_zero_excludes_unique_but_preserves_columns(merged_two_cgs):
    """cg_b keep=0 → cg_b unique items (6,7,8) drop, but cg_b_rank stays
    populated on rows kept by cg_a (items 4 and 5 overlap)."""
    cg_cfg_list = [
        {"name": "cg_a", "n_cand": 5, "n_cand_keep": 5},
        {"name": "cg_b", "n_cand": 5, "n_cand_keep": 0},
    ]
    out = apply_n_cand_keep(merged_two_cgs, cg_cfg_list)
    # cg_a fully kept → items 1..5 survive.
    # cg_b unique (6, 7, 8) dropped.
    assert sorted(out["item_id"].to_list()) == [1, 2, 3, 4, 5]
    # On items 4, 5 — cg_b_rank should still be non-null (overlap rows).
    overlap = out.filter(pl.col("item_id").is_in([4, 5]))
    assert overlap["cg_b_rank"].null_count() == 0
    # On items 1..3 — cg_b_rank is null (cg_a-only rows).
    cg_a_only = out.filter(pl.col("item_id").is_in([1, 2, 3]))
    assert cg_a_only["cg_b_rank"].null_count() == 3


def test_apply_n_cand_keep_raises_when_all_zero(merged_two_cgs):
    cg_cfg_list = [
        {"name": "cg_a", "n_cand": 5, "n_cand_keep": 0},
        {"name": "cg_b", "n_cand": 5, "n_cand_keep": 0},
    ]
    with pytest.raises(ValueError, match="every CG"):
        apply_n_cand_keep(merged_two_cgs, cg_cfg_list)


def test_apply_n_cand_keep_raises_on_missing_rank_column(merged_two_cgs):
    cg_cfg_list = [
        {"name": "cg_a", "n_cand": 5, "n_cand_keep": 2},
        {"name": "cg_missing", "n_cand": 5, "n_cand_keep": 2},
    ]
    with pytest.raises(ValueError, match="cg_missing_rank"):
        apply_n_cand_keep(merged_two_cgs, cg_cfg_list)


def test_apply_n_cand_keep_partial_field_treats_unset_as_no_gate(merged_two_cgs):
    """If only cg_a has n_cand_keep, cg_b doesn't gate but its columns
    survive on cg_a-kept rows."""
    cg_cfg_list = [
        {"name": "cg_a", "n_cand": 5, "n_cand_keep": 2},
        {"name": "cg_b", "n_cand": 5},  # no n_cand_keep
    ]
    out = apply_n_cand_keep(merged_two_cgs, cg_cfg_list)
    # Only cg_a contributes the keep predicate → items 1, 2 survive.
    # cg_b's unique items (6, 7, 8) and cg_b's contribution at rank 3..5
    # (items 4, 5) are NOT enough by themselves to keep a row.
    assert sorted(out["item_id"].to_list()) == [1, 2]


# ── cg_recall (sanity, unchanged behaviour) ──────────────────────────────


def test_cg_recall_basic():
    cands = pl.DataFrame({"uid": [1, 1, 2], "item_id": [10, 20, 30]}).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
    ])
    gt = pl.DataFrame({"uid": [1, 2], "item_id": [10, 99]}).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
    ])
    # uid=1: 1/1 hit (denom=1) → 1.0; uid=2: 0/1 → 0.0; mean ×1000 = 500.0
    assert cg_recall(cands, gt) == 500.0
