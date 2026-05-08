"""Tests for src.inference.merge_candidates."""
from __future__ import annotations

import polars as pl
import pytest

from src.inference.merge_candidates import (
    apply_n_cand_keep,
    cg_recall,
    compute_cg_aggregates,
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


# ── compute_cg_aggregates ────────────────────────────────────────────────


def test_compute_cg_aggregates_adds_two_columns(merged_two_cgs):
    """Output schema must contain ``cg_count`` and ``cg_mean_score_norm``.

    The rank-derived aggregates (``cg_min/max/mean_rank``, ``cg_rrf_score``)
    were intentionally removed — they leak the ranker's GPU train-pool
    selection criterion (``head(1023)`` keyed on RRF over ``*_rank`` cols).
    See ``compute_cg_aggregates`` docstring.
    """
    cg_cfg_list = [{"name": "cg_a"}, {"name": "cg_b"}]
    out = compute_cg_aggregates(merged_two_cgs, cg_cfg_list)
    assert {"cg_count", "cg_mean_score_norm"}.issubset(set(out.columns))
    assert out["cg_count"].dtype == pl.Int32
    assert out["cg_mean_score_norm"].dtype == pl.Float32
    # Removed leaky columns must NOT be present.
    leaky = {"cg_min_rank", "cg_max_rank", "cg_mean_rank", "cg_rrf_score"}
    assert leaky.isdisjoint(set(out.columns)), (
        f"Leaky aggregate columns must not be added: {leaky & set(out.columns)}"
    )


def test_compute_cg_aggregates_count_correct_on_overlap(merged_two_cgs):
    """Hand-computed cg_count for items 1 (cg_a only), 4 (both), 8 (cg_b only)."""
    cg_cfg_list = [{"name": "cg_a"}, {"name": "cg_b"}]
    out = compute_cg_aggregates(merged_two_cgs, cg_cfg_list).sort("item_id")

    row1 = out.filter(pl.col("item_id") == 1).row(0, named=True)
    assert row1["cg_count"] == 1

    row4 = out.filter(pl.col("item_id") == 4).row(0, named=True)
    assert row4["cg_count"] == 2

    row8 = out.filter(pl.col("item_id") == 8).row(0, named=True)
    assert row8["cg_count"] == 1


def test_compute_cg_aggregates_score_norm_per_cg_minmax(merged_two_cgs):
    """``cg_mean_score_norm`` is mean of per-CG MinMax-normalized scores.

    cg_a scores: 1.0, 0.5, 0.333, 0.25, 0.2 → range [0.2, 1.0]
    cg_b scores: 0.5, 0.25, 0.166, 0.125, 0.1 → range [0.1, 0.5]
    item_id=1 (cg_a only, score=1.0)        → norm_a=(1.0-0.2)/(1.0-0.2) = 1.0; mean=1.0
    item_id=8 (cg_b only, score=0.1)        → norm_b=(0.1-0.1)/(0.5-0.1) = 0.0; mean=0.0
    """
    cg_cfg_list = [{"name": "cg_a"}, {"name": "cg_b"}]
    out = compute_cg_aggregates(merged_two_cgs, cg_cfg_list)

    row1 = out.filter(pl.col("item_id") == 1).row(0, named=True)
    assert row1["cg_mean_score_norm"] == pytest.approx(1.0, abs=1e-5)

    row8 = out.filter(pl.col("item_id") == 8).row(0, named=True)
    assert row8["cg_mean_score_norm"] == pytest.approx(0.0, abs=1e-5)

    # All values in [0, 1] (both CGs are MinMax-normalized to [0, 1]
    # before the horizontal mean).
    norms = out["cg_mean_score_norm"].to_list()
    assert all(0.0 - 1e-5 <= v <= 1.0 + 1e-5 for v in norms)


def test_compute_cg_aggregates_does_not_mutate_original_score_columns(merged_two_cgs):
    """Original ``cg_a_score`` / ``cg_b_score`` must be preserved unchanged."""
    cg_cfg_list = [{"name": "cg_a"}, {"name": "cg_b"}]
    before_a = merged_two_cgs["cg_a_score"].to_list()
    before_b = merged_two_cgs["cg_b_score"].to_list()
    out = compute_cg_aggregates(merged_two_cgs, cg_cfg_list)
    assert out["cg_a_score"].to_list() == before_a
    assert out["cg_b_score"].to_list() == before_b
    # Row count preserved.
    assert len(out) == len(merged_two_cgs)


def test_compute_cg_aggregates_missing_cg_in_merged_skipped(merged_two_cgs):
    """A CG named in the cfg list but absent from the DataFrame is ignored
    (e.g. someone disabled a CG without removing its config block)."""
    cg_cfg_list = [
        {"name": "cg_a"},
        {"name": "cg_b"},
        {"name": "cg_phantom"},  # no cg_phantom_rank column in merged
    ]
    out = compute_cg_aggregates(merged_two_cgs, cg_cfg_list)
    # Same behaviour as the 2-CG case for cg_count on item 4 (both real CGs).
    row4 = out.filter(pl.col("item_id") == 4).row(0, named=True)
    assert row4["cg_count"] == 2


def test_compute_cg_aggregates_no_rank_columns_returns_unchanged():
    """Empty cfg or all CGs absent from merged → no-op (no aggregate cols)."""
    df = pl.DataFrame({
        "uid": [1, 1, 2],
        "item_id": [10, 20, 30],
    }).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
    ])
    cg_cfg_list = [{"name": "cg_missing"}]  # not in df
    out = compute_cg_aggregates(df, cg_cfg_list)
    assert out.equals(df)


def test_compute_cg_aggregates_constant_score_column_skipped():
    """If a CG's score column is constant, MinMax normalization is undefined
    → that CG is dropped from the score-norm mean (other CGs still count)."""
    cg_a = _cg_df("cg_a", [(1, 10, 1.0, 1), (1, 20, 0.5, 2)])
    cg_b = _cg_df("cg_b", [(1, 10, 0.7, 1), (1, 20, 0.7, 2)])  # constant
    merged = merge_candidates({"cg_a": cg_a, "cg_b": cg_b})
    out = compute_cg_aggregates(merged, [{"name": "cg_a"}, {"name": "cg_b"}])

    # cg_b score is constant — only cg_a contributes to cg_mean_score_norm.
    # cg_a scores 1.0/0.5 normalize to 1.0/0.0 → mean per row = 1.0/0.0.
    rows = {r["item_id"]: r for r in out.iter_rows(named=True)}
    assert rows[10]["cg_mean_score_norm"] == pytest.approx(1.0, abs=1e-5)
    assert rows[20]["cg_mean_score_norm"] == pytest.approx(0.0, abs=1e-5)


def test_compute_cg_aggregates_count_uses_only_present_ranks():
    """``cg_count`` reflects how many CGs returned the item.

    Singleton items (only one CG saw them) → cg_count == 1.
    """
    cg_a = _cg_df("cg_a", [(1, 10, 0.9, 5)])  # only cg_a sees item 10
    cg_b = _cg_df("cg_b", [(1, 99, 0.1, 5)])  # only cg_b sees item 99
    merged = merge_candidates({"cg_a": cg_a, "cg_b": cg_b})
    out = compute_cg_aggregates(merged, [{"name": "cg_a"}, {"name": "cg_b"}])
    rows = {r["item_id"]: r for r in out.iter_rows(named=True)}
    assert rows[10]["cg_count"] == 1
    assert rows[99]["cg_count"] == 1


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
