"""Tests for the post-features ``_apply_n_cand_keep`` helpers in
:mod:`scripts.refit_ranker_topk` and :mod:`scripts.submit_ranker_topk`.

The helpers mirror :func:`src.inference.merge_candidates.apply_n_cand_keep`
but operate on a features parquet (columns ``{cg}_rank`` already present
from the merge step). The contract — same OR-by-CG row-keep semantics —
is critical: if it diverges, refit_ranker_topk produces a different pool
than the orthodox train_ranker.py path and the optuna optimum doesn't
transfer.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

from src.inference.merge_candidates import apply_n_cand_keep, merge_candidates


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _load_script_module(filename: str, module_name: str):
    """Load a top-level scripts/*.py file as a module.

    The scripts/ directory is not a package (no __init__.py), so we go
    through importlib's spec_from_file_location to pull just the helpers
    without triggering Hydra's @hydra.main decorator.
    """
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def refit_topk():
    return _load_script_module("refit_ranker_topk.py", "_refit_ranker_topk")


@pytest.fixture(scope="module")
def submit_topk():
    return _load_script_module("submit_ranker_topk.py", "_submit_ranker_topk")


# ── Synthetic features parquet (post-merge, post-add_features schema) ────────


def _cg_df(name: str, rows: list[tuple[int, int, float, int]]) -> pl.DataFrame:
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
def merged_three_cgs() -> pl.DataFrame:
    """Outer-merge of 3 CGs covering overlapping items for one uid.

    cg_a: items 1..5 (rank 1..5)
    cg_b: items 4..8 (rank 1..5; items 4,5 overlap with cg_a)
    cg_c: items 6..10 (rank 1..5; items 6,7,8 overlap with cg_b)

    After merge_candidates, columns are: uid, item_id,
    cg_a_score, cg_a_rank, cg_b_score, cg_b_rank, cg_c_score, cg_c_rank.
    Pretend this is the FULL n_cand=800 pool from gen_candidates — features
    parquet just adds more columns on top, the rank columns stay intact.
    """
    a = _cg_df("cg_a", [(100, i, 1.0 / i, i) for i in range(1, 6)])
    b = _cg_df("cg_b", [(100, i, 0.5 / (i - 3), i - 3) for i in range(4, 9)])
    c = _cg_df("cg_c", [(100, i, 0.2 / (i - 5), i - 5) for i in range(6, 11)])
    return merge_candidates({"cg_a": a, "cg_b": b, "cg_c": c})


def _add_dummy_feature_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Simulate features parquet — extra columns must not interfere."""
    return df.with_columns(
        [
            pl.lit(0.5, dtype=pl.Float32).alias("user_n_listens"),
            pl.lit(1, dtype=pl.Int8).alias("label"),
            pl.lit(0.0, dtype=pl.Float32).alias("pair_decay_listens"),
        ]
    )


# ── _apply_n_cand_keep ────────────────────────────────────────────────────────


def test_noop_when_no_cg_has_field(refit_topk, merged_three_cgs):
    """No CG carries ``n_cand_keep`` → returned df is row-equivalent to input."""
    feats = _add_dummy_feature_columns(merged_three_cgs)
    cgs = [
        {"name": "cg_a", "n_cand": 5},
        {"name": "cg_b", "n_cand": 5},
        {"name": "cg_c", "n_cand": 5},
    ]
    out = refit_topk._apply_n_cand_keep(feats, cgs)
    assert len(out) == len(feats)


def test_filter_drops_only_outside_keep_ranges(refit_topk, merged_three_cgs):
    """Only rows with at least one CG-rank ≤ keep survive."""
    feats = _add_dummy_feature_columns(merged_three_cgs)
    cgs = [
        {"name": "cg_a", "n_cand": 5, "n_cand_keep": 2},  # ranks 1..2
        {"name": "cg_b", "n_cand": 5, "n_cand_keep": 0},  # gates none
        {"name": "cg_c", "n_cand": 5, "n_cand_keep": 1},  # rank 1 only
    ]
    out = refit_topk._apply_n_cand_keep(feats, cgs)

    # cg_a rank ≤ 2 → items 1, 2 (rank 1, 2 in cg_a)
    # cg_c rank ≤ 1 → item 6 (rank 1 in cg_c)
    expected_items = {1, 2, 6}
    survivors = set(out["item_id"].to_list())
    assert survivors == expected_items


def test_matches_apply_n_cand_keep_exact(refit_topk, merged_three_cgs):
    """Post-features helper must produce the SAME row set as the merge-time
    function for the same CG list — otherwise refit_ranker_topk diverges
    from the orthodox train_ranker.py pipeline.
    """
    feats = _add_dummy_feature_columns(merged_three_cgs)
    cgs = [
        {"name": "cg_a", "n_cand": 5, "n_cand_keep": 3},
        {"name": "cg_b", "n_cand": 5, "n_cand_keep": 2},
        {"name": "cg_c", "n_cand": 5, "n_cand_keep": 0},
    ]
    # Reference: merge-time filter.
    ref = apply_n_cand_keep(merged_three_cgs, cgs)
    ref_keys = set(zip(ref["uid"].to_list(), ref["item_id"].to_list()))

    # Post-features filter on enriched dataframe.
    got = refit_topk._apply_n_cand_keep(feats, cgs)
    got_keys = set(zip(got["uid"].to_list(), got["item_id"].to_list()))

    assert got_keys == ref_keys, (
        f"refit_ranker_topk._apply_n_cand_keep diverges from "
        f"merge_candidates.apply_n_cand_keep:\n"
        f"  ref - got = {ref_keys - got_keys}\n"
        f"  got - ref = {got_keys - ref_keys}"
    )


def test_raises_when_all_keeps_zero(refit_topk, merged_three_cgs):
    """Same edge case as merge-time — empty pool is a config error."""
    feats = _add_dummy_feature_columns(merged_three_cgs)
    cgs = [
        {"name": "cg_a", "n_cand": 5, "n_cand_keep": 0},
        {"name": "cg_b", "n_cand": 5, "n_cand_keep": 0},
        {"name": "cg_c", "n_cand": 5, "n_cand_keep": 0},
    ]
    with pytest.raises(ValueError, match="every CG with 'n_cand_keep' set was 0"):
        refit_topk._apply_n_cand_keep(feats, cgs)


def test_raises_when_rank_column_missing(refit_topk, merged_three_cgs):
    """A CG name with no matching {name}_rank column is a hard error."""
    feats = _add_dummy_feature_columns(merged_three_cgs)
    cgs = [
        {"name": "cg_a", "n_cand": 5, "n_cand_keep": 2},
        {"name": "cg_phantom", "n_cand": 5, "n_cand_keep": 1},
    ]
    with pytest.raises(ValueError, match="cg_phantom_rank"):
        refit_topk._apply_n_cand_keep(feats, cgs)


def test_null_ranks_dont_keep_rows(refit_topk):
    """A row with NULL rank for a gating CG must NOT be kept by that CG.

    apply_n_cand_keep uses rank.is_not_null() & rank ≤ keep — the helper
    must do the same. Otherwise outer-join NULL fills (rows that exist in
    one CG only) would be incorrectly retained by other CGs.
    """
    df = pl.DataFrame(
        {
            "uid": [100, 100],
            "item_id": [1, 2],
            "cg_a_rank": [1, None],
            "cg_a_score": [1.0, None],
            "cg_b_rank": [None, 5],
            "cg_b_score": [None, 0.1],
        }
    ).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("cg_a_rank").cast(pl.Int32),
        pl.col("cg_b_rank").cast(pl.Int32),
    ])
    cgs = [
        {"name": "cg_a", "n_cand": 5, "n_cand_keep": 1},
        {"name": "cg_b", "n_cand": 5, "n_cand_keep": 1},
    ]
    out = refit_topk._apply_n_cand_keep(df, cgs)
    # Item 1: cg_a_rank=1 ≤ 1 → kept (cg_b NULL doesn't interfere).
    # Item 2: cg_a NULL, cg_b_rank=5 > 1 → dropped.
    assert out["item_id"].to_list() == [1]


def test_submit_helper_matches_refit_helper(refit_topk, submit_topk, merged_three_cgs):
    """The submit-side helper is a literal copy — must agree with refit-side
    on every input. If somebody edits one without the other, this test
    catches it before a server run.
    """
    feats = _add_dummy_feature_columns(merged_three_cgs)
    cgs = [
        {"name": "cg_a", "n_cand": 5, "n_cand_keep": 3},
        {"name": "cg_b", "n_cand": 5, "n_cand_keep": 2},
        {"name": "cg_c", "n_cand": 5, "n_cand_keep": 1},
    ]
    a = refit_topk._apply_n_cand_keep(feats, cgs)
    b = submit_topk._apply_n_cand_keep(feats, cgs)
    assert (
        sorted(zip(a["uid"].to_list(), a["item_id"].to_list()))
        == sorted(zip(b["uid"].to_list(), b["item_id"].to_list()))
    )


# ── _cascade_cut sanity (refit_ranker_topk only — submit uses inline polars) ──


def test_cascade_cut_keeps_top_n_per_user(refit_topk):
    feats = pl.DataFrame(
        {
            "uid": [100, 100, 100, 200, 200],
            "item_id": [1, 2, 3, 4, 5],
            "user_n_listens": [10, 10, 10, 20, 20],
        }
    ).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("user_n_listens").cast(pl.Int32),
    ])
    lgbm = pl.DataFrame(
        {
            "uid": [100, 100, 100, 200, 200],
            "item_id": [1, 2, 3, 4, 5],
            "lgbm_score": [0.1, 0.5, 0.9, 0.2, 0.7],
        }
    ).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("lgbm_score").cast(pl.Float32),
    ])
    out = refit_topk._cascade_cut(feats, lgbm, n=2)
    # Per uid, keep top-2 by lgbm_score:
    # uid=100: items 3 (0.9), 2 (0.5).
    # uid=200: items 5 (0.7), 4 (0.2).
    out_keys = sorted(zip(out["uid"].to_list(), out["item_id"].to_list()))
    assert out_keys == [(100, 2), (100, 3), (200, 4), (200, 5)]
    # lgbm_rank is dense 1..N per uid.
    out_sorted = out.sort(["uid", "lgbm_score"], descending=[False, True])
    ranks = out_sorted["lgbm_rank"].to_list()
    assert ranks == [1, 2, 1, 2]
