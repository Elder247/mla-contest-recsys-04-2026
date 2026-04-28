import polars as pl
import pytest

from src.evaluation.metrics import recall_at_k


def _make_true(pairs: list[tuple[int, int]]) -> pl.DataFrame:
    uids, items = zip(*pairs)
    return pl.DataFrame({"uid": list(uids), "item_id": list(items)}).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
    ])


def _make_pred(rows: list[tuple[int, int, float]]) -> pl.DataFrame:
    uids, items, scores = zip(*rows)
    return pl.DataFrame({
        "uid": list(uids),
        "item_id": list(items),
        "score": list(scores),
    }).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("score").cast(pl.Float64),
    ])


def test_perfect_recall():
    true = _make_true([(1, 10), (1, 20), (2, 30)])
    pred = _make_pred([(1, 10, 1.0), (1, 20, 0.9), (2, 30, 1.0)])
    assert recall_at_k(true, pred, k=100) == pytest.approx(1000.0)


def test_zero_recall():
    true = _make_true([(1, 10), (2, 20)])
    pred = _make_pred([(1, 99, 1.0), (2, 88, 1.0)])
    assert recall_at_k(true, pred, k=100) == pytest.approx(0.0)


def test_partial_recall():
    # user 1: 1 hit out of 2 ground truth → 0.5; user 2: 0 hits → 0.0; avg = 0.25 * 1000 = 250
    true = _make_true([(1, 10), (1, 20), (2, 30)])
    pred = _make_pred([(1, 10, 1.0), (1, 99, 0.5), (2, 88, 1.0)])
    assert recall_at_k(true, pred, k=100) == pytest.approx(250.0)


def test_k_limits_denominator():
    # k=1 → denom = min(1, |G_u|=3) = 1; predict the right item → recall = 1.0 * 1000
    true = _make_true([(1, 10), (1, 20), (1, 30)])
    pred = _make_pred([(1, 10, 1.0)])
    assert recall_at_k(true, pred, k=1) == pytest.approx(1000.0)


def test_scores_select_top_k():
    # k=1: only highest-score prediction used; it's a miss
    true = _make_true([(1, 10)])
    pred = _make_pred([(1, 10, 0.5), (1, 99, 1.0)])
    assert recall_at_k(true, pred, k=1) == pytest.approx(0.0)


def test_duplicate_pred_raises():
    true = _make_true([(1, 10)])
    pred = _make_pred([(1, 10, 1.0), (1, 10, 0.9)])
    with pytest.raises(AssertionError):
        recall_at_k(true, pred, k=100)
