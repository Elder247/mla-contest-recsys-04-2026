import polars as pl
import pytest

from src.data.splits import temporal_split

UPD = 17_280  # 1 day in 5-second units


def _make_df(timestamps: list[int]) -> pl.DataFrame:
    n = len(timestamps)
    return pl.DataFrame({
        "uid": list(range(n)),
        "item_id": list(range(n)),
        "timestamp": timestamps,
    })


def test_split_sizes():
    # 30 days of data, val_days=7, gap_days=1
    # train < day 22;  val [22, 29);  test [23, 30)
    n_days = 30
    ts = list(range(0, n_days * UPD, UPD))
    df = _make_df(ts)
    split = temporal_split(df, val_days=7, gap_days=1)

    assert len(split.train) + len(split.val) + len(split.test) > 0
    assert len(split.train) > 0
    assert len(split.val) > 0
    assert len(split.test) > 0


def test_train_before_val():
    ts = list(range(0, 30 * UPD, UPD))
    df = _make_df(ts)
    split = temporal_split(df, val_days=7, gap_days=1)

    max_train = split.train["timestamp"].max()
    min_val = split.val["timestamp"].min()
    assert max_train < min_val


def test_val_test_overlap():
    # val and test windows intentionally overlap (public/private leaderboard simulation)
    ts = list(range(0, 30 * UPD, UPD))
    df = _make_df(ts)
    split = temporal_split(df, val_days=7, gap_days=1)

    val_ts = set(split.val["timestamp"].to_list())
    test_ts = set(split.test["timestamp"].to_list())
    assert len(val_ts & test_ts) > 0, "val and test should overlap"


def test_no_future_in_train():
    ts = list(range(0, 20 * UPD, UPD))
    df = _make_df(ts)
    split = temporal_split(df, val_days=7, gap_days=1)

    t_max_train = split.train["timestamp"].max()
    t_min_val = split.val["timestamp"].min()
    assert t_max_train < t_min_val


def test_deterministic():
    ts = list(range(0, 20 * UPD, UPD))
    df = _make_df(ts)
    s1 = temporal_split(df, val_days=7, gap_days=1)
    s2 = temporal_split(df, val_days=7, gap_days=1)
    assert s1.train.shape == s2.train.shape
    assert s1.val.shape == s2.val.shape
    assert s1.test.shape == s2.test.shape
