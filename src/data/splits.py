import polars as pl
from dataclasses import dataclass


@dataclass
class Split:
    train: pl.DataFrame
    val: pl.DataFrame   # simulates public leaderboard
    test: pl.DataFrame  # simulates private leaderboard (shifted +gap_days)


def temporal_split(
    df: pl.DataFrame,
    val_days: int = 7,
    gap_days: int = 1,
    timestamp_col: str = "timestamp",
) -> Split:
    """
    Temporal split mirroring the contest structure.
    Timestamps are in 5-second units (1 day = 17280 units).
    """
    upd = 86400 // 5  # units per day = 17280

    t_max = df[timestamp_col].max()
    t_end = t_max + 1  # exclusive upper bound to include t_max in test

    val_start  = t_end - (val_days + gap_days) * upd
    test_start = val_start + gap_days * upd        # = t_end - val_days * upd
    val_end    = val_start + val_days * upd        # = t_end - gap_days * upd

    train = df.filter(pl.col(timestamp_col) < val_start)
    val   = df.filter((pl.col(timestamp_col) >= val_start)  & (pl.col(timestamp_col) < val_end))
    test  = df.filter((pl.col(timestamp_col) >= test_start) & (pl.col(timestamp_col) < t_end))

    return Split(train=train, val=val, test=test)
