"""Repeat-listen candidate generator.

For music recommendations the strongest single signal is "user comes back to a
track they already played". This model surfaces a user's own history ranked
by listen frequency, recency and average completion ratio.
"""
import logging

import polars as pl

from src.data.dataset import positive_listens
from src.models.base import BaseModel

log = logging.getLogger(__name__)

_DEFAULT_HALF_LIFE = 86_400  # 5 days in 5-second timestamp units


class RepeatListenModel(BaseModel):
    """Re-rank a user's own listening history.

    Score per (uid, item_id):
        log1p(n_listens)
            * exp((last_ts - max_train_ts) / half_life_units)
            * (sum_played_ratio / 100)

    The ``sum_played_ratio`` factor rewards items the user actually completes.
    Output columns: ``uid, item_id, score, repeat_rank``.
    """

    def __init__(
        self,
        name: str = "repeat",
        n_cand: int = 200,
        half_life_units: int = _DEFAULT_HALF_LIFE,
    ):
        self.name = name
        self.n_cand = n_cand
        self.half_life_units = half_life_units
        self._scored: pl.DataFrame | None = None

    def fit(self, train: pl.DataFrame, **kwargs) -> None:
        pos = positive_listens(train)
        max_ts = float(pos["timestamp"].max())
        log.info(
            "fitting RepeatListen: rows=%d max_ts=%s half_life=%d",
            len(pos), max_ts, self.half_life_units,
        )

        scored = (
            pos
            .group_by(["uid", "item_id"])
            .agg([
                pl.len().alias("n_listens"),
                pl.col("timestamp").max().alias("last_ts"),
                pl.col("played_ratio_pct").sum().alias("sum_played_ratio"),
            ])
            .with_columns(
                (
                    (pl.col("n_listens").cast(pl.Float64) + 1).log()
                    * (
                        (pl.col("last_ts").cast(pl.Float64) - max_ts)
                        / self.half_life_units
                    ).exp()
                    * (pl.col("sum_played_ratio").cast(pl.Float64) / 100.0)
                )
                .cast(pl.Float32)
                .alias("score")
            )
            .with_columns([
                pl.col("uid").cast(pl.Int64),
                pl.col("item_id").cast(pl.Int64),
            ])
            .select(["uid", "item_id", "score"])
        )
        self._scored = scored
        log.info("RepeatListen fitted: %d (uid, item_id) pairs", len(scored))

    def recommend(self, users: list[int], n: int = 100, **kwargs) -> pl.DataFrame:
        users_df = pl.DataFrame({"uid": users}).with_columns(pl.col("uid").cast(pl.Int64))
        return (
            self._scored
            .join(users_df, on="uid", how="inner")
            .sort(["uid", "score"], descending=[False, True])
            .group_by("uid")
            .head(n)
            .sort(["uid", "score"], descending=[False, True])
            .with_columns(
                pl.int_range(pl.len(), dtype=pl.Int32).over("uid").alias("repeat_rank")
            )
            .with_columns((pl.col("repeat_rank") + 1).alias("repeat_rank"))
            .select(["uid", "item_id", "score", "repeat_rank"])
        )
