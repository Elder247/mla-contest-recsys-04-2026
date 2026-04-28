"""Popularity-based candidate generators.

DecayPop   — global top-n with exponential time decay (matches notebook baseline).
UserPersonalPop — same pool but filtered to exclude user's own history.
"""
import logging

import polars as pl

from src.models.base import BaseModel

log = logging.getLogger(__name__)

# Denominator for decay: matches notebook (86400 = ~5-day half-life in real time
# because timestamps are in 5-second units, so 86400 units = 86400*5 = 432000 sec = 5 days)
_DEFAULT_HALF_LIFE = 86_400


class DecayPop(BaseModel):
    """Global popularity ranking with exponential decay. Same list for all users."""

    def __init__(self, name: str = "pop", half_life_units: int = _DEFAULT_HALF_LIFE):
        self.name = name
        self.half_life_units = half_life_units
        self._item_scores: pl.DataFrame | None = None

    def fit(self, train: pl.DataFrame, **kwargs) -> None:
        max_ts = float(train["timestamp"].max())
        log.info("fitting DecayPop: max_ts=%s, half_life=%d", max_ts, self.half_life_units)
        self._item_scores = (
            train
            .group_by("item_id")
            .agg(
                pl.lit(2)
                .pow(
                    (pl.col("timestamp").cast(pl.Float64) - max_ts) / self.half_life_units
                )
                .sum()
                .alias("score")
            )
            .sort("score", descending=True)
            .with_columns([
                pl.col("item_id").cast(pl.Int64),
                pl.col("score").cast(pl.Float64),
            ])
        )
        log.info("DecayPop fitted: %d unique items", len(self._item_scores))

    def recommend(self, users: list[int], n: int = 100, **kwargs) -> pl.DataFrame:
        top = self._item_scores.head(n)
        users_df = pl.DataFrame({"uid": users}).with_columns(pl.col("uid").cast(pl.Int64))
        return (
            users_df
            .join(top, how="cross")
            .select(["uid", "item_id", "score"])
        )


class UserPersonalPop(BaseModel):
    """Popular items filtered to exclude each user's own listening history."""

    def __init__(
        self,
        name: str = "user_pop",
        half_life_units: int = _DEFAULT_HALF_LIFE,
        candidates_pool: int = 500,
    ):
        self.name = name
        self.half_life_units = half_life_units
        self.candidates_pool = candidates_pool
        self._item_scores: pl.DataFrame | None = None
        self._seen_df: pl.DataFrame | None = None

    def fit(self, train: pl.DataFrame, **kwargs) -> None:
        max_ts = float(train["timestamp"].max())
        log.info("fitting UserPersonalPop: pool=%d", self.candidates_pool)
        self._item_scores = (
            train
            .group_by("item_id")
            .agg(
                pl.lit(2)
                .pow(
                    (pl.col("timestamp").cast(pl.Float64) - max_ts) / self.half_life_units
                )
                .sum()
                .alias("score")
            )
            .sort("score", descending=True)
            .with_columns([
                pl.col("item_id").cast(pl.Int64),
                pl.col("score").cast(pl.Float64),
            ])
        )
        self._seen_df = (
            train
            .select(["uid", "item_id"])
            .unique()
            .with_columns([
                pl.col("uid").cast(pl.Int64),
                pl.col("item_id").cast(pl.Int64),
            ])
        )
        log.info("UserPersonalPop fitted: %d items, %d seen pairs",
                 len(self._item_scores), len(self._seen_df))

    def recommend(self, users: list[int], n: int = 100, **kwargs) -> pl.DataFrame:
        pool = self._item_scores.head(self.candidates_pool)
        users_df = pl.DataFrame({"uid": users}).with_columns(pl.col("uid").cast(pl.Int64))
        return (
            users_df
            .join(pool, how="cross")
            .join(self._seen_df, on=["uid", "item_id"], how="anti")
            .sort(["uid", "score"], descending=[False, True])
            .group_by("uid")
            .head(n)
            .select(["uid", "item_id", "score"])
        )
