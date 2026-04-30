"""Recent-likes candidate generator.

Likes are a much stronger and rarer signal than listens — a user explicitly
chose to mark a track. ``RecentLikesModel`` surfaces a user's own liked tracks,
ranked by recency only. Pair this with `RepeatListenModel` for "stuff the user
already engaged with".

Note: ``fit`` expects a likes DataFrame (columns: ``uid, item_id, timestamp``),
NOT listens. Routing is handled by the pipeline scripts via the
``data_source: likes`` field in the candidate-generator config.
"""
import logging

import polars as pl

from src.models.base import BaseModel

log = logging.getLogger(__name__)

_DEFAULT_HALF_LIFE = 86_400  # 5 days in 5-second timestamp units (matches DecayPop)


class RecentLikesModel(BaseModel):
    """Per-user top-N liked tracks ranked by ``exp((like_ts - max_ts) / tau)``.

    Output columns: ``uid, item_id, score, recent_likes_rank``.
    """

    def __init__(
        self,
        name: str = "recent_likes",
        n_cand: int = 100,
        half_life_units: int = _DEFAULT_HALF_LIFE,
    ):
        self.name = name
        self.n_cand = n_cand
        self.half_life_units = half_life_units
        self._scored: pl.DataFrame | None = None

    def fit(self, likes: pl.DataFrame, **kwargs) -> None:
        if "timestamp" not in likes.columns:
            raise ValueError(
                f"RecentLikesModel.fit expects a likes-like DataFrame "
                f"with 'timestamp', got columns: {likes.columns}"
            )

        max_ts = float(likes["timestamp"].max())
        log.info(
            "fitting RecentLikes: rows=%d max_ts=%s half_life=%d",
            len(likes), max_ts, self.half_life_units,
        )

        scored = (
            likes
            .group_by(["uid", "item_id"])
            .agg(pl.col("timestamp").max().alias("last_like_ts"))
            .with_columns(
                (
                    (pl.col("last_like_ts").cast(pl.Float64) - max_ts)
                    / self.half_life_units
                )
                .exp()
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
        log.info("RecentLikes fitted: %d (uid, item_id) pairs", len(scored))

    def recommend(self, users: list[int], n: int = 100, **kwargs) -> pl.DataFrame:
        users_df = pl.DataFrame({"uid": users}).with_columns(pl.col("uid").cast(pl.Int64))
        rank_col = f"{self.name}_rank"
        return (
            self._scored
            .join(users_df, on="uid", how="inner")
            .sort(["uid", "score"], descending=[False, True])
            .group_by("uid", maintain_order=True)
            .head(n)
            .with_columns(
                pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over("uid").alias(rank_col)
            )
            .select(["uid", "item_id", "score", rank_col])
        )
