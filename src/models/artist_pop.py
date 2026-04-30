"""Artist/Album-aware popularity candidate generator.

Music-specific: take a user's top-K favourite artists (or albums) by
decay-weighted listen affinity, then surface those entities' top tracks.

Score = user_entity_affinity × entity_track_score.

The same class serves both ``artist`` and ``album`` granularity via the
``entity`` parameter — different yamls (``artist_pop.yaml``, ``album_pop.yaml``)
instantiate it with different names + entity values.
"""
import logging
from typing import Literal

import polars as pl

from src.data.dataset import (
    load_album_item_mapping,
    load_artist_item_mapping,
    positive_listens,
)
from src.models.base import BaseModel

log = logging.getLogger(__name__)

_DEFAULT_HALF_LIFE = 86_400


class ArtistAlbumPopModel(BaseModel):
    """Personalised entity-aware popularity (artist or album).

    Output columns: ``uid, item_id, score, {name}_rank``.
    """

    def __init__(
        self,
        name: str = "artist_pop",
        entity: Literal["artist", "album"] = "artist",
        top_entities: int = 10,
        n_cand: int = 100,
        half_life_units: int = _DEFAULT_HALF_LIFE,
    ):
        if entity not in ("artist", "album"):
            raise ValueError(f"entity must be 'artist' or 'album', got {entity!r}")
        self.name = name
        self.entity = entity
        self.top_entities = top_entities
        self.n_cand = n_cand
        self.half_life_units = half_life_units
        self._user_entity: pl.DataFrame | None = None
        self._entity_track: pl.DataFrame | None = None
        self._entity_col = f"{entity}_id"

    def fit(self, train: pl.DataFrame, **kwargs) -> None:
        pos = positive_listens(train)
        max_ts = float(pos["timestamp"].max())

        # decay-weighted listen score per (uid, item_id)
        decay = (
            (pl.col("timestamp").cast(pl.Float64) - max_ts) / self.half_life_units
        ).exp()

        user_item_score = (
            pos
            .with_columns(decay.alias("decay"))
            .group_by(["uid", "item_id"])
            .agg(pl.col("decay").sum().alias("listen_score"))
            .with_columns([
                pl.col("uid").cast(pl.Int64),
                pl.col("item_id").cast(pl.Int64),
                pl.col("listen_score").cast(pl.Float32),
            ])
        )

        if self.entity == "artist":
            mapping = load_artist_item_mapping()
        else:
            mapping = load_album_item_mapping()
        mapping = mapping.with_columns([
            pl.col(self._entity_col).cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])

        log.info(
            "fitting %s: users=%d items=%d entities=%d half_life=%d",
            self.name,
            user_item_score["uid"].n_unique(),
            user_item_score["item_id"].n_unique(),
            mapping[self._entity_col].n_unique(),
            self.half_life_units,
        )

        user_entity_item = user_item_score.join(mapping, on="item_id", how="inner")

        # User → entity affinity (sum of listen-scores over the entity's tracks)
        self._user_entity = (
            user_entity_item
            .group_by(["uid", self._entity_col])
            .agg(pl.col("listen_score").sum().alias("affinity"))
            .with_columns(pl.col("affinity").cast(pl.Float32))
        )

        # Entity → track popularity (decay-weighted across all users)
        self._entity_track = (
            user_entity_item
            .group_by([self._entity_col, "item_id"])
            .agg(pl.col("listen_score").sum().alias("track_score"))
            .with_columns(pl.col("track_score").cast(pl.Float32))
        )

        log.info(
            "%s fitted: %d (uid, %s) affinities, %d (%s, item_id) tracks",
            self.name,
            len(self._user_entity), self._entity_col,
            len(self._entity_track), self._entity_col,
        )

    def recommend(self, users: list[int], n: int = 100, **kwargs) -> pl.DataFrame:
        rank_col = f"{self.name}_rank"
        users_df = pl.DataFrame({"uid": users}).with_columns(pl.col("uid").cast(pl.Int64))

        # Top-K entities per user
        top_user_entities = (
            self._user_entity
            .join(users_df, on="uid", how="inner")
            .sort(["uid", "affinity"], descending=[False, True])
            .group_by("uid", maintain_order=True)
            .head(self.top_entities)
        )

        # Cross with their tracks
        return (
            top_user_entities
            .join(self._entity_track, on=self._entity_col, how="inner")
            .with_columns(
                (pl.col("affinity") * pl.col("track_score"))
                .cast(pl.Float32)
                .alias("score")
            )
            # if the same item is reachable via multiple top entities,
            # keep the strongest path
            .group_by(["uid", "item_id"], maintain_order=False)
            .agg(pl.col("score").max())
            .sort(["uid", "score"], descending=[False, True])
            .group_by("uid", maintain_order=True)
            .head(n)
            .with_columns(
                pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over("uid").alias(rank_col)
            )
            .select(["uid", "item_id", "score", rank_col])
        )
