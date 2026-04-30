"""Feature builders for the multi-CG ranker pipeline.

All functions accept and return ``pl.LazyFrame`` so the optimiser can fuse
filters/projections and Polars can stream execution. Materialisation
(``.collect(streaming=True)`` or ``.sink_parquet()``) is the responsibility of
the calling script.

Inputs:
    - listens: full listens.parquet (NOT pre-filtered to positive — features
      need ``played_ratio_pct`` distribution incl. ≤50%)
    - likes/dislikes/unlikes/undislikes: full feedback tables
    - artist_map / album_map: item_id → entity_id mappings (may have multiple
      rows per item; we take ``min(entity_id)`` as primary entity to avoid
      row explosion in joins)
    - cutoff_ts: timestamp boundary (5-second units). All features are
      computed over events with ``timestamp <= cutoff_ts``.

Day conversion: 1 day = 17_280 timestamp units (5-second buckets).

For the pair-features we apply a ``semi-join`` between listens and the
candidate (uid, item_id) set BEFORE group-by — this is the key optimisation
that lets the same code scale to 500m / 5B.
"""
from __future__ import annotations

import logging

import polars as pl

log = logging.getLogger(__name__)

ONE_DAY_TS = 17_280  # 5-second units in 24h
DEFAULT_DECAY_HALF_LIFE = 86_400  # ~5 days, matches DecayPop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _primary_entity_map(map_lf: pl.LazyFrame, entity_col: str) -> pl.LazyFrame:
    """Reduce a multi-row item→entity mapping to one entity per item.

    Picks ``min(entity_id)`` deterministically. Avoids row-explosion when we
    later join candidates with the mapping.
    """
    return (
        map_lf
        .group_by("item_id")
        .agg(pl.col(entity_col).min().alias(entity_col))
    )


# ---------------------------------------------------------------------------
# User features
# ---------------------------------------------------------------------------

def build_user_features(
    listens_lf: pl.LazyFrame,
    likes_lf: pl.LazyFrame,
    dislikes_lf: pl.LazyFrame,
    unlikes_lf: pl.LazyFrame,
    artist_map_lf: pl.LazyFrame,
    album_map_lf: pl.LazyFrame,
    cutoff_ts: int,
) -> pl.LazyFrame:
    """User-level features (key: ``uid``).

    See module docstring for full column list. ~17 features.
    """
    cutoff_1d = cutoff_ts - 1 * ONE_DAY_TS
    cutoff_7d = cutoff_ts - 7 * ONE_DAY_TS
    cutoff_30d = cutoff_ts - 30 * ONE_DAY_TS

    L = listens_lf.filter(pl.col("timestamp") <= cutoff_ts)

    user_listen_feats = (
        L
        .group_by("uid")
        .agg([
            pl.len().cast(pl.Int32).alias("user_n_listens"),
            (pl.col("timestamp") >= cutoff_1d).sum().cast(pl.Int32).alias("user_n_listens_1d"),
            (pl.col("timestamp") >= cutoff_7d).sum().cast(pl.Int32).alias("user_n_listens_7d"),
            (pl.col("timestamp") >= cutoff_30d).sum().cast(pl.Int32).alias("user_n_listens_30d"),
            pl.col("item_id").n_unique().cast(pl.Int32).alias("user_n_unique_items"),
            pl.col("played_ratio_pct").mean().cast(pl.Float32).alias("user_avg_played_ratio"),
            ((pl.col("played_ratio_pct") > 99).sum() / pl.len()).cast(pl.Float32).alias("user_share_completed"),
            ((pl.col("played_ratio_pct") <= 50).sum() / pl.len()).cast(pl.Float32).alias("user_share_low_played"),
            ((pl.col("played_ratio_pct") > 100).sum() / pl.len()).cast(pl.Float32).alias("user_share_replayed"),
            pl.col("is_organic").cast(pl.Float32).mean().alias("user_share_organic"),
            pl.col("timestamp").max().alias("_last_ts"),
            pl.col("timestamp").min().alias("_first_ts"),
        ])
        .with_columns([
            ((cutoff_ts - pl.col("_last_ts")).cast(pl.Float32) / ONE_DAY_TS).alias("user_recency_last_listen"),
            (
                pl.col("user_n_listens").cast(pl.Float32)
                / ((cutoff_ts - pl.col("_first_ts")).cast(pl.Float32) / ONE_DAY_TS + 1.0)
            ).cast(pl.Float32).alias("user_avg_listens_per_day"),
        ])
        .drop(["_last_ts", "_first_ts"])
    )

    artist_primary = _primary_entity_map(artist_map_lf, "artist_id")
    album_primary = _primary_entity_map(album_map_lf, "album_id")

    user_unique_entities = (
        L.select(["uid", "item_id"]).unique()
        .join(artist_primary, on="item_id", how="left")
        .join(album_primary, on="item_id", how="left")
        .group_by("uid")
        .agg([
            pl.col("artist_id").n_unique().cast(pl.Int32).alias("user_n_unique_artists"),
            pl.col("album_id").n_unique().cast(pl.Int32).alias("user_n_unique_albums"),
        ])
    )

    likes_count = (
        likes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .group_by("uid").agg(pl.len().cast(pl.Int32).alias("user_n_likes"))
    )
    dislikes_count = (
        dislikes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .group_by("uid").agg(pl.len().cast(pl.Int32).alias("user_n_dislikes"))
    )
    unlikes_count = (
        unlikes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .group_by("uid").agg(pl.len().cast(pl.Int32).alias("user_n_unlikes"))
    )

    return (
        user_listen_feats
        .join(user_unique_entities, on="uid", how="left")
        .join(likes_count, on="uid", how="left")
        .join(dislikes_count, on="uid", how="left")
        .join(unlikes_count, on="uid", how="left")
        .with_columns([
            pl.col("user_n_unique_artists").fill_null(0).cast(pl.Int32),
            pl.col("user_n_unique_albums").fill_null(0).cast(pl.Int32),
            pl.col("user_n_likes").fill_null(0).cast(pl.Int32),
            pl.col("user_n_dislikes").fill_null(0).cast(pl.Int32),
            pl.col("user_n_unlikes").fill_null(0).cast(pl.Int32),
            (
                pl.col("user_n_likes").cast(pl.Float32)
                / (pl.col("user_n_listens").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("user_like_rate"),
            (
                pl.col("user_n_dislikes").cast(pl.Float32)
                / (pl.col("user_n_listens").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("user_dislike_rate"),
            pl.col("uid").cast(pl.Int64),
        ])
    )


# ---------------------------------------------------------------------------
# Item features
# ---------------------------------------------------------------------------

def build_item_features(
    listens_lf: pl.LazyFrame,
    likes_lf: pl.LazyFrame,
    dislikes_lf: pl.LazyFrame,
    cutoff_ts: int,
    decay_half_life_units: int = DEFAULT_DECAY_HALF_LIFE,
) -> pl.LazyFrame:
    """Item-level features (key: ``item_id``). ~14 features."""
    cutoff_7d = cutoff_ts - 7 * ONE_DAY_TS
    cutoff_30d = cutoff_ts - 30 * ONE_DAY_TS

    L = listens_lf.filter(pl.col("timestamp") <= cutoff_ts)

    decay_expr = (
        pl.lit(2.0).pow(
            (pl.col("timestamp").cast(pl.Float64) - cutoff_ts) / decay_half_life_units
        )
    )

    item_listen_feats = (
        L
        .group_by("item_id")
        .agg([
            pl.len().cast(pl.Int32).alias("item_pop"),
            (pl.col("timestamp") >= cutoff_7d).sum().cast(pl.Int32).alias("item_pop_7d"),
            (pl.col("timestamp") >= cutoff_30d).sum().cast(pl.Int32).alias("item_pop_30d"),
            decay_expr.sum().cast(pl.Float32).alias("item_decay_pop"),
            pl.col("uid").n_unique().cast(pl.Int32).alias("item_n_unique_users"),
            pl.col("played_ratio_pct").mean().cast(pl.Float32).alias("item_avg_played_ratio"),
            ((pl.col("played_ratio_pct") > 99).sum() / pl.len()).cast(pl.Float32).alias("item_share_completed"),
            pl.col("is_organic").cast(pl.Float32).mean().alias("item_share_organic"),
            pl.col("track_length_seconds").first().cast(pl.Int32).alias("item_track_length_seconds"),
            ((cutoff_ts - pl.col("timestamp").max()).cast(pl.Float32) / ONE_DAY_TS).alias("item_recency"),
        ])
    )

    likes_count = (
        likes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .group_by("item_id").agg(pl.len().cast(pl.Int32).alias("item_n_likes"))
    )
    dislikes_count = (
        dislikes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .group_by("item_id").agg(pl.len().cast(pl.Int32).alias("item_n_dislikes"))
    )

    return (
        item_listen_feats
        .join(likes_count, on="item_id", how="left")
        .join(dislikes_count, on="item_id", how="left")
        .with_columns([
            pl.col("item_n_likes").fill_null(0).cast(pl.Int32),
            pl.col("item_n_dislikes").fill_null(0).cast(pl.Int32),
            (
                pl.col("item_n_likes").cast(pl.Float32)
                / (pl.col("item_pop").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("item_like_rate"),
            (
                pl.col("item_n_dislikes").cast(pl.Float32)
                / (pl.col("item_pop").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("item_dislike_rate"),
            pl.col("item_id").cast(pl.Int64),
        ])
    )


# ---------------------------------------------------------------------------
# Pair features
# ---------------------------------------------------------------------------

def build_pair_features(
    candidates_lf: pl.LazyFrame,
    listens_lf: pl.LazyFrame,
    likes_lf: pl.LazyFrame,
    unlikes_lf: pl.LazyFrame,
    undislikes_lf: pl.LazyFrame,
    cutoff_ts: int,
) -> pl.LazyFrame:
    """Pair-level features (key: ``(uid, item_id)``). ~12 features.

    KEY OPTIMISATION: semi-join listens with candidates BEFORE group_by.
    On 50m: 30M rows → ~9M; on 5B: 4.65B → ~700M. Without this, group_by
    on the full listens table OOMs.
    """
    cand_pairs = (
        candidates_lf
        .select(["uid", "item_id"])
        .with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
        .unique()
    )

    rel_listens = (
        listens_lf
        .filter(pl.col("timestamp") <= cutoff_ts)
        .with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
        .join(cand_pairs, on=["uid", "item_id"], how="semi")
    )

    pair_listen_aggs = (
        rel_listens
        .group_by(["uid", "item_id"])
        .agg([
            pl.len().cast(pl.Int32).alias("pair_n_listens"),
            pl.col("played_ratio_pct").mean().cast(pl.Float32).alias("pair_avg_played_ratio"),
            pl.col("played_ratio_pct").max().cast(pl.Float32).alias("pair_max_played_ratio"),
            ((pl.col("played_ratio_pct") > 99).sum() / pl.len()).cast(pl.Float32).alias("pair_share_completed"),
            ((pl.col("played_ratio_pct") > 100).sum() / pl.len()).cast(pl.Float32).alias("pair_share_replayed"),
            (pl.col("played_ratio_pct") <= 50).sum().cast(pl.Int32).alias("pair_n_low_played"),
            ((cutoff_ts - pl.col("timestamp").max()).cast(pl.Float32) / ONE_DAY_TS).alias("pair_days_since_last_listen"),
            ((cutoff_ts - pl.col("timestamp").min()).cast(pl.Float32) / ONE_DAY_TS).alias("pair_days_since_first_listen"),
            pl.col("is_organic").cast(pl.Float32).mean().alias("pair_share_organic"),
        ])
    )

    # Effective like = like exists AND no later unlike for the same pair.
    likes_per_pair = (
        likes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
        .group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").max().alias("_last_like_ts"))
    )
    unlikes_per_pair = (
        unlikes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
        .group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").max().alias("_last_unlike_ts"))
    )
    pair_is_liked = (
        likes_per_pair
        .join(unlikes_per_pair, on=["uid", "item_id"], how="left")
        .with_columns(
            (
                pl.col("_last_unlike_ts").is_null()
                | (pl.col("_last_like_ts") > pl.col("_last_unlike_ts"))
            ).cast(pl.Int8).alias("pair_is_liked")
        )
        .select(["uid", "item_id", "pair_is_liked"])
    )

    pair_is_unliked = (
        unlikes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .select(["uid", "item_id"])
        .with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("pair_is_unliked"))
    )

    pair_is_undisliked = (
        undislikes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .select(["uid", "item_id"])
        .with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("pair_is_undisliked"))
    )

    return (
        cand_pairs
        .join(pair_listen_aggs, on=["uid", "item_id"], how="left")
        .join(pair_is_liked, on=["uid", "item_id"], how="left")
        .join(pair_is_unliked, on=["uid", "item_id"], how="left")
        .join(pair_is_undisliked, on=["uid", "item_id"], how="left")
        .with_columns([
            pl.col("pair_n_listens").fill_null(0).cast(pl.Int32),
            pl.col("pair_n_low_played").fill_null(0).cast(pl.Int32),
            pl.col("pair_is_liked").fill_null(0).cast(pl.Int8),
            pl.col("pair_is_unliked").fill_null(0).cast(pl.Int8),
            pl.col("pair_is_undisliked").fill_null(0).cast(pl.Int8),
        ])
    )


# ---------------------------------------------------------------------------
# Cross artist/album features
# ---------------------------------------------------------------------------

def build_cross_features(
    candidates_lf: pl.LazyFrame,
    listens_lf: pl.LazyFrame,
    artist_map_lf: pl.LazyFrame,
    album_map_lf: pl.LazyFrame,
    cutoff_ts: int,
) -> pl.LazyFrame:
    """Cross artist/album features (key: ``(uid, item_id)``). ~5 features.

    Uses primary artist/album per item (``min(entity_id)``) to keep joins
    explosion-free.
    """
    L = (
        listens_lf
        .filter(pl.col("timestamp") <= cutoff_ts)
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
    )
    artist_primary = _primary_entity_map(artist_map_lf, "artist_id").with_columns(
        pl.col("item_id").cast(pl.Int64)
    )
    album_primary = _primary_entity_map(album_map_lf, "album_id").with_columns(
        pl.col("item_id").cast(pl.Int64)
    )

    L_with_entities = (
        L.join(artist_primary, on="item_id", how="left")
        .join(album_primary, on="item_id", how="left")
    )

    user_artist = (
        L_with_entities
        .group_by(["uid", "artist_id"])
        .agg(pl.len().cast(pl.Int32).alias("user_artist_listens"))
    )
    user_album = (
        L_with_entities
        .group_by(["uid", "album_id"])
        .agg(pl.len().cast(pl.Int32).alias("user_album_listens"))
    )
    user_total = (
        L.group_by("uid")
        .agg(pl.len().cast(pl.Int32).alias("_user_total_listens"))
    )
    artist_pop = (
        L_with_entities
        .group_by("artist_id")
        .agg(pl.len().cast(pl.Int32).alias("artist_pop"))
    )
    album_pop = (
        L_with_entities
        .group_by("album_id")
        .agg(pl.len().cast(pl.Int32).alias("album_pop"))
    )

    cand = (
        candidates_lf
        .select(["uid", "item_id"])
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
        .unique()
    )

    return (
        cand
        .join(artist_primary, on="item_id", how="left")
        .join(album_primary, on="item_id", how="left")
        .join(user_artist, on=["uid", "artist_id"], how="left")
        .join(user_album, on=["uid", "album_id"], how="left")
        .join(artist_pop, on="artist_id", how="left")
        .join(album_pop, on="album_id", how="left")
        .join(user_total, on="uid", how="left")
        .with_columns([
            pl.col("user_artist_listens").fill_null(0).cast(pl.Int32),
            pl.col("user_album_listens").fill_null(0).cast(pl.Int32),
            pl.col("artist_pop").fill_null(0).cast(pl.Int32),
            pl.col("album_pop").fill_null(0).cast(pl.Int32),
            (
                pl.col("user_artist_listens").cast(pl.Float32)
                / (pl.col("_user_total_listens").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("user_artist_share"),
        ])
        .drop(["artist_id", "album_id", "_user_total_listens"])
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def add_features(
    candidates_lf: pl.LazyFrame,
    listens_lf: pl.LazyFrame,
    likes_lf: pl.LazyFrame,
    dislikes_lf: pl.LazyFrame,
    unlikes_lf: pl.LazyFrame,
    undislikes_lf: pl.LazyFrame,
    artist_map_lf: pl.LazyFrame,
    album_map_lf: pl.LazyFrame,
    cutoff_ts: int,
    decay_half_life_units: int = DEFAULT_DECAY_HALF_LIFE,
) -> pl.LazyFrame:
    """Enrich candidates with user / item / pair / cross features.

    The result preserves the original ``candidates_lf`` rows (left-joins
    everywhere) plus all feature columns. Caller decides on materialisation:

        features_lf = add_features(...)
        df = features_lf.collect(streaming=True)
        # OR
        features_lf.sink_parquet("cache.parquet", compression="zstd")
    """
    user_feats = build_user_features(
        listens_lf, likes_lf, dislikes_lf, unlikes_lf,
        artist_map_lf, album_map_lf, cutoff_ts,
    )
    item_feats = build_item_features(
        listens_lf, likes_lf, dislikes_lf, cutoff_ts, decay_half_life_units,
    )
    pair_feats = build_pair_features(
        candidates_lf, listens_lf, likes_lf, unlikes_lf, undislikes_lf, cutoff_ts,
    )
    cross_feats = build_cross_features(
        candidates_lf, listens_lf, artist_map_lf, album_map_lf, cutoff_ts,
    )

    enriched_cands = (
        candidates_lf
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
        .join(user_feats, on="uid", how="left")
        .join(item_feats, on="item_id", how="left")
        .join(pair_feats, on=["uid", "item_id"], how="left")
        .join(cross_feats, on=["uid", "item_id"], how="left")
    )
    return enriched_cands


def load_features_from_cache(path: str) -> pl.LazyFrame:
    """Lazy scan of a previously-cached features parquet."""
    return pl.scan_parquet(path)
