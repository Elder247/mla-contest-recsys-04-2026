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

import gc
import logging

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)

ONE_DAY_TS = 17_280  # 5-second units in 24h
DEFAULT_DECAY_HALF_LIFE = 518_400  # ~30 days, empirical half-life from EDA (notebooks/01_eda.ipynb §time-decay)


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
            pl.col("user_n_listens").cast(pl.Float32).log1p().cast(pl.Float32).alias("user_n_listens_log"),
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
    decay_half_life_units: int = DEFAULT_DECAY_HALF_LIFE,
) -> pl.LazyFrame:
    """Pair-level features (key: ``(uid, item_id)``). ~20 features.

    KEY OPTIMISATION: semi-join listens with candidates BEFORE group_by.
    On 50m: 30M rows → ~9M; on 5B: 4.65B → ~700M. Without this, group_by
    on the full listens table OOMs.
    """
    cutoff_7d = cutoff_ts - 7 * ONE_DAY_TS
    cutoff_30d = cutoff_ts - 30 * ONE_DAY_TS
    cutoff_90d = cutoff_ts - 90 * ONE_DAY_TS

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

    decay_expr = (
        pl.lit(2.0).pow(
            (pl.col("timestamp").cast(pl.Float64) - cutoff_ts) / decay_half_life_units
        )
    )
    played_seconds_expr = (
        pl.col("played_ratio_pct").cast(pl.Float32)
        * pl.col("track_length_seconds").cast(pl.Float32)
        / 100.0
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
            decay_expr.sum().cast(pl.Float32).alias("pair_decay_listens"),
            (pl.col("timestamp") >= cutoff_7d).sum().cast(pl.Int32).alias("_pair_n_listens_7d"),
            (pl.col("timestamp") >= cutoff_30d).sum().cast(pl.Int32).alias("pair_n_listens_30d"),
            (pl.col("timestamp") >= cutoff_90d).sum().cast(pl.Int32).alias("pair_n_listens_90d"),
            pl.col("played_ratio_pct").filter(pl.col("timestamp") >= cutoff_30d)
                .mean().cast(pl.Float32).alias("pair_avg_played_ratio_30d"),
            pl.col("played_ratio_pct").filter(pl.col("timestamp") >= cutoff_30d)
                .max().cast(pl.Float32).alias("pair_max_played_ratio_30d"),
            played_seconds_expr.mean().cast(pl.Float32).alias("pair_avg_played_seconds"),
        ])
        .with_columns([
            (
                pl.col("pair_n_listens").cast(pl.Float32)
                / (pl.col("pair_days_since_first_listen").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("pair_listens_per_day"),
            (
                pl.col("_pair_n_listens_7d").cast(pl.Float32)
                / (pl.col("pair_n_listens").cast(pl.Float32) + 1e-6)
            ).cast(pl.Float32).alias("pair_share_recent_listens"),
        ])
        .drop("_pair_n_listens_7d")
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

    Memory note: the ``user_artist`` / ``user_album`` / ``user_total`` group_bys
    only matter for candidate users (10K eval users). Restricting listens to
    that user set BEFORE the group_by drops the heavy intermediate from
    500M rows to ~10M on 500m — the same ``semi-join`` trick used in
    ``build_pair_features``. ``artist_pop`` / ``album_pop`` stay global since
    they aggregate by entity_id only (small result, cheap intermediate).
    """
    cand_uids = (
        candidates_lf
        .select(pl.col("uid").cast(pl.Int64))
        .unique()
    )

    L_full = (
        listens_lf
        .filter(pl.col("timestamp") <= cutoff_ts)
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
    )
    L_user = L_full.join(cand_uids, on="uid", how="semi")

    artist_primary = _primary_entity_map(artist_map_lf, "artist_id").with_columns(
        pl.col("item_id").cast(pl.Int64)
    )
    album_primary = _primary_entity_map(album_map_lf, "album_id").with_columns(
        pl.col("item_id").cast(pl.Int64)
    )

    L_user_entities = (
        L_user.join(artist_primary, on="item_id", how="left")
        .join(album_primary, on="item_id", how="left")
    )
    L_full_entities = (
        L_full.join(artist_primary, on="item_id", how="left")
        .join(album_primary, on="item_id", how="left")
    )

    user_artist = (
        L_user_entities
        .group_by(["uid", "artist_id"])
        .agg(pl.len().cast(pl.Int32).alias("user_artist_listens"))
    )
    user_album = (
        L_user_entities
        .group_by(["uid", "album_id"])
        .agg(pl.len().cast(pl.Int32).alias("user_album_listens"))
    )
    user_total = (
        L_user.group_by("uid")
        .agg(pl.len().cast(pl.Int32).alias("_user_total_listens"))
    )
    artist_pop = (
        L_full_entities
        .group_by("artist_id")
        .agg(pl.len().cast(pl.Int32).alias("artist_pop"))
    )
    album_pop = (
        L_full_entities
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
# Audio-embedding ranker features (Phase C.4 / D4)
# ---------------------------------------------------------------------------

def build_embed_features(
    candidates_lf: pl.LazyFrame,
    listens_lf: pl.LazyFrame,
    likes_lf: pl.LazyFrame,
    unlikes_lf: pl.LazyFrame,
    dislikes_lf: pl.LazyFrame,
    undislikes_lf: pl.LazyFrame,
    embeddings_path: str,
    cutoff_ts: int,
    last_k_list: list[int] | None = None,
) -> pl.LazyFrame:
    """Per-candidate audio-embedding cosine features (key: ``(uid, item_id)``).

    Ranker features — orthogonal to the AudioEmbedKNN CG (which does
    *retrieval*; these features give CatBoost a per-candidate similarity
    score it can weigh against other signals):

      * ``embed_cos_user_mean``        — vs L2-normed mean of all user listens
      * ``embed_cos_user_last_{K}``    — vs mean of last K listens (one per K
                                          in ``last_k_list``; default 5/20/50/100)
      * ``embed_cos_user_liked_mean``  — vs mean of effective likes (likes − unlikes)
      * ``embed_cos_user_disliked_mean`` — vs mean of effective dislikes (negative signal)

    Multi-window ``last_K`` cosines capture user intent at different time
    scales: K=5 = current session, K=20 ≈ today, K=50 ≈ this week, K=100 ≈
    this month. Each K is computed by re-aggregating the **same**
    pre-loaded listen pairs — no extra parquet scans.

    Implementation: materialises only the rows / items needed (semi-join on
    candidate items + per-user history items), loads embeddings restricted
    to that set, builds the user-mean vectors via numpy scatter-add, then
    rows-row dot product with the item embedding for each candidate.

    Items / users with no embedding coverage emit NULL — CatBoost handles
    that via ``nan_mode="Min"``. Cosine = inner-product because embeddings
    are L2-normalised in the parquet (``normalized_embed``).

    Returns a LazyFrame keyed by ``(uid, item_id)`` with ``2 + 1 + 1 + len(last_k_list)``
    Float32 cosine columns, suitable for left-join in :func:`add_features`.
    """
    if last_k_list is None:
        last_k_list = [5, 20, 50, 100]
    # Stable de-dup + sort to keep column order deterministic.
    last_k_list = sorted({int(k) for k in last_k_list})
    if not last_k_list or any(k <= 0 for k in last_k_list):
        raise ValueError(
            f"last_k_list must be non-empty positive ints; got {last_k_list}"
        )
    cand_pairs_eager = (
        candidates_lf
        .select(["uid", "item_id"])
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
        .unique()
        .collect()
    )
    log.info("build_embed_features: %d candidate pairs", len(cand_pairs_eager))

    # Only candidate users matter for user-mean embeddings — filter all history
    # LazyFrames early so we don't scan the full 500m/5b event tables.
    cand_uid_list = cand_pairs_eager["uid"].unique().to_list()

    # ── Determine which items we need embeddings for ────────────────────────
    # We need:
    #   (a) every candidate item (to look up the candidate's own embedding)
    #   (b) every item that appears in any *candidate* user's relevant history
    cand_items = cand_pairs_eager["item_id"].unique()

    L_pos = (
        listens_lf
        .filter(pl.col("uid").is_in(cand_uid_list))
        .filter(pl.col("timestamp") <= cutoff_ts)
        .filter(pl.col("played_ratio_pct") > 50)
        .select(["uid", "item_id", "timestamp"])
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
    )
    listen_items = L_pos.select("item_id").unique().collect()["item_id"]

    likes_active = (
        _effective_likes_lf(likes_lf, unlikes_lf, cutoff_ts)
        .filter(pl.col("uid").is_in(cand_uid_list))
    )
    like_items = likes_active.select("item_id").unique().collect()["item_id"]

    dislikes_active = (
        _effective_dislikes_lf(dislikes_lf, undislikes_lf, cutoff_ts)
        .filter(pl.col("uid").is_in(cand_uid_list))
    )
    dislike_items = dislikes_active.select("item_id").unique().collect()["item_id"]

    items_needed = (
        pl.concat([cand_items, listen_items, like_items, dislike_items])
        .unique()
        .to_list()
    )
    log.info(
        "build_embed_features: %d items need embeddings (cands=%d, listens=%d, likes=%d, dislikes=%d)",
        len(items_needed), len(cand_items), len(listen_items),
        len(like_items), len(dislike_items),
    )

    # ── Load embeddings ─────────────────────────────────────────────────────
    emb_df = (
        pl.scan_parquet(embeddings_path)
        .filter(pl.col("item_id").is_in(items_needed))
        .select(["item_id", "normalized_embed"])
        .collect()
    )
    if len(emb_df) == 0:
        log.warning("build_embed_features: no embeddings matched — returning NULL features")
        return _empty_embed_frame(cand_pairs_eager, last_k_list).lazy()

    # .to_list() on a List(Float64) column creates a Python list-of-lists (~8 GB peak
    # for 1.8M×128 items). explode() → 1-D Float32 series → zero-copy numpy → reshape
    # stays under 1 GB.
    n_emb = len(emb_df)
    emb_arr = (
        emb_df["normalized_embed"]
        .explode()
        .cast(pl.Float32)
        .to_numpy()
        .reshape(n_emb, -1)
    )
    emb_item_ids = emb_df["item_id"].cast(pl.Int64).to_numpy()
    del emb_df  # Float64 list column no longer needed (~700 MB freed)
    item_to_row: dict[int, int] = {int(i): k for k, i in enumerate(emb_item_ids.tolist())}
    dim = emb_arr.shape[1]
    log.info("build_embed_features: embedding matrix %s", emb_arr.shape)

    # ── Build user mean vectors (4 variants) ─────────────────────────────────
    # All means produce arrays of shape (n_users, dim) indexed by ``uids_with_*``.
    # Drop each per-history eager DF as soon as its mean(s) are computed —
    # on 500m these are multi-GB and otherwise live until function return.
    listens_user_item = (
        L_pos
        .filter(pl.col("item_id").is_in(items_needed))
        .select(["uid", "item_id", "timestamp"])
        .collect()
    )
    mean_all = _user_mean_embeddings(listens_user_item, item_to_row, emb_arr, dim)
    # One user-mean vector per window K — same pre-loaded listens, just a
    # different per-user .head(K) slice. Cheap relative to the embedding load.
    mean_lastk_per_k: dict[int, tuple[dict[int, int], np.ndarray]] = {
        k: _user_mean_last_k(listens_user_item, k, item_to_row, emb_arr, dim)
        for k in last_k_list
    }
    del listens_user_item
    gc.collect()

    likes_user_item = (
        likes_active
        .filter(pl.col("item_id").is_in(items_needed))
        .select(["uid", "item_id"])
        .collect()
    )
    mean_liked = _user_mean_embeddings(likes_user_item, item_to_row, emb_arr, dim)
    del likes_user_item
    gc.collect()

    dislikes_user_item = (
        dislikes_active
        .filter(pl.col("item_id").is_in(items_needed))
        .select(["uid", "item_id"])
        .collect()
    )
    mean_disliked = _user_mean_embeddings(dislikes_user_item, item_to_row, emb_arr, dim)
    del dislikes_user_item
    gc.collect()

    # ── Compute 4 cosines per candidate ──────────────────────────────────────
    # Old approach: emb_arr[ir] where |ir|=5M creates a 5M×128 float32 copy
    # (2.56 GB) called 4 times.  New approach: sort candidates by uid once,
    # then per-user matrix-vector multiply (max ~560×128 at a time → <1 MB).
    uid_arr = cand_pairs_eager["uid"].cast(pl.Int64).to_numpy()
    iid_arr = cand_pairs_eager["item_id"].cast(pl.Int64).to_numpy()

    # Vectorised item_id → emb_arr row index (searchsorted, -1 if missing)
    emb_sorted_pos = np.argsort(emb_item_ids)
    emb_sorted_ids = emb_item_ids[emb_sorted_pos]

    def _item_rows(ids: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(emb_sorted_ids, ids)
        pos_c = np.minimum(pos, len(emb_sorted_ids) - 1)
        return np.where(emb_sorted_ids[pos_c] == ids, emb_sorted_pos[pos_c], np.int64(-1))

    item_rows_all = _item_rows(iid_arr)

    # Sort candidates by uid once; all four _cos_sorted calls share the result.
    sort_order = np.argsort(uid_arr, kind="stable")
    s_uids = uid_arr[sort_order]
    s_irows = item_rows_all[sort_order]
    uid_changes = np.concatenate(([True], s_uids[1:] != s_uids[:-1]))
    block_starts = np.where(uid_changes)[0]
    block_uids = s_uids[block_starts]
    block_ends = np.append(block_starts[1:], len(s_uids))

    def _cos_sorted(uid_to_row: dict[int, int], user_emb: np.ndarray) -> np.ndarray:
        """Per-user matmul; peak memory ≈ max_cands_per_user × dim × 4B."""
        result = np.full(len(sort_order), np.nan, dtype=np.float32)
        for uid, start, end in zip(block_uids, block_starts, block_ends):
            u_row = uid_to_row.get(int(uid))
            if u_row is None:
                continue
            irows = s_irows[start:end]
            valid = irows >= 0
            if valid.any():
                result[start:end][valid] = emb_arr[irows[valid]] @ user_emb[u_row]
        out = np.empty(len(sort_order), dtype=np.float32)
        out[sort_order] = result
        return out

    cos_all = _cos_sorted(*mean_all)
    cos_liked = _cos_sorted(*mean_liked)
    cos_disliked = _cos_sorted(*mean_disliked)
    cos_lastk_per_k = {k: _cos_sorted(*mean_lastk_per_k[k]) for k in last_k_list}

    # Release the embedding matrix (~730 MB on 500m) and all per-user-mean
    # arrays + sort scratch BEFORE we build the output DataFrame and return.
    # Otherwise these stay alive through the downstream Polars hash-joins
    # in add_features → contributes to OOM during features_lf.collect().
    # Killing _cos_sorted / _item_rows releases their cell-captured references
    # to emb_arr / sort_order / s_irows / etc.
    del emb_arr, emb_item_ids, item_to_row
    del mean_all, mean_liked, mean_disliked, mean_lastk_per_k
    del s_uids, s_irows, item_rows_all, sort_order
    del block_starts, block_ends, block_uids, uid_changes
    del emb_sorted_pos, emb_sorted_ids
    del _cos_sorted, _item_rows
    gc.collect()

    out_cols: dict[str, np.ndarray] = {
        "uid": uid_arr,
        "item_id": iid_arr,
        "embed_cos_user_mean": cos_all,
    }
    # Insert all last_K cosines in ascending K order — keeps schema stable.
    for k in last_k_list:
        out_cols[f"embed_cos_user_last_{k}"] = cos_lastk_per_k[k]
    out_cols["embed_cos_user_liked_mean"] = cos_liked
    out_cols["embed_cos_user_disliked_mean"] = cos_disliked

    return (
        pl.DataFrame(out_cols)
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
        .lazy()
    )


def _effective_likes_lf(
    likes_lf: pl.LazyFrame,
    unlikes_lf: pl.LazyFrame,
    cutoff_ts: int,
) -> pl.LazyFrame:
    """``(uid, item_id)`` pairs that are *currently* liked at ``cutoff_ts``."""
    last_like = (
        likes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
        .group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").max().alias("_lts"))
    )
    last_unlike = (
        unlikes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
        .group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").max().alias("_uts"))
    )
    return (
        last_like
        .join(last_unlike, on=["uid", "item_id"], how="left")
        .filter(pl.col("_uts").is_null() | (pl.col("_lts") > pl.col("_uts")))
        .select(["uid", "item_id"])
    )


def _effective_dislikes_lf(
    dislikes_lf: pl.LazyFrame,
    undislikes_lf: pl.LazyFrame,
    cutoff_ts: int,
) -> pl.LazyFrame:
    """LazyFrame variant of ``dataset.effective_dislikes`` — same semantics."""
    last_dis = (
        dislikes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
        .group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").max().alias("_dts"))
    )
    last_undis = (
        undislikes_lf.filter(pl.col("timestamp") <= cutoff_ts)
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
        .group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").max().alias("_udts"))
    )
    return (
        last_dis
        .join(last_undis, on=["uid", "item_id"], how="left")
        .filter(pl.col("_udts").is_null() | (pl.col("_dts") > pl.col("_udts")))
        .select(["uid", "item_id"])
    )


def _user_mean_embeddings(
    pairs: pl.DataFrame,
    item_to_row: dict[int, int],
    emb_arr: np.ndarray,
    dim: int,
) -> tuple[dict[int, int], np.ndarray]:
    """For each uid, average emb_arr rows of the items it pairs with.

    Returns ``(uid -> row_idx, mean_emb)`` where ``mean_emb`` is L2-normed
    so dot product against item rows is cosine similarity.
    """
    if len(pairs) == 0:
        return {}, np.zeros((0, dim), dtype=np.float32)

    uid_np = pairs["uid"].cast(pl.Int64).to_numpy()
    iid_np = pairs["item_id"].cast(pl.Int64).to_numpy()
    rows = np.fromiter(
        (item_to_row.get(int(i), -1) for i in iid_np),
        dtype=np.int64,
        count=len(iid_np),
    )
    valid = rows >= 0
    if not valid.any():
        return {}, np.zeros((0, dim), dtype=np.float32)

    uid_v = uid_np[valid]
    rows_v = rows[valid]
    unique_uids, inv = np.unique(uid_v, return_inverse=True)
    n_users = len(unique_uids)
    n_emb = emb_arr.shape[0]
    # Sparse-scatter via CSR @ dense: avoids the (len(rows_v), dim) fancy-index
    # temporary that np.add.at materialises (~50 GB on 500m) and is 5–20× faster
    # via BLAS SpMM. csr_matrix sums duplicate (row, col) pairs by construction —
    # same semantics as np.add.at.
    ones = np.ones(len(rows_v), dtype=np.float32)
    S = csr_matrix((ones, (inv, rows_v)), shape=(n_users, n_emb))
    sums = np.asarray(S @ emb_arr, dtype=np.float32)
    counts = np.asarray(S.sum(axis=1)).reshape(-1).astype(np.float32)
    means = sums / counts[:, None]
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    means = (means / norms).astype(np.float32)
    uid_to_row = {int(u): k for k, u in enumerate(unique_uids.tolist())}
    return uid_to_row, means


def _user_mean_last_k(
    listens_pairs: pl.DataFrame,
    last_k: int,
    item_to_row: dict[int, int],
    emb_arr: np.ndarray,
    dim: int,
) -> tuple[dict[int, int], np.ndarray]:
    """User mean over their last ``last_k`` listens (chronological)."""
    if len(listens_pairs) == 0:
        return {}, np.zeros((0, dim), dtype=np.float32)
    # Tiebreaker by item_id makes head(K) deterministic when several
    # listens for the same user share a timestamp (Yambda bins ts to 5s
    # buckets so collisions are common). Without it, the row-order of
    # ``listens_pairs`` leaks into the result — chunked vs single-pass
    # builds produce different "last K" item sets and thus different
    # embed_cos_user_last_{K} values for affected users.
    last_k_df = (
        listens_pairs
        .sort(
            ["uid", "timestamp", "item_id"],
            descending=[False, True, False],
        )
        .group_by("uid", maintain_order=True)
        .head(last_k)
        .select(["uid", "item_id"])
    )
    return _user_mean_embeddings(last_k_df, item_to_row, emb_arr, dim)


def _empty_embed_frame(
    cand_pairs: pl.DataFrame,
    last_k_list: list[int],
) -> pl.DataFrame:
    """All-NULL embed feature frame for the case of no embedding coverage.

    Schema must match the populated path in :func:`build_embed_features` —
    one ``embed_cos_user_last_{K}`` column per K in ``last_k_list``.
    """
    n = len(cand_pairs)
    cols: dict[str, np.ndarray] = {
        "uid": cand_pairs["uid"].to_numpy(),
        "item_id": cand_pairs["item_id"].to_numpy(),
        "embed_cos_user_mean": np.full(n, np.nan, dtype=np.float32),
    }
    for k in last_k_list:
        cols[f"embed_cos_user_last_{k}"] = np.full(n, np.nan, dtype=np.float32)
    cols["embed_cos_user_liked_mean"] = np.full(n, np.nan, dtype=np.float32)
    cols["embed_cos_user_disliked_mean"] = np.full(n, np.nan, dtype=np.float32)
    return pl.DataFrame(cols).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
    ])


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
    embeddings_path: str | None = None,
    embed_last_k_list: list[int] | None = None,
) -> pl.LazyFrame:
    """Enrich candidates with user / item / pair / cross / embed features.

    The result preserves the original ``candidates_lf`` rows (left-joins
    everywhere) plus all feature columns. Caller decides on materialisation:

        features_lf = add_features(...)
        df = features_lf.collect(streaming=True)
        # OR
        features_lf.sink_parquet("cache.parquet", compression="zstd")

    ``embeddings_path``: if provided, append 4 audio-embedding cosine
    features (Phase C.4). Pass ``None`` (default) to skip.
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
        decay_half_life_units,
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
        # Cross feature: how stale is this pair vs the user's overall recency.
        # Adds 1d to the denominator to avoid div-by-0 for hyperactive users.
        .with_columns(
            (
                pl.col("pair_days_since_last_listen").cast(pl.Float32)
                / (pl.col("user_recency_last_listen").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("pair_recency_ratio")
        )
        # Five cross-ratio features capturing pair vs user/item normalisation,
        # item velocity (7d/30d), pair recency share, and user-artist focus.
        # All Float32; +1 in denominators avoids div-by-0 for cold rows.
        .with_columns([
            (
                pl.col("pair_n_listens").cast(pl.Float32)
                / (pl.col("user_n_listens").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("pair_share_user_listens"),
            (
                pl.col("pair_n_listens").cast(pl.Float32)
                / (pl.col("item_pop").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("pair_share_item_pop"),
            (
                pl.col("item_pop_7d").cast(pl.Float32)
                / (pl.col("item_pop_30d").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("item_pop_acceleration"),
            (
                pl.col("pair_n_listens_30d").cast(pl.Float32)
                / (pl.col("pair_n_listens").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("pair_recency_share_30d"),
            (
                pl.col("user_artist_listens").cast(pl.Float32)
                / (pl.col("user_n_listens").cast(pl.Float32) + 1.0)
            ).cast(pl.Float32).alias("user_artist_focus"),
        ])
    )

    if embeddings_path is not None:
        effective_k_list = embed_last_k_list or [5, 20, 50, 100]
        embed_feats = build_embed_features(
            candidates_lf, listens_lf, likes_lf, unlikes_lf,
            dislikes_lf, undislikes_lf,
            embeddings_path, cutoff_ts,
            last_k_list=embed_last_k_list,  # None → default [5, 20, 50, 100]
        )
        enriched_cands = enriched_cands.join(embed_feats, on=["uid", "item_id"], how="left")
        # Aggregate window cosines: max across windows captures the best-aligned
        # window for each pair, a robust alternative to selecting K by hand.
        last_k_cols = [f"embed_cos_user_last_{k}" for k in effective_k_list]
        if last_k_cols:
            enriched_cands = enriched_cands.with_columns(
                pl.max_horizontal(*[pl.col(c) for c in last_k_cols])
                .cast(pl.Float32)
                .alias("embed_cos_user_last_max")
            )

    return enriched_cands


def load_features_from_cache(path: str) -> pl.LazyFrame:
    """Lazy scan of a previously-cached features parquet."""
    return pl.scan_parquet(path)
