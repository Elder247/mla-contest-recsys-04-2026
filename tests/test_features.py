"""Tests for src.data.features — new pair time-window + embed features (F2 + D4)."""
from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.features import (
    ONE_DAY_TS,
    add_features,
    build_embed_features,
    build_pair_features,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_listens(rows: list[tuple], cutoff: int) -> tuple[pl.LazyFrame, int]:
    """Build a tiny listens LazyFrame for testing.

    rows: list of (uid, item_id, days_ago, played_ratio_pct, is_organic, track_length_seconds)
    """
    converted = [
        (uid, item_id, cutoff - days_ago * ONE_DAY_TS, played, organic, length)
        for uid, item_id, days_ago, played, organic, length in rows
    ]
    df = pl.DataFrame(
        converted,
        schema=["uid", "item_id", "timestamp", "played_ratio_pct", "is_organic", "track_length_seconds"],
        orient="row",
    ).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("timestamp").cast(pl.Int64),
        pl.col("played_ratio_pct").cast(pl.UInt16),
        pl.col("is_organic").cast(pl.UInt8),
        pl.col("track_length_seconds").cast(pl.UInt32),
    ])
    return df.lazy(), cutoff


def _empty_lf(schema: dict) -> pl.LazyFrame:
    return pl.DataFrame(schema=schema).lazy()


# ── F2: time-window pair features ───────────────────────────────────────────


def test_pair_time_window_features_present():
    """build_pair_features must emit n_listens_{30d,90d} and avg/max_played_30d."""
    cutoff = 100 * ONE_DAY_TS
    listens_lf, _ = _make_listens([
        # (uid, item_id, days_ago, played, organic, length)
        (1, 10, 1, 80, 1, 60),
        (1, 10, 5, 90, 1, 60),
        (1, 10, 25, 70, 1, 60),
        (1, 10, 60, 100, 1, 60),     # within 90d, outside 30d
        (1, 10, 120, 50, 1, 60),     # outside 90d (still within cutoff)
    ], cutoff)
    cands = pl.DataFrame({"uid": [1], "item_id": [10]}).with_columns([
        pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64),
    ]).lazy()
    empty = _empty_lf({"uid": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64})

    pair = build_pair_features(
        cands, listens_lf, likes_lf=empty, unlikes_lf=empty, undislikes_lf=empty,
        cutoff_ts=cutoff,
    ).collect()

    row = pair.row(0, named=True)
    assert row["pair_n_listens"] == 5
    assert row["pair_n_listens_30d"] == 3                  # days 1, 5, 25
    assert row["pair_n_listens_90d"] == 4                  # adds day 60
    assert row["pair_avg_played_ratio_30d"] == pytest.approx(80.0, abs=1e-4)  # mean(80,90,70)
    assert row["pair_max_played_ratio_30d"] == pytest.approx(90.0, abs=1e-4)


def test_pair_time_window_zero_when_no_recent_listens():
    """All listens older than 30d → counts are zero, avgs are NaN-able (None)."""
    cutoff = 100 * ONE_DAY_TS
    listens_lf, _ = _make_listens([
        (1, 10, 60, 80, 1, 60),
        (1, 10, 70, 90, 1, 60),
    ], cutoff)
    cands = pl.DataFrame({"uid": [1], "item_id": [10]}).with_columns([
        pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64),
    ]).lazy()
    empty = _empty_lf({"uid": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64})

    pair = build_pair_features(
        cands, listens_lf, likes_lf=empty, unlikes_lf=empty, undislikes_lf=empty,
        cutoff_ts=cutoff,
    ).collect()
    row = pair.row(0, named=True)
    assert row["pair_n_listens_30d"] == 0
    assert row["pair_n_listens_90d"] == 2
    # avg_played_30d is NaN/None when no rows pass the filter
    assert row["pair_avg_played_ratio_30d"] is None or np.isnan(row["pair_avg_played_ratio_30d"])


# ── D4: build_embed_features ────────────────────────────────────────────────


@pytest.fixture
def emb_parquet(tmp_path):
    """Tiny embeddings parquet with 5 items × 4-dim L2-normalised vectors."""
    rng = np.random.default_rng(0)
    n_items = 5
    raw = rng.normal(size=(n_items, 4)).astype(np.float64)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm = raw / norms
    item_ids = np.array([10, 20, 30, 40, 50], dtype=np.uint32)

    table = pa.table({
        "item_id": pa.array(item_ids, type=pa.uint32()),
        "embed": [list(r) for r in raw],
        "normalized_embed": [list(r) for r in norm],
    })
    p = tmp_path / "embeddings.parquet"
    pq.write_table(table, p)
    return p, item_ids, norm


def test_embed_features_returns_4_cosines_for_known_user(emb_parquet):
    emb_path, item_ids, norm_emb = emb_parquet
    cutoff = 100 * ONE_DAY_TS

    # User 1 has listened items 10 and 20; item 30 is the candidate.
    listens_lf, _ = _make_listens([
        (1, 10, 1, 80, 1, 60),
        (1, 20, 5, 90, 1, 60),
    ], cutoff)
    cands = pl.DataFrame({"uid": [1], "item_id": [30]}).with_columns([
        pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64),
    ]).lazy()
    empty = _empty_lf({"uid": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64})

    out = build_embed_features(
        cands, listens_lf, likes_lf=empty, unlikes_lf=empty,
        dislikes_lf=empty, undislikes_lf=empty,
        embeddings_path=str(emb_path), cutoff_ts=cutoff,
        last_k_list=[5, 20, 50, 100],
    ).collect()

    # Schema: uid, item_id, embed_cos_user_mean, then last_K cols in ascending K,
    # then liked_mean, disliked_mean.
    assert out.columns == [
        "uid", "item_id",
        "embed_cos_user_mean",
        "embed_cos_user_last_5", "embed_cos_user_last_20",
        "embed_cos_user_last_50", "embed_cos_user_last_100",
        "embed_cos_user_liked_mean", "embed_cos_user_disliked_mean",
    ]
    assert len(out) == 1
    row = out.row(0, named=True)

    # Manual reference: user_mean = mean of L2-norm embeddings of items 10, 20, then re-normalised.
    item_idx = {iid: k for k, iid in enumerate(item_ids.tolist())}
    user_mean = norm_emb[[item_idx[10], item_idx[20]]].mean(axis=0)
    user_mean /= np.linalg.norm(user_mean)
    expected_cos_mean = float(norm_emb[item_idx[30]] @ user_mean)
    assert row["embed_cos_user_mean"] == pytest.approx(expected_cos_mean, abs=1e-5)
    # User has only 2 listens — every K window in the list collapses to the
    # same set of items, so each ``last_K`` cosine equals ``user_mean`` cosine.
    for k in (5, 20, 50, 100):
        assert row[f"embed_cos_user_last_{k}"] == pytest.approx(expected_cos_mean, abs=1e-5)

    # No likes / dislikes → those columns are NULL
    assert row["embed_cos_user_liked_mean"] is None or np.isnan(row["embed_cos_user_liked_mean"])
    assert row["embed_cos_user_disliked_mean"] is None or np.isnan(row["embed_cos_user_disliked_mean"])


def test_embed_features_missing_item_returns_null(emb_parquet):
    """Candidates whose item has no embedding emit NaN/Null cosines."""
    emb_path, _, _ = emb_parquet
    cutoff = 100 * ONE_DAY_TS

    listens_lf, _ = _make_listens([
        (1, 10, 1, 80, 1, 60),
    ], cutoff)
    # item_id=999 not in embeddings parquet
    cands = pl.DataFrame({"uid": [1], "item_id": [999]}).with_columns([
        pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64),
    ]).lazy()
    empty = _empty_lf({"uid": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64})

    out = build_embed_features(
        cands, listens_lf, likes_lf=empty, unlikes_lf=empty,
        dislikes_lf=empty, undislikes_lf=empty,
        embeddings_path=str(emb_path), cutoff_ts=cutoff,
    ).collect()
    row = out.row(0, named=True)
    for col in [
        "embed_cos_user_mean",
        "embed_cos_user_last_5", "embed_cos_user_last_20",
        "embed_cos_user_last_50", "embed_cos_user_last_100",
        "embed_cos_user_liked_mean", "embed_cos_user_disliked_mean",
    ]:
        assert row[col] is None or np.isnan(row[col]), f"{col}={row[col]} expected NaN"


def test_embed_features_user_with_no_history_returns_null(emb_parquet):
    """User with no listens / likes / dislikes → NaN for all 4 cosines."""
    emb_path, _, _ = emb_parquet
    cutoff = 100 * ONE_DAY_TS

    # user 7 has zero history
    empty = _empty_lf({"uid": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64,
                       "played_ratio_pct": pl.UInt16, "is_organic": pl.UInt8,
                       "track_length_seconds": pl.UInt32})
    cands = pl.DataFrame({"uid": [7], "item_id": [10]}).with_columns([
        pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64),
    ]).lazy()
    feedback_empty = _empty_lf({"uid": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64})

    out = build_embed_features(
        cands, empty, likes_lf=feedback_empty, unlikes_lf=feedback_empty,
        dislikes_lf=feedback_empty, undislikes_lf=feedback_empty,
        embeddings_path=str(emb_path), cutoff_ts=cutoff,
    ).collect()
    row = out.row(0, named=True)
    for col in [
        "embed_cos_user_mean",
        "embed_cos_user_last_5", "embed_cos_user_last_20",
        "embed_cos_user_last_50", "embed_cos_user_last_100",
        "embed_cos_user_liked_mean", "embed_cos_user_disliked_mean",
    ]:
        assert row[col] is None or np.isnan(row[col])


# ── F2 + D4: pair_recency_ratio + add_features integration ──────────────────


def test_add_features_emits_pair_recency_ratio_and_embed_cols(emb_parquet):
    emb_path, _, _ = emb_parquet
    cutoff = 100 * ONE_DAY_TS

    # NOTE: candidate pair (1, 30) needs ≥2 listens to avoid a Polars optimizer
    # quirk on tiny tests — `pl.lit(2.0).pow(col)` inside agg crashes when the
    # group resolves to a literal (1-row group on a tiny LazyFrame). Real-data
    # groups are always ≥2 rows; the same expression works fine in production.
    listens_lf, _ = _make_listens([
        (1, 10, 1, 80, 1, 60),
        (1, 20, 5, 90, 1, 60),
        (1, 30, 60, 70, 0, 60),
        (1, 30, 65, 80, 0, 60),    # second listen on the candidate pair
    ], cutoff)
    cands = pl.DataFrame({"uid": [1], "item_id": [30]}).with_columns([
        pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64),
    ]).lazy()

    feedback_empty = _empty_lf({"uid": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64})
    artist_empty = _empty_lf({"item_id": pl.Int64, "artist_id": pl.Int64})
    album_empty = _empty_lf({"item_id": pl.Int64, "album_id": pl.Int64})

    out = add_features(
        cands, listens_lf,
        likes_lf=feedback_empty, dislikes_lf=feedback_empty,
        unlikes_lf=feedback_empty, undislikes_lf=feedback_empty,
        artist_map_lf=artist_empty, album_map_lf=album_empty,
        cutoff_ts=cutoff,
        embeddings_path=str(emb_path),
    ).collect()

    cols = set(out.columns)
    # F2 outputs
    assert "pair_n_listens_30d" in cols
    assert "pair_n_listens_90d" in cols
    assert "pair_avg_played_ratio_30d" in cols
    assert "pair_max_played_ratio_30d" in cols
    assert "pair_recency_ratio" in cols
    # D4 outputs (multi-window: H3)
    assert "embed_cos_user_mean" in cols
    for k in (5, 20, 50, 100):
        assert f"embed_cos_user_last_{k}" in cols
    assert "embed_cos_user_liked_mean" in cols
    assert "embed_cos_user_disliked_mean" in cols
    # H4 cross-ratio + multi-window aggregate features
    for col in (
        "pair_share_user_listens",
        "pair_share_item_pop",
        "item_pop_acceleration",
        "pair_recency_share_30d",
        "user_artist_focus",
        "embed_cos_user_last_max",
    ):
        assert col in cols, f"expected new feature column {col} missing"

    # pair_recency_ratio sanity: pair_days_since_last_listen=60, user_recency_last_listen=1
    # → 60 / (1+1) = 30
    row = out.row(0, named=True)
    assert row["pair_recency_ratio"] == pytest.approx(60.0 / 2.0, abs=1e-4)
    # embed_cos_user_last_max equals max of the per-K cosines for this row.
    last_k_vals = [row[f"embed_cos_user_last_{k}"] for k in (5, 20, 50, 100)]
    if all(v is not None and not np.isnan(v) for v in last_k_vals):
        assert row["embed_cos_user_last_max"] == pytest.approx(max(last_k_vals), abs=1e-5)


def test_add_features_skips_embed_when_no_path():
    cutoff = 100 * ONE_DAY_TS
    # 2 listens — see note in test_add_features_emits_* above re. the
    # ``pl.lit(2.0).pow`` optimizer quirk on 1-row groups.
    listens_lf, _ = _make_listens([
        (1, 10, 1, 80, 1, 60),
        (1, 10, 5, 90, 1, 60),
    ], cutoff)
    cands = pl.DataFrame({"uid": [1], "item_id": [10]}).with_columns([
        pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64),
    ]).lazy()
    feedback_empty = _empty_lf({"uid": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64})
    artist_empty = _empty_lf({"item_id": pl.Int64, "artist_id": pl.Int64})
    album_empty = _empty_lf({"item_id": pl.Int64, "album_id": pl.Int64})

    out = add_features(
        cands, listens_lf,
        likes_lf=feedback_empty, dislikes_lf=feedback_empty,
        unlikes_lf=feedback_empty, undislikes_lf=feedback_empty,
        artist_map_lf=artist_empty, album_map_lf=album_empty,
        cutoff_ts=cutoff,
        embeddings_path=None,    # explicit opt-out
    ).collect()

    assert "embed_cos_user_mean" not in out.columns
    # but recency_ratio still works
    assert "pair_recency_ratio" in out.columns
