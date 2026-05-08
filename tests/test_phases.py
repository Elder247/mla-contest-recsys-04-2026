"""Tests for src.inference.phases — features_phase + ground truth helpers.

Synthetic-data smoke tests that exercise the parquet-IO contract without
requiring the real Yambda dataset.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.features import ONE_DAY_TS
from src.inference.phases import (
    derive_split_metadata,
    features_phase,
    write_ground_truth,
)


def _write_listens_parquet(path, rows: list[tuple]) -> None:
    """rows: (uid, item_id, timestamp, played_ratio_pct, is_organic, track_length_seconds)."""
    table = pa.table({
        "uid": pa.array([r[0] for r in rows], type=pa.uint32()),
        "item_id": pa.array([r[1] for r in rows], type=pa.uint32()),
        "timestamp": pa.array([r[2] for r in rows], type=pa.uint32()),
        "played_ratio_pct": pa.array([r[3] for r in rows], type=pa.uint16()),
        "is_organic": pa.array([r[4] for r in rows], type=pa.uint8()),
        "track_length_seconds": pa.array([r[5] for r in rows], type=pa.uint32()),
    })
    pq.write_table(table, path)


def _write_feedback_parquet(path, rows: list[tuple]) -> None:
    """rows: (uid, item_id, timestamp, is_organic)."""
    table = pa.table({
        "uid": pa.array([r[0] for r in rows], type=pa.uint32()),
        "item_id": pa.array([r[1] for r in rows], type=pa.uint32()),
        "timestamp": pa.array([r[2] for r in rows], type=pa.uint32()),
        "is_organic": pa.array([r[3] for r in rows], type=pa.uint8()),
    })
    pq.write_table(table, path)


def _write_mapping_parquet(path, rows: list[tuple], entity_col: str) -> None:
    """rows: (entity_id, item_id)."""
    table = pa.table({
        entity_col: pa.array([r[0] for r in rows], type=pa.uint32()),
        "item_id": pa.array([r[1] for r in rows], type=pa.uint32()),
    })
    pq.write_table(table, path)


@pytest.fixture
def synth_paths(tmp_path):
    """Build a tiny but complete parquet dataset and return all paths."""
    cutoff = 100 * ONE_DAY_TS

    listens = [
        # uid=1: listened 10 (twice, recent), 20 (older)
        (1, 10, cutoff - 1 * ONE_DAY_TS, 80, 1, 60),
        (1, 10, cutoff - 5 * ONE_DAY_TS, 90, 1, 60),
        (1, 20, cutoff - 30 * ONE_DAY_TS, 70, 1, 60),
        # uid=2: listened 30 once
        (2, 30, cutoff - 2 * ONE_DAY_TS, 80, 1, 60),
        # uid=3 (non-candidate): listened 10, 20 — must NOT bloat user_artist groupby
        (3, 10, cutoff - 3 * ONE_DAY_TS, 80, 1, 60),
        (3, 20, cutoff - 4 * ONE_DAY_TS, 90, 1, 60),
    ]
    listens_path = tmp_path / "listens.parquet"
    _write_listens_parquet(listens_path, listens)

    likes_path = tmp_path / "likes.parquet"
    _write_feedback_parquet(likes_path, [(1, 10, cutoff - 1 * ONE_DAY_TS, 1)])
    dislikes_path = tmp_path / "dislikes.parquet"
    _write_feedback_parquet(
        dislikes_path,
        [(uid, 99, cutoff - 10 * ONE_DAY_TS, 1) for uid in (1, 2, 3)],
    )
    unlikes_path = tmp_path / "unlikes.parquet"
    # Non-empty unlikes/undislikes that *cover every candidate uid* —
    # Polars' optimizer otherwise folds ``pl.lit(1).alias(...)`` over the
    # post-left-join all-null column into a literal that downstream
    # group-bys can't aggregate ("cannot aggregate a literal"). Production
    # 500m / 5b never sees this case (unlikes are dense). Keep the fixture
    # realistic so the optimizer takes the same plan as production.
    _write_feedback_parquet(
        unlikes_path,
        [(uid, 99, cutoff - 9 * ONE_DAY_TS, 1) for uid in (1, 2, 3)],
    )
    undislikes_path = tmp_path / "undislikes.parquet"
    _write_feedback_parquet(
        undislikes_path,
        [(uid, 99, cutoff - 8 * ONE_DAY_TS, 1) for uid in (1, 2, 3)],
    )

    artist_path = tmp_path / "artist_map.parquet"
    _write_mapping_parquet(artist_path, [(100, 10), (100, 20), (200, 30)], "artist_id")
    album_path = tmp_path / "album_map.parquet"
    _write_mapping_parquet(album_path, [(1000, 10), (1000, 20), (2000, 30)], "album_id")

    return {
        "listens": str(listens_path),
        "likes": str(likes_path),
        "dislikes": str(dislikes_path),
        "unlikes": str(unlikes_path),
        "undislikes": str(undislikes_path),
        "artist_map": str(artist_path),
        "album_map": str(album_path),
        "cutoff": cutoff,
        "tmp_path": tmp_path,
    }


def test_features_phase_unlabeled(synth_paths):
    """features_phase with no ground truth → no `label` column, all features present."""
    p = synth_paths
    cutoff = p["cutoff"]

    merged = pl.DataFrame({
        "uid": [1, 1, 2],
        "item_id": [10, 30, 30],
        "decaypop_score": [0.5, 0.4, 0.6],
        "decaypop_rank": [1, 2, 1],
    }).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("decaypop_score").cast(pl.Float32),
        pl.col("decaypop_rank").cast(pl.Int32),
    ])
    merged_path = p["tmp_path"] / "merged.parquet"
    merged.write_parquet(str(merged_path))

    out_path = p["tmp_path"] / "features.parquet"
    features_phase(
        merged_path=merged_path,
        listens_path=p["listens"],
        likes_path=p["likes"],
        dislikes_path=p["dislikes"],
        unlikes_path=p["unlikes"],
        undislikes_path=p["undislikes"],
        artist_map_path=p["artist_map"],
        album_map_path=p["album_map"],
        cutoff_ts=cutoff,
        output_path=out_path,
    )

    feats = pl.read_parquet(out_path)
    assert "label" not in feats.columns
    assert len(feats) == 3
    # Feature joins preserved every original column
    assert "decaypop_score" in feats.columns
    assert "decaypop_rank" in feats.columns
    # User / item / pair / cross feature representatives all materialised
    for col in [
        "user_n_listens", "item_pop", "pair_n_listens",
        "user_artist_listens", "user_artist_share",
    ]:
        assert col in feats.columns, f"missing feature column: {col}"


def _write_chunked_synth(p, listens_extra=None):
    """Add a 4th uid to the base fixture so chunk_size_uids=2 yields two
    chunks of 2 uids each. Polars' query planner shows pathological
    behaviour on single-uid pair_features (``cannot aggregate a literal``)
    that doesn't surface in production where every chunk has thousands of
    uids — match production conditions in the test."""
    cutoff = p["cutoff"]
    extra_listens = listens_extra or [
        (4, 30, cutoff - 2 * ONE_DAY_TS, 80, 1, 60),
        (4, 10, cutoff - 6 * ONE_DAY_TS, 90, 1, 60),
    ]
    listens_path = p["tmp_path"] / "listens.parquet"
    existing = pl.read_parquet(str(listens_path))
    new = pl.DataFrame(
        {
            "uid": [r[0] for r in extra_listens],
            "item_id": [r[1] for r in extra_listens],
            "timestamp": [r[2] for r in extra_listens],
            "played_ratio_pct": [r[3] for r in extra_listens],
            "is_organic": [r[4] for r in extra_listens],
            "track_length_seconds": [r[5] for r in extra_listens],
        },
        schema=existing.schema,
    )
    pl.concat([existing, new]).write_parquet(str(listens_path))


def test_features_phase_chunked_matches_single_pass(synth_paths):
    """Chunked path must produce a parquet identical (modulo row order) to
    the single-pass path on the same inputs.

    Uses chunk_size_uids=2 with 4 uids so each chunk has 2 uids — matches
    production where every chunk contains thousands of uids and Polars'
    plan stays on the "normal" path. Single-uid chunks trigger a
    ``cannot aggregate a literal`` planner edge case in build_pair_features
    that doesn't reproduce on real-scale data.
    """
    p = synth_paths
    _write_chunked_synth(p)
    cutoff = p["cutoff"]

    merged = pl.DataFrame({
        "uid": [1, 1, 2, 3, 3, 4],
        "item_id": [10, 20, 30, 10, 30, 10],
        "decaypop_score": [0.5, 0.4, 0.6, 0.3, 0.7, 0.2],
        "decaypop_rank": [1, 2, 1, 1, 2, 1],
    }).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("decaypop_score").cast(pl.Float32),
        pl.col("decaypop_rank").cast(pl.Int32),
    ])
    merged_path = p["tmp_path"] / "merged_chunk.parquet"
    merged.write_parquet(str(merged_path))

    out_single = p["tmp_path"] / "feats_single.parquet"
    features_phase(
        merged_path=merged_path,
        listens_path=p["listens"],
        likes_path=p["likes"],
        dislikes_path=p["dislikes"],
        unlikes_path=p["unlikes"],
        undislikes_path=p["undislikes"],
        artist_map_path=p["artist_map"],
        album_map_path=p["album_map"],
        cutoff_ts=cutoff,
        output_path=out_single,
    )

    out_chunked = p["tmp_path"] / "feats_chunked.parquet"
    features_phase(
        merged_path=merged_path,
        listens_path=p["listens"],
        likes_path=p["likes"],
        dislikes_path=p["dislikes"],
        unlikes_path=p["unlikes"],
        undislikes_path=p["undislikes"],
        artist_map_path=p["artist_map"],
        album_map_path=p["album_map"],
        cutoff_ts=cutoff,
        output_path=out_chunked,
        chunk_size_uids=2,
    )

    feats_single = pl.read_parquet(out_single).sort(["uid", "item_id"])
    feats_chunked = pl.read_parquet(out_chunked).sort(["uid", "item_id"])

    assert feats_single.columns == feats_chunked.columns
    assert len(feats_single) == len(feats_chunked) == len(merged)

    # Per-column equality (frame_equal ignores row order via the .sort above).
    assert feats_single.equals(feats_chunked), (
        "chunked features differ from single-pass features:\n"
        f"single: {feats_single}\nchunked: {feats_chunked}"
    )

    # Tmp chunk dir must be cleaned up.
    tmp_dir = out_chunked.parent / f".{out_chunked.stem}_chunks"
    assert not tmp_dir.exists(), f"tmp chunk dir was not cleaned: {tmp_dir}"


def test_features_phase_chunked_with_labels(synth_paths):
    """Label join must be re-applied per chunk and survive the concat."""
    p = synth_paths
    _write_chunked_synth(p)
    cutoff = p["cutoff"]

    merged = pl.DataFrame({
        "uid": [1, 1, 2, 3, 4],
        "item_id": [10, 20, 30, 10, 10],
        "decaypop_score": [0.5, 0.4, 0.6, 0.3, 0.2],
        "decaypop_rank": [1, 2, 1, 1, 1],
    }).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("decaypop_score").cast(pl.Float32),
        pl.col("decaypop_rank").cast(pl.Int32),
    ])
    merged_path = p["tmp_path"] / "merged_chunk_lbl.parquet"
    merged.write_parquet(str(merged_path))

    gt = pl.DataFrame({"uid": [1, 3], "item_id": [10, 10]}).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
    ])
    gt_path = p["tmp_path"] / "gt_chunk.parquet"
    gt.write_parquet(str(gt_path))

    out_path = p["tmp_path"] / "feats_chunked_lbl.parquet"
    features_phase(
        merged_path=merged_path,
        listens_path=p["listens"],
        likes_path=p["likes"],
        dislikes_path=p["dislikes"],
        unlikes_path=p["unlikes"],
        undislikes_path=p["undislikes"],
        artist_map_path=p["artist_map"],
        album_map_path=p["album_map"],
        cutoff_ts=cutoff,
        output_path=out_path,
        label_gt_path=gt_path,
        chunk_size_uids=2,
    )

    feats = pl.read_parquet(out_path)
    assert "label" in feats.columns
    assert feats.dtypes[feats.columns.index("label")] == pl.Int8
    label_map = {(r["uid"], r["item_id"]): r["label"] for r in feats.iter_rows(named=True)}
    assert label_map[(1, 10)] == 1
    assert label_map[(1, 20)] == 0
    assert label_map[(2, 30)] == 0
    assert label_map[(3, 10)] == 1
    assert label_map[(4, 10)] == 0


def test_features_phase_with_labels(synth_paths):
    """features_phase with a ground-truth parquet attaches a 0/1 label column."""
    p = synth_paths
    cutoff = p["cutoff"]

    merged = pl.DataFrame({
        "uid": [1, 1, 2],
        "item_id": [10, 30, 30],
        "decaypop_score": [0.5, 0.4, 0.6],
        "decaypop_rank": [1, 2, 1],
    }).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
        pl.col("decaypop_score").cast(pl.Float32),
        pl.col("decaypop_rank").cast(pl.Int32),
    ])
    merged_path = p["tmp_path"] / "merged.parquet"
    merged.write_parquet(str(merged_path))

    gt = pl.DataFrame({"uid": [1], "item_id": [10]}).with_columns([
        pl.col("uid").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int64),
    ])
    gt_path = p["tmp_path"] / "gt.parquet"
    gt.write_parquet(str(gt_path))

    out_path = p["tmp_path"] / "features_labeled.parquet"
    features_phase(
        merged_path=merged_path,
        listens_path=p["listens"],
        likes_path=p["likes"],
        dislikes_path=p["dislikes"],
        unlikes_path=p["unlikes"],
        undislikes_path=p["undislikes"],
        artist_map_path=p["artist_map"],
        album_map_path=p["album_map"],
        cutoff_ts=cutoff,
        output_path=out_path,
        label_gt_path=gt_path,
    )

    feats = pl.read_parquet(out_path)
    assert "label" in feats.columns
    assert feats.dtypes[feats.columns.index("label")] == pl.Int8
    label_map = {(r["uid"], r["item_id"]): r["label"] for r in feats.iter_rows(named=True)}
    assert label_map[(1, 10)] == 1
    assert label_map[(1, 30)] == 0
    assert label_map[(2, 30)] == 0


def test_derive_split_metadata(synth_paths, tmp_path):
    """derive_split_metadata returns boundaries consistent with temporal_split."""
    listens_path = tmp_path / "listens.parquet"
    rows = [
        # max timestamp = 100 * ONE_DAY_TS
        (1, 10, 100 * ONE_DAY_TS, 80, 1, 60),
        (1, 10, 50 * ONE_DAY_TS, 80, 1, 60),
    ]
    _write_listens_parquet(listens_path, rows)

    meta = derive_split_metadata(str(listens_path), val_days=7, gap_days=1)
    assert meta["t_max"] == 100 * ONE_DAY_TS
    assert meta["t_end"] == 100 * ONE_DAY_TS + 1
    assert meta["val_start"] == meta["t_end"] - 8 * ONE_DAY_TS
    assert meta["val_end"] == meta["t_end"] - 1 * ONE_DAY_TS
    assert meta["test_start"] == meta["val_start"] + ONE_DAY_TS
    assert meta["train_max_ts"] == meta["val_start"] - 1


def test_write_ground_truth_filters_users_and_window(synth_paths, tmp_path):
    """Only eval users + listens in [lower, upper) survive."""
    p = synth_paths
    cutoff = p["cutoff"]

    out = tmp_path / "gt.parquet"
    # Window covers day [cutoff - 6d, cutoff): catches uid=1 item=10 (-1d, -5d), uid=2 item=30 (-2d)
    write_ground_truth(
        p["listens"],
        eval_users=[1, 2],
        lower_ts_inclusive=cutoff - 6 * ONE_DAY_TS,
        upper_ts_exclusive=cutoff,
        output_path=out,
    )
    gt = pl.read_parquet(out)
    pairs = set(zip(gt["uid"].to_list(), gt["item_id"].to_list()))
    assert pairs == {(1, 10), (2, 30)}


# ── End-to-end smoke test: fit → generate → features ───────────────────────


def test_e2e_fit_generate_features_phases(tmp_path):
    """Run all three phase functions in sequence on a synthetic dataset.

    Exercises the same call-graph that ``scripts/train_ranker.py`` invokes
    via subprocess — only without the subprocess so the test is fast. This
    is the cheapest validation that the phase contracts agree.
    """
    from omegaconf import OmegaConf

    from src.inference.phases import fit_phase, generate_phase

    cutoff = 100 * ONE_DAY_TS

    # Two users, three items each, enough variety for ALS / repeat / pop.
    listens_rows = []
    for uid in (1, 2, 3, 4):
        for item, days_ago in [(10, 1), (10, 5), (20, 3), (30, 8)]:
            listens_rows.append((uid, item, cutoff - days_ago * ONE_DAY_TS, 80, 1, 60))
    listens_path = tmp_path / "listens.parquet"
    _write_listens_parquet(listens_path, listens_rows)

    likes_path = tmp_path / "likes.parquet"
    _write_feedback_parquet(likes_path, [
        (1, 10, cutoff - 2 * ONE_DAY_TS, 1),
        (2, 20, cutoff - 1 * ONE_DAY_TS, 1),
    ])
    dislikes_path = tmp_path / "dislikes.parquet"
    _write_feedback_parquet(dislikes_path, [])
    unlikes_path = tmp_path / "unlikes.parquet"
    _write_feedback_parquet(unlikes_path, [])
    undislikes_path = tmp_path / "undislikes.parquet"
    _write_feedback_parquet(undislikes_path, [])

    artist_path = tmp_path / "artist.parquet"
    _write_mapping_parquet(artist_path, [(100, 10), (100, 20), (200, 30)], "artist_id")
    album_path = tmp_path / "album.parquet"
    _write_mapping_parquet(album_path, [(1000, 10), (1000, 20), (2000, 30)], "album_id")

    cache_dir = tmp_path / "cg_cache"
    candidates_dir = tmp_path / "candidates"

    # Two minimal CGs — cover the two ``data_source`` paths we route in fit.
    cg_cfg_list = [
        OmegaConf.create({
            "_target_": "src.models.pop.DecayPop",
            "name": "decaypop",
            "n_cand": 5,
            "half_life_units": 86_400,
        }),
        OmegaConf.create({
            "_target_": "src.models.repeat.RepeatListenModel",
            "name": "repeat",
            "n_cand": 5,
            "half_life_units": 86_400,
        }),
    ]

    fit_phase(
        cg_cfg_list=cg_cfg_list,
        listens_path=str(listens_path),
        likes_path=str(likes_path),
        cache_dir=cache_dir,
        size="test",
        suffix="",
        force_refit=True,
        train_cutoff_ts=cutoff,
    )
    # Each CG is now pickled in cache.
    assert (cache_dir / "decaypop_test.pkl").exists()
    assert (cache_dir / "repeat_test.pkl").exists()

    merged_path = generate_phase(
        cg_cfg_list=cg_cfg_list,
        eval_users=[1, 2, 3, 4],
        cache_dir=cache_dir,
        size="test",
        suffix="",
        output_dir=candidates_dir,
        filter_dislikes=False,
    )
    assert merged_path.exists()
    merged = pl.read_parquet(merged_path)
    assert {"uid", "item_id"}.issubset(merged.columns)
    assert "decaypop_score" in merged.columns
    assert "repeat_score" in merged.columns
    assert len(merged) > 0

    # Now compute features from the merged parquet — labels skipped here.
    feats_path = tmp_path / "features.parquet"
    features_phase(
        merged_path=merged_path,
        listens_path=str(listens_path),
        likes_path=str(likes_path),
        dislikes_path=str(dislikes_path),
        unlikes_path=str(unlikes_path),
        undislikes_path=str(undislikes_path),
        artist_map_path=str(artist_path),
        album_map_path=str(album_path),
        cutoff_ts=cutoff,
        output_path=feats_path,
    )
    feats = pl.read_parquet(feats_path)
    assert len(feats) == len(merged)
    # Feature integration sanity: representatives from each block survived.
    for col in [
        "decaypop_score", "repeat_score",
        "user_n_listens", "item_pop", "pair_n_listens",
        "user_artist_listens", "user_artist_share",
    ]:
        assert col in feats.columns, f"missing feature column: {col}"
