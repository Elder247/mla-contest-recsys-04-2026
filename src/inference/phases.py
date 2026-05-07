"""Pipeline phase functions shared between train_ranker and submit_ranker.

Each phase is intentionally a pure function with a small, well-typed surface
so it can be invoked in either an in-process call or a subprocess. The
subprocess pattern is the production path on 500m / 5B because the OS
reclaims the Polars/Arrow allocator's RSS at process exit — without that the
fit / generate stages leave 30-70 GB of high-water-mark RSS that blocks the
features stage from completing on a 120 GB box.

Phase contract:

- ``fit_phase``      — eagerly load listens (filtered by cutoff if given) and
  likes, then ``fit_or_load_cg`` for each candidate generator. Side-effect:
  the CG cache pickles under ``{cache_dir}/{name}_{size}{suffix}.pkl``.
- ``generate_phase`` — for each CG: ``pickle.load`` → ``recommend`` → write
  ``cg_{name}.parquet`` → drop the CG before loading the next. Then merge
  from disk via ``merge_candidates`` and write ``merged.parquet``. Optional
  dislike anti-join is applied before the write.
- ``features_phase`` — ``scan_parquet`` the merged candidates, optionally
  label-join against a ground-truth parquet, run :func:`add_features`, and
  ``write_parquet`` the result. Materialisation is plain ``collect()``: the
  embed sub-pipeline already calls ``collect`` internally, so a streaming
  sink would partially fall back. Subprocess isolation is what makes the
  peak acceptable, not streaming.

All three accept Hydra ``DictConfig`` or plain dicts for the CG list — they
read items via ``cg_cfg.get(key)`` which both types support.
"""
from __future__ import annotations

import gc
import logging
import pickle
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from src.data.dataset import (
    effective_dislikes,
    load_dislikes,
    load_likes,
    load_listens,
    load_undislikes,
    positive_listens,
)
from src.data.features import add_features
from src.inference.merge_candidates import apply_n_cand_keep, merge_candidates
from src.inference.pipeline import apply_exclude_filter, load_eval_users
from src.training.cg_cache import cg_cache_path, fit_or_load_cg

log = logging.getLogger(__name__)

CGConfigList = Iterable[Any]  # list[DictConfig] | list[dict]


# ---------------------------------------------------------------------------
# Phase 1 — fit candidate generators
# ---------------------------------------------------------------------------

def fit_phase(
    cg_cfg_list: CGConfigList,
    listens_path: str,
    likes_path: str,
    cache_dir: str | Path,
    size: str,
    suffix: str = "",
    force_refit: bool = False,
    train_cutoff_ts: int | None = None,
) -> None:
    """Fit (or load from cache) every CG in ``cg_cfg_list``.

    ``train_cutoff_ts`` is the inclusive upper bound for training data.
    When given, listens are filtered to ``timestamp < train_cutoff_ts`` (and
    ``played_ratio_pct > 50``); likes are filtered to
    ``timestamp <= train_cutoff_ts``. When ``None``, the full event tables
    are used (the submit_ranker pattern, "_full" suffix path).
    """
    log.info(
        "fit_phase: cutoff=%s suffix=%s force_refit=%s",
        train_cutoff_ts, suffix, force_refit,
    )

    if train_cutoff_ts is not None:
        listens = (
            pl.scan_parquet(listens_path)
            .filter(pl.col("played_ratio_pct") > 50)
            .filter(pl.col("timestamp") < train_cutoff_ts)
            .collect()
        )
        likes = (
            pl.scan_parquet(likes_path)
            .filter(pl.col("timestamp") <= train_cutoff_ts)
            .collect()
        )
    else:
        listens = positive_listens(load_listens(path=listens_path))
        likes = load_likes(path=likes_path)

    log.info(
        "fit_phase: listens=%d likes=%d (post-filter)",
        len(listens), len(likes),
    )

    data_sources = {"listens": listens, "likes": likes}

    for cg_cfg in cg_cfg_list:
        source_name = cg_cfg.get("data_source", "listens")
        if source_name not in data_sources:
            raise ValueError(
                f"unknown data_source '{source_name}' for CG "
                f"'{cg_cfg.get('name')}'; valid: {list(data_sources)}"
            )
        # fit_or_load_cg pickles the fitted model itself; we discard the
        # returned reference so it goes out of scope before the next CG.
        cg = fit_or_load_cg(
            cg_cfg,
            data_sources[source_name],
            size=size,
            suffix=suffix,
            force_refit=force_refit,
            cache_dir=cache_dir,
        )
        del cg
        gc.collect()

    log.info("fit_phase: done")


# ---------------------------------------------------------------------------
# Phase 2 — generate candidates and merge
# ---------------------------------------------------------------------------

def _load_cg_from_cache(
    name: str,
    size: str,
    suffix: str,
    cache_dir: str | Path,
):
    """Strict cache-only loader. Raises if the pickle is missing."""
    path = cg_cache_path(name, size, suffix, cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"CG cache miss for '{name}' at {path}. "
            f"Run fit_phase before generate_phase."
        )
    log.info("loading CG '%s' from %s", name, path)
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_phase(
    cg_cfg_list: CGConfigList,
    eval_users: list[int],
    cache_dir: str | Path,
    size: str,
    suffix: str,
    output_dir: str | Path,
    *,
    dislikes_path: str | None = None,
    undislikes_path: str | None = None,
    dislike_cutoff_ts: int | None = None,
    filter_dislikes: bool = False,
) -> Path:
    """Generate candidates per CG, merge, and write ``merged.parquet``.

    Loads one CG at a time from its pickle cache so peak memory stays
    bounded by the largest CG plus its recommend-buffer (typically <20 GB
    even on 500m).

    Returns:
        Path to the merged parquet (``output_dir/merged.parquet``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cg_paths: list[tuple[str, Path]] = []
    for cg_cfg in cg_cfg_list:
        name = cg_cfg.get("name")
        cg = _load_cg_from_cache(name, size, suffix, cache_dir)
        n_cand = cg_cfg.get("n_cand", getattr(cg, "n_cand", 100))
        log.info(
            "CG '%s': generating top-%d for %d users",
            name, n_cand, len(eval_users),
        )
        df = cg.recommend(eval_users, n=n_cand)
        log.info("CG '%s': %d candidate rows", name, len(df))
        out_path = output_dir / f"cg_{name}.parquet"
        df.write_parquet(str(out_path), compression="zstd")
        cg_paths.append((name, out_path))
        del cg, df
        gc.collect()

    # Re-read each per-CG parquet for the merge. Each is small (<200 MB on
    # 500m) so eager read is fine and merge_candidates expects DataFrames.
    cg_dfs = {name: pl.read_parquet(str(p)) for name, p in cg_paths}
    merged = merge_candidates(cg_dfs)
    del cg_dfs
    gc.collect()

    # Optional post-merge row filter — see apply_n_cand_keep docstring.
    # No-op when no CG block has the ``n_cand_keep`` field set.
    merged = apply_n_cand_keep(merged, cg_cfg_list)
    gc.collect()

    if filter_dislikes:
        if dislikes_path is None or undislikes_path is None:
            raise ValueError(
                "filter_dislikes=True requires both dislikes_path and "
                "undislikes_path"
            )
        dislikes = load_dislikes(path=dislikes_path)
        undislikes = load_undislikes(path=undislikes_path)
        if dislike_cutoff_ts is not None:
            dislikes = dislikes.filter(pl.col("timestamp") <= dislike_cutoff_ts)
            undislikes = undislikes.filter(pl.col("timestamp") <= dislike_cutoff_ts)
        active_dislikes = effective_dislikes(dislikes, undislikes)
        log.info(
            "generate_phase: effective dislikes %d active / %d raw / %d undislikes",
            len(active_dislikes), len(dislikes), len(undislikes),
        )
        before = len(merged)
        merged = apply_exclude_filter(merged, active_dislikes)
        log.info(
            "generate_phase: dislike filter dropped %d / %d rows",
            before - len(merged), before,
        )
        del dislikes, undislikes, active_dislikes
        gc.collect()

    merged_path = output_dir / "merged.parquet"
    merged.write_parquet(str(merged_path), compression="zstd")
    log.info(
        "generate_phase: merged %d rows → %s",
        len(merged), merged_path,
    )
    return merged_path


# ---------------------------------------------------------------------------
# Phase 3 — features
# ---------------------------------------------------------------------------

def features_phase(
    merged_path: str | Path,
    listens_path: str,
    likes_path: str,
    dislikes_path: str,
    unlikes_path: str,
    undislikes_path: str,
    artist_map_path: str,
    album_map_path: str,
    cutoff_ts: int,
    output_path: str | Path,
    *,
    embeddings_path: str | None = None,
    label_gt_path: str | Path | None = None,
) -> Path:
    """Compute the full feature table for a candidate parquet.

    When ``label_gt_path`` is given, a ``label`` Int8 column is appended
    (1 if the (uid, item_id) appears in the GT parquet, 0 otherwise) before
    feature joins, so the resulting parquet is directly usable as a ranker
    training table.

    Returns:
        Path to the features parquet.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates_lf = pl.scan_parquet(str(merged_path))

    if label_gt_path is not None:
        gt_lf = (
            pl.scan_parquet(str(label_gt_path))
            .select(["uid", "item_id"])
            .with_columns([
                pl.col("uid").cast(pl.Int64),
                pl.col("item_id").cast(pl.Int64),
                pl.lit(1, dtype=pl.Int8).alias("label"),
            ])
        )
        candidates_lf = (
            candidates_lf
            .with_columns([
                pl.col("uid").cast(pl.Int64),
                pl.col("item_id").cast(pl.Int64),
            ])
            .join(gt_lf, on=["uid", "item_id"], how="left")
            .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
        )

    listens_lf = pl.scan_parquet(listens_path)
    likes_lf = pl.scan_parquet(likes_path)
    dislikes_lf = pl.scan_parquet(dislikes_path)
    unlikes_lf = pl.scan_parquet(unlikes_path)
    undislikes_lf = pl.scan_parquet(undislikes_path)
    artist_map_lf = pl.scan_parquet(artist_map_path)
    album_map_lf = pl.scan_parquet(album_map_path)

    log.info(
        "features_phase: cutoff_ts=%d label=%s embeddings=%s",
        cutoff_ts, label_gt_path is not None, embeddings_path is not None,
    )
    features_lf = add_features(
        candidates_lf,
        listens_lf=listens_lf,
        likes_lf=likes_lf,
        dislikes_lf=dislikes_lf,
        unlikes_lf=unlikes_lf,
        undislikes_lf=undislikes_lf,
        artist_map_lf=artist_map_lf,
        album_map_lf=album_map_lf,
        cutoff_ts=cutoff_ts,
        embeddings_path=embeddings_path,
    )

    features = features_lf.collect()
    log.info(
        "features_phase: %d rows × %d cols → %s",
        len(features), len(features.columns), output_path,
    )
    features.write_parquet(str(output_path), compression="zstd")
    return output_path


# ---------------------------------------------------------------------------
# Helpers used by orchestrators (not phase functions themselves)
# ---------------------------------------------------------------------------

ONE_DAY_TS = 17_280


def derive_split_metadata(
    listens_path: str,
    val_days: int,
    gap_days: int,
    timestamp_col: str = "timestamp",
) -> dict:
    """Compute temporal-split boundaries from listens metadata only.

    Avoids materialising the full listens table in the orchestrator. Polars
    pushes the ``max`` aggregation into the parquet reader.
    """
    t_max = int(
        pl.scan_parquet(listens_path)
        .select(pl.col(timestamp_col).max())
        .collect()
        .row(0)[0]
    )
    upd = ONE_DAY_TS
    t_end = t_max + 1
    val_start = t_end - (val_days + gap_days) * upd
    test_start = val_start + gap_days * upd
    val_end = val_start + val_days * upd
    return {
        "t_max": t_max,
        "t_end": t_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        # Inclusive max of the train period (matches the legacy
        # ``train_max_ts = float(split.train["timestamp"].max())``).
        "train_max_ts": val_start - 1,
    }


def write_ground_truth(
    listens_path: str,
    eval_users: list[int],
    lower_ts_inclusive: int,
    upper_ts_exclusive: int,
    output_path: str | Path,
) -> Path:
    """Save (uid, item_id) for eval users in [lower, upper) as parquet.

    Uses lazy scan so the orchestrator never holds the full listens table
    in memory.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gt = (
        pl.scan_parquet(listens_path)
        .filter(pl.col("played_ratio_pct") > 50)
        .filter(pl.col("timestamp") >= lower_ts_inclusive)
        .filter(pl.col("timestamp") < upper_ts_exclusive)
        .filter(pl.col("uid").is_in(eval_users))
        .select(["uid", "item_id"])
        .with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
        .unique()
        .collect()
    )
    gt.write_parquet(str(output_path), compression="zstd")
    log.info("ground truth: %d pairs → %s", len(gt), output_path)
    return output_path


def load_eval_users_from_csv(users_csv: str) -> list[int]:
    """Re-export to keep all phase-related helpers in one module."""
    return load_eval_users(users_csv)
