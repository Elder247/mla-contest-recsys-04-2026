"""Generate a submission from a trained ranker.

Reuses the ranker pickle from a prior ``train_ranker.py`` run (matched by
``run_id``) and either reuses or fits each CG on the *full* listens data
(suffix ``_full`` in the cg cache).

Steps:
  1. Load full listens (no split).
  2. Load ranker from artifacts/ranker_{run_id}.pkl.
  3. For each CG: fit_or_load_cg(..., suffix="_full") on full data.
  4. Generate candidates for all 10k eval users → merge.
  5. Add features (same as train_ranker — basic features in A1).
  6. ranker.predict → top-100 → submission CSV.

Usage:
    python scripts/submit_ranker.py data=50m run_id=003
"""
import logging
import pickle
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import (
    effective_dislikes,
    load_dislikes,
    load_likes,
    load_listens,
    load_undislikes,
    positive_listens,
)
from src.data.features import add_features
from src.inference.merge_candidates import merge_candidates
from src.inference.pipeline import (
    apply_exclude_filter,
    generate_candidates,
    load_eval_users,
)
from src.training.cg_cache import fit_or_load_cg
from src.utils import setup_logging

log = logging.getLogger(__name__)


def _format_submission(preds: pl.DataFrame, top_k: int) -> pl.DataFrame:
    score_col = "ranker_score" if "ranker_score" in preds.columns else "score"
    return (
        preds
        .sort(["uid", score_col], descending=[False, True])
        .group_by("uid", maintain_order=True)
        .head(top_k)
        .group_by("uid")
        .agg(pl.col("item_id").cast(pl.Utf8).str.join(delimiter=" ").alias("item_ids"))
        .sort("uid")
    )


@hydra.main(config_path="../configs", config_name="submit_ranker", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    # ── 1. Load full data ────────────────────────────────────────────────────
    log.info("loading listens from %s", cfg.data.listens)
    listens = positive_listens(load_listens(path=cfg.data.listens))
    log.info("positive listens (full): %d rows", len(listens))

    eval_users = load_eval_users(cfg.data.users_csv)
    log.info("eval users: %d", len(eval_users))

    # ── 2. Load trained ranker ───────────────────────────────────────────────
    run_id = str(cfg.run_id)
    ranker_path = Path(cfg.ranker_dir) / f"ranker_{run_id}.pkl"
    log.info("loading ranker from %s", ranker_path)
    with open(ranker_path, "rb") as f:
        ranker = pickle.load(f)

    # ── 3. Fit / load each CG on FULL data (suffix="_full") ──────────────────
    # Route by data_source: listens (default) or likes.
    likes = load_likes(path=cfg.data.likes)
    log.info("likes (full): %d rows", len(likes))

    data_sources = {
        "listens": listens,
        "likes": likes,
    }

    cgs = []
    for cg_cfg in cfg.candidate_generators:
        source_name = cg_cfg.get("data_source", "listens")
        if source_name not in data_sources:
            raise ValueError(
                f"unknown data_source '{source_name}' for CG '{cg_cfg.get('name')}'; "
                f"valid: {list(data_sources)}"
            )
        cg = fit_or_load_cg(
            cg_cfg,
            data_sources[source_name],
            size=cfg.data.size,
            suffix="_full",
            force_refit=cfg.force_refit_cg,
            cache_dir=cfg.cg_cache_dir,
        )
        cgs.append(cg)

    # ── 4. Generate candidates → merge → optional dislike filter ────────────
    cg_dfs = generate_candidates(cgs, eval_users)
    merged = merge_candidates(cg_dfs)

    if cfg.filter_dislikes:
        # Submission uses the full event tables — at inference time we know
        # every dislike (and every undislike that overrides it).
        dislikes = load_dislikes(path=cfg.data.dislikes)
        undislikes = load_undislikes(path=cfg.data.undislikes)
        active_dislikes = effective_dislikes(dislikes, undislikes)
        log.info(
            "effective dislikes (full): %d active / %d raw / %d undislikes",
            len(active_dislikes), len(dislikes), len(undislikes),
        )
        before = len(merged)
        merged = apply_exclude_filter(merged, active_dislikes)
        log.info(
            "dislike filter: dropped %d / %d candidate rows",
            before - len(merged), before,
        )

    # ── 5. Features (LazyFrame, full data — cutoff_ts = max ts) ─────────────
    listens_lf = pl.scan_parquet(cfg.data.listens)
    likes_lf = pl.scan_parquet(cfg.data.likes)
    dislikes_lf = pl.scan_parquet(cfg.data.dislikes)
    unlikes_lf = pl.scan_parquet(cfg.data.unlikes)
    undislikes_lf = pl.scan_parquet(cfg.data.undislikes)
    artist_map_lf = pl.scan_parquet(cfg.data.artist_item_mapping)
    album_map_lf = pl.scan_parquet(cfg.data.album_item_mapping)

    cutoff_ts = int(listens["timestamp"].max())
    log.info("computing features (LazyFrame, cutoff_ts=%d, full data)", cutoff_ts)
    feats_lf = add_features(
        merged.lazy(),
        listens_lf=listens_lf,
        likes_lf=likes_lf,
        dislikes_lf=dislikes_lf,
        unlikes_lf=unlikes_lf,
        undislikes_lf=undislikes_lf,
        artist_map_lf=artist_map_lf,
        album_map_lf=album_map_lf,
        cutoff_ts=cutoff_ts,
        embeddings_path=cfg.data.embeddings if cfg.get("enable_embed_features", True) else None,
    )
    feats = feats_lf.collect()
    log.info("submission features: %d rows × %d cols", len(feats), len(feats.columns))

    # ── 6. Rank + format submission ──────────────────────────────────────────
    preds = ranker.predict(feats, n=cfg.top_k)
    submission = _format_submission(preds, top_k=cfg.top_k)
    log.info("submission rows: %d", len(submission))

    missing = set(eval_users) - set(submission["uid"].cast(pl.Int64).to_list())
    if missing:
        log.warning("%d eval users have no predictions (cold users)", len(missing))

    sub_dir = Path(cfg.submission_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    archive_path = sub_dir / f"sub_{run_id}_{cfg.submission_name}.csv"
    submission.write_csv(archive_path)
    submission.write_csv(sub_dir / "test.csv")
    log.info("submission saved to %s and %s", archive_path, sub_dir / "test.csv")


if __name__ == "__main__":
    main()
