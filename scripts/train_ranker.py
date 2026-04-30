"""Multi-CG → CatBoost Ranker training pipeline.

This script trains the ranker only — it does NOT generate a submission.
For submission, run ``scripts/submit_ranker.py`` with the same ``run_id``;
that script reuses CGs fitted on full data via the cache.

Steps:
  1. Load listens, temporal_split (val_days=7, gap_days=1).
  2. For each CG in cfg.candidate_generators: fit_or_load_cg (cache lookup).
  3. Generate ``cg.n_cand`` candidates per eval user from each CG.
  4. merge_candidates → cg_recall (upper bound for the ranker).
  5. add_features (LazyFrame-based: ~48 user/item/pair/cross features).
  6. Label vs val ground truth, GroupShuffleSplit 80/20, fit CatBoost YetiRank.
  7. Recall@100 on val + test → experiment-log.md + artifacts/results.csv.
  8. Pickle ranker → artifacts/ranker_{run_id}.pkl.
  9. Optional CatBoost feature importance → artifacts/feature_importance_{run_id}.csv.

Usage:
    python scripts/train_ranker.py data=50m run_id=006
"""
import csv
import logging
import pickle
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupShuffleSplit

from src.data.dataset import (
    effective_dislikes,
    load_dislikes,
    load_likes,
    load_listens,
    load_undislikes,
    positive_listens,
)
from src.data.features import add_features
from src.data.splits import temporal_split
from src.evaluation.metrics import recall_at_k
from src.inference.merge_candidates import cg_recall, merge_candidates
from src.inference.pipeline import (
    apply_exclude_filter,
    generate_candidates,
    load_eval_users,
)
from src.models.catboost_ranker import RankerModel
from src.training.cg_cache import fit_or_load_cg
from src.utils import setup_logging

log = logging.getLogger(__name__)


def _ground_truth(df: pl.DataFrame, users: list[int]) -> pl.DataFrame:
    return (
        df.select(["uid", "item_id"])
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
        .filter(pl.col("uid").is_in(users))
        .unique()
    )


def _label_candidates(candidates: pl.DataFrame, ground_truth: pl.DataFrame) -> pl.DataFrame:
    pos = ground_truth.with_columns(pl.lit(1).cast(pl.Int8).alias("label"))
    return (
        candidates
        .join(pos, on=["uid", "item_id"], how="left")
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    )


def _split_for_ranker(df: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    pdf = df.to_pandas()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(pdf, groups=pdf["uid"]))
    return pl.from_pandas(pdf.iloc[train_idx]), pl.from_pandas(pdf.iloc[val_idx])


def _append_results(path: Path, row: dict) -> None:
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@hydra.main(config_path="../configs", config_name="ranker", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.seed)

    # ── 1. Load data & split ──────────────────────────────────────────────────
    log.info("loading listens from %s", cfg.data.listens)
    listens = positive_listens(load_listens(path=cfg.data.listens))
    log.info("positive listens: %d rows", len(listens))

    split = temporal_split(
        listens,
        val_days=cfg.split.val_days,
        gap_days=cfg.split.gap_days,
        timestamp_col=cfg.split.timestamp_col,
    )
    log.info("train=%d  val=%d  test=%d", len(split.train), len(split.val), len(split.test))

    eval_users = load_eval_users(cfg.data.users_csv)
    gt_val = _ground_truth(split.val, eval_users)
    gt_test = _ground_truth(split.test, eval_users)
    log.info(
        "val ground truth: %d pairs / %d users; test: %d pairs / %d users",
        len(gt_val), gt_val["uid"].n_unique(),
        len(gt_test), gt_test["uid"].n_unique(),
    )

    # ── 2. Fit / load each CG ────────────────────────────────────────────────
    # Some CGs (e.g. RecentLikesModel) train on likes, not listens.
    # cg_cfg.data_source picks which source they get; default = listens.
    train_max_ts = float(split.train["timestamp"].max())
    likes_train = (
        load_likes(path=cfg.data.likes)
        .filter(pl.col("timestamp") <= train_max_ts)
    )
    log.info("likes (train period): %d rows", len(likes_train))

    data_sources = {
        "listens": split.train,
        "likes": likes_train,
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
            suffix="",
            force_refit=cfg.force_refit_cg,
        )
        cgs.append(cg)

    # ── 3. Generate candidates for users with val ground truth ───────────────
    val_users_with_gt = gt_val["uid"].unique().to_list()
    cg_dfs = generate_candidates(cgs, val_users_with_gt)

    # ── 4. Merge → optional dislike filter → cg_recall ───────────────────────
    merged = merge_candidates(cg_dfs)

    if cfg.filter_dislikes:
        # Use only events recorded up to the train cutoff (computed above)
        # to avoid leaking validation-period information into the offline eval.
        dislikes_train = (
            load_dislikes(path=cfg.data.dislikes)
            .filter(pl.col("timestamp") <= train_max_ts)
        )
        undislikes_train = (
            load_undislikes(path=cfg.data.undislikes)
            .filter(pl.col("timestamp") <= train_max_ts)
        )
        active_dislikes = effective_dislikes(dislikes_train, undislikes_train)
        log.info(
            "effective dislikes (train period): %d active / %d raw / %d undislikes",
            len(active_dislikes), len(dislikes_train), len(undislikes_train),
        )
        before = len(merged)
        merged = apply_exclude_filter(merged, active_dislikes)
        log.info(
            "dislike filter: dropped %d / %d candidate rows",
            before - len(merged), before,
        )

    upper_bound = cg_recall(merged, gt_val)
    log.info("CG-recall@∞ on val (upper bound for ranker, ×1000 scale): %.2f", upper_bound)

    # ── 5. Label + features (LazyFrame, train period only via cutoff_ts) ─────
    labeled = _label_candidates(merged, gt_val)
    pos_rate = float(labeled["label"].mean())
    log.info(
        "label rate: %.4f (neg_ratio ~%d:1, %d rows)",
        pos_rate, int(1 / pos_rate) if pos_rate > 0 else 0, len(labeled),
    )

    cutoff_ts = int(train_max_ts)
    listens_lf = pl.scan_parquet(cfg.data.listens)
    likes_lf = pl.scan_parquet(cfg.data.likes)
    dislikes_lf = pl.scan_parquet(cfg.data.dislikes)
    unlikes_lf = pl.scan_parquet(cfg.data.unlikes)
    undislikes_lf = pl.scan_parquet(cfg.data.undislikes)
    artist_map_lf = pl.scan_parquet(cfg.data.artist_item_mapping)
    album_map_lf = pl.scan_parquet(cfg.data.album_item_mapping)

    log.info("computing features (LazyFrame, cutoff_ts=%d)", cutoff_ts)
    features_lf = add_features(
        labeled.lazy(),
        listens_lf=listens_lf,
        likes_lf=likes_lf,
        dislikes_lf=dislikes_lf,
        unlikes_lf=unlikes_lf,
        undislikes_lf=undislikes_lf,
        artist_map_lf=artist_map_lf,
        album_map_lf=album_map_lf,
        cutoff_ts=cutoff_ts,
    )

    if cfg.cache_features:
        features_dir = Path(cfg.features_dir)
        features_dir.mkdir(parents=True, exist_ok=True)
        cache_path = features_dir / f"{cfg.run_id}_train.parquet"
        log.info("caching labeled features → %s", cache_path)
        features_lf.sink_parquet(str(cache_path), compression="zstd")
        labeled = pl.scan_parquet(str(cache_path)).collect()
    else:
        labeled = features_lf.collect()
    log.info("labeled+features: %d rows × %d cols", len(labeled), len(labeled.columns))

    # ── 6. Train ranker ──────────────────────────────────────────────────────
    df_train, df_val_ranker = _split_for_ranker(labeled, cfg.seed)
    log.info("ranker train=%d  val=%d", len(df_train), len(df_val_ranker))

    ranker = RankerModel(**cfg.ranker)
    ranker.fit(df_train, df_val_ranker)

    # ── 7. Eval Recall@100 on val + test ─────────────────────────────────────
    cg_dfs_full = generate_candidates(cgs, eval_users)
    merged_full = merge_candidates(cg_dfs_full)
    if cfg.filter_dislikes:
        merged_full = apply_exclude_filter(merged_full, active_dislikes)
    log.info("computing features for full eval set")
    feats_full_lf = add_features(
        merged_full.lazy(),
        listens_lf=listens_lf,
        likes_lf=likes_lf,
        dislikes_lf=dislikes_lf,
        unlikes_lf=unlikes_lf,
        undislikes_lf=undislikes_lf,
        artist_map_lf=artist_map_lf,
        album_map_lf=album_map_lf,
        cutoff_ts=cutoff_ts,
    )
    if cfg.cache_features:
        features_dir = Path(cfg.features_dir)
        features_dir.mkdir(parents=True, exist_ok=True)
        eval_cache_path = features_dir / f"{cfg.run_id}_eval.parquet"
        log.info("caching eval features → %s", eval_cache_path)
        feats_full_lf.sink_parquet(str(eval_cache_path), compression="zstd")
        feats_full = pl.scan_parquet(str(eval_cache_path)).collect()
    else:
        feats_full = feats_full_lf.collect()
    log.info("full eval features: %d rows × %d cols", len(feats_full), len(feats_full.columns))

    preds_val = ranker.predict(feats_full, n=cfg.top_k)
    score_val = recall_at_k(gt_val, preds_val, k=cfg.top_k)
    log.info("val  Recall@%d = %.2f", cfg.top_k, score_val)

    preds_test = ranker.predict(feats_full, n=cfg.top_k)
    score_test = recall_at_k(gt_test, preds_test, k=cfg.top_k)
    log.info("test Recall@%d = %.2f", cfg.top_k, score_test)

    # ── 8. Persist ranker + log results ──────────────────────────────────────
    ranker_dir = Path(cfg.ranker_dir)
    ranker_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(cfg.run_id)
    ranker_path = ranker_dir / f"ranker_{run_id}.pkl"
    with open(ranker_path, "wb") as f:
        pickle.dump(ranker, f)
    log.info("ranker saved to %s", ranker_path)

    results_path = Path(cfg.output_dir) / "results.csv"
    run_id_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cg_names = ",".join(cg.name for cg in cgs)
    for split_name, score in [("val", score_val), ("test", score_test)]:
        _append_results(results_path, {
            "run_id": run_id_ts,
            "model": f"ranker[{cg_names}]",
            "dataset_size": cfg.data.size,
            "split": split_name,
            "score": round(score, 4),
            "config_path": "configs/ranker.yaml",
        })
    log.info("results appended to %s", results_path)

    # ── 9. Optional feature importance ───────────────────────────────────────
    if cfg.compute_feature_importance:
        try:
            fi = ranker.feature_importance(prettified=True)
            fi_path = Path(cfg.output_dir) / f"feature_importance_{run_id}.csv"
            fi.to_csv(fi_path, index=False)
            log.info("feature importance saved to %s", fi_path)
            log.info("top-10 features:\n%s", fi.head(10).to_string(index=False))
        except Exception as e:
            log.warning("feature_importance failed: %s", e)

    log.info(
        "DONE. run_id=%s val=%.2f test=%.2f cg_recall=%.4f. "
        "Run submit_ranker.py to generate the submission CSV.",
        run_id, score_val, score_test, upper_bound,
    )


if __name__ == "__main__":
    main()
