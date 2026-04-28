"""ALS → CatBoost Ranker full pipeline.

Steps:
  1. Temporal split → fit ALS on train split
  2. Generate 500 ALS candidates for eval users with val interactions
  3. Label candidates (1 = item in val ground truth)
  4. Add features: als_score, als_rank, item_pop, user_n_listens
  5. GroupShuffleSplit 80/20 → fit CatBoostRanker with early stopping
  6. Evaluate Recall@100 on val split
  7. Retrain ALS on full data → generate candidates for 10k users → rerank → submission

Usage:
    python scripts/train_ranker.py data=50m run_id=002
"""
import csv
import logging
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupShuffleSplit

from src.data.dataset import load_listens, positive_listens
from src.data.splits import temporal_split
from src.evaluation.metrics import recall_at_k
from src.models.als import ALSModel
from src.models.catboost_ranker import RankerModel
from src.utils import setup_logging

log = logging.getLogger(__name__)


def _load_users(users_csv: str) -> list[int]:
    return pl.read_csv(users_csv).get_column("uid").cast(pl.Int64).to_list()


def _ground_truth(df: pl.DataFrame, users: list[int]) -> pl.DataFrame:
    return (
        df.select(["uid", "item_id"])
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
        .filter(pl.col("uid").is_in(users))
        .unique()
    )


def _item_features(train: pl.DataFrame) -> pl.DataFrame:
    return (
        train
        .group_by("item_id")
        .agg(pl.len().alias("item_pop"))
        .with_columns(pl.col("item_id").cast(pl.Int64))
    )


def _user_features(train: pl.DataFrame) -> pl.DataFrame:
    return (
        train
        .group_by("uid")
        .agg(pl.len().alias("user_n_listens"))
        .with_columns(pl.col("uid").cast(pl.Int64))
    )


def _add_features(
    candidates: pl.DataFrame,
    train: pl.DataFrame,
) -> pl.DataFrame:
    item_feats = _item_features(train)
    user_feats = _user_features(train)
    return (
        candidates
        .join(item_feats, on="item_id", how="left")
        .join(user_feats, on="uid", how="left")
        .with_columns([
            pl.col("item_pop").fill_null(0).cast(pl.Int32),
            pl.col("user_n_listens").fill_null(0).cast(pl.Int32),
        ])
    )


def _label_candidates(candidates: pl.DataFrame, ground_truth: pl.DataFrame) -> pl.DataFrame:
    pos = ground_truth.with_columns(pl.lit(1).cast(pl.Int8).alias("label"))
    return (
        candidates
        .join(pos, on=["uid", "item_id"], how="left")
        .with_columns(pl.col("label").fill_null(0))
    )


def _split_for_ranker(df: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    pdf = df.to_pandas()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(pdf, groups=pdf["uid"]))
    return pl.from_pandas(pdf.iloc[train_idx]), pl.from_pandas(pdf.iloc[val_idx])


def _format_submission(preds: pl.DataFrame, top_k: int) -> pl.DataFrame:
    score_col = "ranker_score" if "ranker_score" in preds.columns else "score"
    return (
        preds
        .sort(["uid", score_col], descending=[False, True])
        .group_by("uid")
        .head(top_k)
        .group_by("uid")
        .agg(pl.col("item_id").cast(pl.Utf8).str.join(delimiter=" ").alias("item_ids"))
        .sort("uid")
    )


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

    eval_users = _load_users(cfg.data.users_csv)
    gt_val = _ground_truth(split.val, eval_users)
    log.info("val ground truth: %d (uid, item_id) pairs", len(gt_val))

    # ── 2. Fit ALS on train ───────────────────────────────────────────────────
    als = ALSModel(**cfg.als)
    als.fit(split.train)

    # ── 3. Generate candidates for val users ──────────────────────────────────
    val_users_with_gt = gt_val["uid"].unique().to_list()
    log.info("generating %d candidates for %d val users", cfg.als.n_cand, len(val_users_with_gt))
    candidates = als.recommend(users=val_users_with_gt, n=cfg.als.n_cand)
    log.info("candidates: %d rows", len(candidates))

    # ── 4. Label + features ───────────────────────────────────────────────────
    candidates = _label_candidates(candidates, gt_val)
    candidates = _add_features(candidates, split.train)

    pos_rate = candidates["label"].mean()
    log.info("label rate: %.4f (neg_ratio ~%d:1)", pos_rate, int(1 / pos_rate) if pos_rate > 0 else 0)

    # ── 5. Train ranker ───────────────────────────────────────────────────────
    df_train, df_val_ranker = _split_for_ranker(candidates, cfg.seed)
    log.info("ranker train=%d  val=%d", len(df_train), len(df_val_ranker))

    ranker = RankerModel(**cfg.ranker)
    ranker.fit(df_train, df_val_ranker)

    # ── 6. Evaluate val + test Recall@100 ────────────────────────────────────
    all_val_candidates = als.recommend(users=eval_users, n=cfg.als.n_cand)
    all_val_candidates = _add_features(all_val_candidates, split.train)

    preds_val = ranker.predict(all_val_candidates, n=cfg.top_k)
    score_val = recall_at_k(gt_val, preds_val, k=cfg.top_k)
    log.info("val  Recall@%d = %.2f", cfg.top_k, score_val)

    gt_test = _ground_truth(split.test, eval_users)
    preds_test = ranker.predict(all_val_candidates, n=cfg.top_k)
    score_test = recall_at_k(gt_test, preds_test, k=cfg.top_k)
    log.info("test Recall@%d = %.2f", cfg.top_k, score_test)

    # ── 7. Submission: retrain ALS on full data ───────────────────────────────
    log.info("retraining ALS on full data for submission")
    als_full = ALSModel(**cfg.als)
    als_full.fit(listens)

    sub_candidates = als_full.recommend(users=eval_users, n=cfg.als.n_cand)
    sub_candidates = _add_features(sub_candidates, listens)
    preds_sub = ranker.predict(sub_candidates, n=cfg.top_k)

    submission = _format_submission(preds_sub, top_k=cfg.top_k)
    log.info("submission rows: %d", len(submission))

    missing = set(eval_users) - set(submission["uid"].cast(pl.Int64).to_list())
    if missing:
        log.warning("%d eval users have no predictions (cold users)", len(missing))

    sub_dir = Path(cfg.submission_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(cfg.run_id)
    archive_path = sub_dir / f"sub_{run_id}_als_ranker.csv"
    submission.write_csv(archive_path)
    log.info("submission saved to %s", archive_path)
    submission.write_csv(sub_dir / "test.csv")

    # ── 8. Log results ────────────────────────────────────────────────────────
    results_path = Path(cfg.output_dir) / "results.csv"
    run_id_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for split_name, score in [("val", score_val), ("test", score_test)]:
        _append_results(results_path, {
            "run_id": run_id_ts,
            "model": "als_ranker",
            "dataset_size": cfg.data.size,
            "split": split_name,
            "score": round(score, 4),
            "config_path": "configs/ranker.yaml",
        })
    log.info("results appended to %s", results_path)


if __name__ == "__main__":
    main()
