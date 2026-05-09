"""Refit LGBM + CatBoost from cached features (no CG / candidate / feature rebuild).

Re-entry point for ``scripts/train_ranker.py`` phases 5-7 only. Use when:
- A bug in LGBM hyperparams produced a degenerate model (e.g. ``best_iter=1``)
  and you want to refit cheaply without redoing the multi-hour features build.
- You changed only ``ranker.yaml`` CatBoost params and want to reuse the
  expensive cascade input.

Pre-condition (already produced by a prior ``train_ranker.py`` run with the
same ``run_id``):

    {features_dir}/{run_id}_train.parquet         labeled features (with ``label``)
    {features_dir}/{run_id}_eval.parquet          unlabeled eval features
    {gt_dir}/{run_id}/gt_val.parquet               val window ground truth
    {gt_dir}/{run_id}/gt_test.parquet              test window ground truth

Usage:
    python -u scripts/refit_ranker.py data=500m run_id=v4_features \\
        2>&1 | tee /tmp/refit_v4_features.log

    # Or via an Optuna top-K overlay (uses its CatBoost params, but the
    # cached features must have been built by a *compatible* train_ranker.py
    # run — i.e. same n_cand and same CG set).
    python -u scripts/refit_ranker.py --config-name=ranker_v4_top1 \\
        data=500m run_id=v4_top1

Outputs:

    artifacts/lgbm_{run_id}.pkl                    new LGBM model
    {features_dir}/{run_id}_train_lgbm.parquet     refreshed LGBM scores
    {features_dir}/{run_id}_eval_lgbm.parquet      refreshed LGBM scores
    artifacts/ranker_{run_id}.pkl                  new CatBoost model
    artifacts/results.csv                           appended val/test rows

The cascade cuts (``n_ranker_train`` / ``n_ranker_eval``) come from the
loaded config — same defaults / overlays as ``train_ranker.py``.

Stage-1 aligned with ``train_ranker.py``: when ``lgbm_oof_folds >= 2``,
labeled rows get out-of-fold ``lgbm_score``; the saved pickle is the
final 80/20 fit used only for eval-feature scoring.
"""
from __future__ import annotations

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

from src.evaluation.metrics import recall_at_k
from src.models.catboost_ranker import RankerModel
from src.models.lightgbm_ranker import LightGBMRanker
from src.utils import setup_logging

log = logging.getLogger(__name__)


def _split_for_ranker(df: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    uids = df["uid"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(np.zeros(len(df)), groups=uids))
    return df[train_idx], df[val_idx]


def _cascade_cut(df_feat: pl.DataFrame, df_lgbm: pl.DataFrame, n: int) -> pl.DataFrame:
    return (
        df_feat
        .join(df_lgbm, on=["uid", "item_id"], how="left")
        .sort(["uid", "lgbm_score"], descending=[False, True])
        .group_by("uid", maintain_order=True)
        .head(n)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("uid").cast(pl.Int32).alias("lgbm_rank")
        )
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

    run_id = str(cfg.run_id)
    features_dir = Path(cfg.features_dir)
    feats_train_path = features_dir / f"{run_id}_train.parquet"
    feats_eval_path = features_dir / f"{run_id}_eval.parquet"
    gt_dir = Path(cfg.gt_dir) / run_id
    gt_val_path = gt_dir / "gt_val.parquet"
    gt_test_path = gt_dir / "gt_test.parquet"

    for p in (feats_train_path, feats_eval_path, gt_val_path, gt_test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"missing prerequisite: {p}\n"
                f"Run scripts/train_ranker.py with run_id={run_id} first to "
                f"build the features + ground-truth caches."
            )

    # ── 1. Refit LGBM on cached labeled features ─────────────────────────────
    log.info("loading labeled train features ← %s", feats_train_path)
    labeled_full = pl.read_parquet(feats_train_path)
    pos_rate = float(labeled_full["label"].mean())
    log.info(
        "labeled features: %d rows × %d cols | label rate: %.4f (neg_ratio ~%d:1)",
        len(labeled_full), len(labeled_full.columns), pos_rate,
        int(1 / pos_rate) if pos_rate > 0 else 0,
    )

    df_train_lgbm, df_val_lgbm = _split_for_ranker(labeled_full, seed=cfg.seed)
    log.info(
        "LGBM ref split: train=%d  val=%d",
        len(df_train_lgbm), len(df_val_lgbm),
    )

    lgbm = LightGBMRanker()
    n_oof = int(cfg.get("lgbm_oof_folds", 5))

    if n_oof >= 2:
        log.info(
            "LGBM OOF: n_folds=%d — leak-free lgbm_score on labeled rows",
            n_oof,
        )
        oof_scores = lgbm.fit_oof(labeled_full, n_folds=n_oof, seed=cfg.seed)
        labeled_lgbm = (
            labeled_full
            .select(["uid", "item_id"])
            .join(oof_scores, on=["uid", "item_id"], how="inner")
        )
        if len(labeled_lgbm) != len(labeled_full):
            raise RuntimeError(
                "OOF join row mismatch — duplicate (uid, item_id) in labeled?",
            )
    else:
        log.info(
            "LGBM OOF disabled (lgbm_oof_folds=%s): single-model labeled scores",
            n_oof,
        )

    lgbm.fit(df_train_lgbm, df_val_lgbm)

    if n_oof < 2:
        labeled_lgbm = lgbm.score(labeled_full)

    ranker_dir = Path(cfg.ranker_dir)
    ranker_dir.mkdir(parents=True, exist_ok=True)
    lgbm_path = ranker_dir / f"lgbm_{run_id}.pkl"
    with open(lgbm_path, "wb") as f:
        pickle.dump(lgbm, f)
    log.info("LGBM saved to %s", lgbm_path)

    # ── 2. Re-score eval pool; labeled scores from OOF or legacy above ──────
    labeled_lgbm_path = features_dir / f"{run_id}_train_lgbm.parquet"
    labeled_lgbm.write_parquet(labeled_lgbm_path, compression="zstd")

    log.info("loading eval features ← %s", feats_eval_path)
    eval_full = pl.read_parquet(feats_eval_path)
    log.info(
        "eval features: %d rows × %d cols",
        len(eval_full), len(eval_full.columns),
    )
    eval_lgbm = lgbm.score(eval_full)
    eval_lgbm_path = features_dir / f"{run_id}_eval_lgbm.parquet"
    eval_lgbm.write_parquet(eval_lgbm_path, compression="zstd")
    log.info(
        "LGBM scores refreshed: %s, %s", labeled_lgbm_path, eval_lgbm_path,
    )

    # ── 3. Cascade cuts ──────────────────────────────────────────────────────
    n_ranker_train = int(cfg.get("n_ranker_train", 1023))
    n_ranker_eval = int(cfg.get("n_ranker_eval", 1500))
    log.info(
        "cascade: train top-%d / eval top-%d per user (LGBM stage-1)",
        n_ranker_train, n_ranker_eval,
    )
    labeled_full = _cascade_cut(labeled_full, labeled_lgbm, n_ranker_train)
    eval_full_cut = _cascade_cut(eval_full, eval_lgbm, n_ranker_eval)
    log.info(
        "after cascade: labeled=%d rows, eval=%d rows",
        len(labeled_full), len(eval_full_cut),
    )
    del eval_full, labeled_lgbm, eval_lgbm

    # ── 4. CatBoost stage-2 fit on cascaded labeled ──────────────────────────
    df_train, df_val_ranker = _split_for_ranker(labeled_full, cfg.seed)
    log.info("ranker train=%d  val=%d", len(df_train), len(df_val_ranker))
    del labeled_full

    ranker = RankerModel(**cfg.ranker)
    ranker.fit(df_train, df_val_ranker)
    del df_train, df_val_ranker

    # ── 5. Eval Recall@100 on val + test ─────────────────────────────────────
    gt_val = pl.read_parquet(gt_val_path)
    gt_test = pl.read_parquet(gt_test_path)
    log.info(
        "val ground truth: %d pairs / %d users; test: %d pairs / %d users",
        len(gt_val), gt_val["uid"].n_unique(),
        len(gt_test), gt_test["uid"].n_unique(),
    )

    preds = ranker.predict(eval_full_cut, n=cfg.top_k)
    score_val = recall_at_k(gt_val, preds, k=cfg.top_k)
    log.info("val  Recall@%d = %.2f", cfg.top_k, score_val)
    score_test = recall_at_k(gt_test, preds, k=cfg.top_k)
    log.info("test Recall@%d = %.2f", cfg.top_k, score_test)

    # ── 6. Persist ranker + log results ──────────────────────────────────────
    ranker_path = ranker_dir / f"ranker_{run_id}.pkl"
    with open(ranker_path, "wb") as f:
        pickle.dump(ranker, f)
    log.info("ranker saved to %s", ranker_path)

    cg_names = ",".join(c.get("name") for c in cfg.candidate_generators)
    results_path = Path(cfg.output_dir) / "results.csv"
    run_id_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for split_name, score in [("val", score_val), ("test", score_test)]:
        _append_results(results_path, {
            "run_id": run_id_ts,
            "model": f"ranker_refit[{cg_names}]",
            "dataset_size": cfg.data.size,
            "split": split_name,
            "score": round(score, 4),
            "config_path": "configs/ranker.yaml",
        })
    log.info("results appended to %s", results_path)

    # ── 7. Optional feature importance ───────────────────────────────────────
    if cfg.get("compute_feature_importance", True):
        try:
            fi = ranker.feature_importance(prettified=True)
            fi_path = Path(cfg.output_dir) / f"feature_importance_{run_id}.csv"
            fi.to_csv(fi_path, index=False)
            log.info("feature importance saved to %s", fi_path)
            log.info("top-10 features:\n%s", fi.head(10).to_string(index=False))
        except Exception as e:
            log.warning("feature_importance failed: %s", e)

    log.info(
        "DONE (refit). run_id=%s val=%.2f test=%.2f. "
        "Run submit_ranker.py to generate the submission CSV.",
        run_id, score_val, score_test,
    )


if __name__ == "__main__":
    main()
