"""Train N rankers with different seeds on cached features and report
the blended val/test recall.

Pre-condition: ``train_ranker.py`` has already been run with the same
config (``--config-name=ranker_v2_topX``) and ``run_id=base_run_id`` —
this script reuses its cached labeled-train features, eval features, and
ground-truth parquets. No CG re-fit, no feature recompute.

Usage:
    python -u scripts/train_ranker_multiseed.py \\
        --config-name=ranker_v2_top1 data=500m \\
        run_id=v2_top1_ms \\
        +base_run_id=v2_top1 \\
        +seed_list=[42,43,44,45,46]

Outputs:
    artifacts/ranker_{run_id}_seed{S}.pkl      one per seed
    artifacts/results.csv                       blended val/test rows

The companion ``submit_ranker_multiseed.py`` consumes these pkls and
mean-blends ``ranker_score`` on the full eval pool when generating a
submission CSV. Splitting train and submit lets us run the cheap part
(training) sequentially without re-doing the expensive full-data feature
build for every config we test.
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
from src.utils import setup_logging

log = logging.getLogger(__name__)


def _split_for_ranker(df: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """80/20 group-by-uid split — same logic as scripts/train_ranker.py."""
    uids = df["uid"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(np.zeros(len(df)), groups=uids))
    return df[train_idx], df[val_idx]


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

    base_run_id = cfg.get("base_run_id")
    if base_run_id is None:
        raise ValueError(
            "+base_run_id=<existing run with cached features> is required. "
            "Run train_ranker.py first to populate {features_dir}/{base_run_id}_*.parquet."
        )
    seeds = list(cfg.get("seed_list", [42, 43, 44, 45, 46]))
    if not seeds:
        raise ValueError("+seed_list must be a non-empty list of ints")
    out_run_id = str(cfg.run_id)
    log.info("multi-seed: base=%s out=%s seeds=%s", base_run_id, out_run_id, seeds)

    features_dir = Path(cfg.features_dir)
    feats_train_path = features_dir / f"{base_run_id}_train.parquet"
    feats_eval_path = features_dir / f"{base_run_id}_eval.parquet"
    gt_dir = Path(cfg.gt_dir) / base_run_id
    gt_val_path = gt_dir / "gt_val.parquet"
    gt_test_path = gt_dir / "gt_test.parquet"
    for p in (feats_train_path, feats_eval_path, gt_val_path, gt_test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"missing prerequisite: {p}\n"
                f"Run scripts/train_ranker.py with run_id={base_run_id} first."
            )

    log.info("loading labeled train features ← %s", feats_train_path)
    labeled = pl.read_parquet(feats_train_path)
    log.info("labeled features: %d rows × %d cols", len(labeled), len(labeled.columns))

    log.info("loading eval features ← %s", feats_eval_path)
    feats_eval = pl.read_parquet(feats_eval_path)
    log.info("eval features: %d rows × %d cols", len(feats_eval), len(feats_eval.columns))

    gt_val = pl.read_parquet(gt_val_path)
    gt_test = pl.read_parquet(gt_test_path)
    log.info(
        "gt_val: %d pairs / %d users; gt_test: %d pairs / %d users",
        len(gt_val), gt_val["uid"].n_unique(),
        len(gt_test), gt_test["uid"].n_unique(),
    )

    # Group-split labeled into train + val once — every seed sees the same split.
    df_train, df_val = _split_for_ranker(labeled, seed=cfg.seed)
    log.info("ranker train=%d  val=%d", len(df_train), len(df_val))
    del labeled

    ranker_dir = Path(cfg.ranker_dir)
    ranker_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed scoring on eval features (collect into one DF for blend).
    score_cols: list[str] = []
    scored = feats_eval.select(["uid", "item_id"])
    per_seed_recalls: list[dict] = []

    for seed in seeds:
        log.info("=" * 70)
        log.info("training ranker for seed=%d", seed)
        ranker_kwargs = {**cfg.ranker, "random_state": int(seed)}
        ranker = RankerModel(**ranker_kwargs)
        ranker.fit(df_train, df_val)

        ranker_path = ranker_dir / f"ranker_{out_run_id}_seed{seed}.pkl"
        with open(ranker_path, "wb") as f:
            pickle.dump(ranker, f)
        log.info("saved %s", ranker_path)

        # Score eval features.
        log.info("scoring eval features with seed=%d ranker", seed)
        s = ranker.score(feats_eval).rename({"ranker_score": f"ranker_score_seed{seed}"})
        score_cols.append(f"ranker_score_seed{seed}")
        scored = scored.join(s, on=["uid", "item_id"], how="left")

        # Per-seed val/test recall (sanity).
        seed_top = (
            scored.select(["uid", "item_id", f"ranker_score_seed{seed}"])
            .rename({f"ranker_score_seed{seed}": "score"})
            .sort(["uid", "score"], descending=[False, True])
            .group_by("uid", maintain_order=True)
            .head(cfg.top_k)
        )
        seed_val = recall_at_k(gt_val, seed_top, k=cfg.top_k)
        seed_test = recall_at_k(gt_test, seed_top, k=cfg.top_k)
        log.info("seed=%d  val=%.2f  test=%.2f", seed, seed_val, seed_test)
        per_seed_recalls.append({"seed": seed, "val": seed_val, "test": seed_test})
        del ranker

    # Mean-blend across seeds.
    log.info("=" * 70)
    log.info("blending %d seeds via mean(ranker_score)", len(seeds))
    blended = scored.with_columns(
        pl.mean_horizontal(*[pl.col(c) for c in score_cols]).alias("score")
    )
    blended_top = (
        blended.select(["uid", "item_id", "score"])
        .sort(["uid", "score"], descending=[False, True])
        .group_by("uid", maintain_order=True)
        .head(cfg.top_k)
    )
    blend_val = recall_at_k(gt_val, blended_top, k=cfg.top_k)
    blend_test = recall_at_k(gt_test, blended_top, k=cfg.top_k)
    log.info("BLENDED  val=%.2f  test=%.2f", blend_val, blend_test)
    log.info("per-seed recalls:")
    for r in per_seed_recalls:
        log.info("  seed=%d  val=%.2f  test=%.2f", r["seed"], r["val"], r["test"])

    # Append results.csv rows for both per-seed and blended.
    results_path = Path(cfg.output_dir) / "results.csv"
    run_id_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cg_names = ",".join(c.get("name") for c in cfg.candidate_generators)
    model_label = f"ranker_multiseed_{len(seeds)}seeds[{cg_names}]"
    for split_name, score in [("val", blend_val), ("test", blend_test)]:
        _append_results(
            results_path,
            {
                "run_id": run_id_ts,
                "model": model_label,
                "dataset_size": cfg.data.size,
                "split": split_name,
                "score": round(float(score), 4),
                "config_path": "configs/ranker.yaml",
            },
        )
    log.info("results appended to %s", results_path)

    log.info(
        "DONE. base=%s out=%s seeds=%s blend_val=%.2f blend_test=%.2f",
        base_run_id, out_run_id, seeds, blend_val, blend_test,
    )
    log.info(
        "Run scripts/submit_ranker_multiseed.py to generate the submission CSV "
        "from these %d ranker pkls.", len(seeds),
    )


if __name__ == "__main__":
    main()
