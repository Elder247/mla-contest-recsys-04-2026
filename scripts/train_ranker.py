"""Multi-CG → CatBoost Ranker training pipeline (subprocess-staged).

The pipeline is split into three phases that run in their own Python
subprocesses so the OS reclaims Polars/Arrow allocator RSS at phase
boundaries (essential on 500m / 5B with 120 GB RAM):

  Phase 1 — fit candidate generators           (scripts/_phases/fit_cgs.py)
  Phase 2 — generate candidates + merge        (scripts/_phases/gen_candidates.py)
  Phase 3 — compute features parquet           (scripts/_phases/compute_features.py)
  Phase 4 — (this process) train the ranker, evaluate, log results

Per-run intermediate artifacts:

  artifacts/gt/{run_id}/gt_val.parquet          ground-truth pairs (val window)
  artifacts/gt/{run_id}/gt_test.parquet         ground-truth pairs (test window)
  artifacts/candidates/{run_id}/val/cg_*.parquet + merged.parquet
  artifacts/candidates/{run_id}/eval/cg_*.parquet + merged.parquet
  artifacts/features/{run_id}_train.parquet     labeled features for ranker fit
  artifacts/features/{run_id}_eval.parquet      features for full eval users

The ranker itself is fit + evaluated in this orchestrator process, since
CatBoost is the next-largest memory consumer and we want it to start with
a clean RSS baseline.

Usage:
    python -u scripts/train_ranker.py data=50m run_id=006
"""
from __future__ import annotations

import csv
import logging
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupShuffleSplit

from src.evaluation.metrics import recall_at_k
from src.inference.merge_candidates import cg_recall
from src.inference.phases import (
    derive_split_metadata,
    load_eval_users_from_csv,
    write_ground_truth,
)
from src.models.catboost_ranker import RankerModel
from src.utils import setup_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run_phase(script: str, overrides: list[str]) -> None:
    """Run a phase script as ``python -u <script> <overrides>``.

    Bubbles up the subprocess's exit code on failure. Stdout/stderr stream
    to the parent's TTY so ``tee /tmp/run.log`` continues to capture
    everything in order.
    """
    cmd = [sys.executable, "-u", script] + overrides
    log.info("running phase: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _hydra_override(key: str, value) -> str:
    """Build a single Hydra CLI override (handles None → 'null' and quoting)."""
    if value is None:
        return f"{key}=null"
    if isinstance(value, bool):
        return f"{key}={'true' if value else 'false'}"
    return f"{key}={value}"


# ---------------------------------------------------------------------------
# Ranker training (in-process)
# ---------------------------------------------------------------------------

def _split_for_ranker(df: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """80/20 group split by uid without round-tripping through pandas."""
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


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="ranker", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.seed)

    run_id = str(cfg.run_id)
    eval_users = load_eval_users_from_csv(cfg.data.users_csv)

    # ── 1. Derive temporal-split metadata + write ground truth ───────────────
    log.info("deriving temporal-split metadata from %s", cfg.data.listens)
    meta = derive_split_metadata(
        cfg.data.listens,
        val_days=cfg.split.val_days,
        gap_days=cfg.split.gap_days,
        timestamp_col=cfg.split.timestamp_col,
    )
    log.info(
        "split: t_max=%d val_start=%d val_end=%d test_start=%d t_end=%d train_max_ts=%d",
        meta["t_max"], meta["val_start"], meta["val_end"],
        meta["test_start"], meta["t_end"], meta["train_max_ts"],
    )

    gt_dir = Path(cfg.gt_dir) / run_id
    gt_val_path = write_ground_truth(
        cfg.data.listens, eval_users,
        meta["val_start"], meta["val_end"],
        gt_dir / "gt_val.parquet",
    )
    gt_test_path = write_ground_truth(
        cfg.data.listens, eval_users,
        meta["test_start"], meta["t_end"],
        gt_dir / "gt_test.parquet",
    )

    # ── 2. Phase 1 — fit CGs (subprocess) ────────────────────────────────────
    # train_cutoff_ts = val_start so listens used for fit have timestamp <
    # val_start (matches the legacy split.train["timestamp"].max() bound).
    train_cutoff_ts = meta["val_start"]
    fit_overrides: list[str] = [
        _hydra_override("data", cfg.data.size),
        _hydra_override("run_id", run_id),
        _hydra_override("artifacts_root", cfg.artifacts_root),
        _hydra_override("data.root", cfg.data.root),
        _hydra_override("force_refit_cg", bool(cfg.force_refit_cg)),
        _hydra_override("suffix", ""),
        _hydra_override("train_cutoff_ts", train_cutoff_ts),
    ]
    _run_phase("scripts/_phases/fit_cgs.py", fit_overrides)

    # ── 3. Phase 2 — generate candidates × 2 (val users + full eval) ─────────
    cand_root = Path(cfg.candidates_dir) / run_id
    val_users_with_gt = (
        pl.read_parquet(gt_val_path).get_column("uid").unique().sort().to_list()
    )
    val_users_csv = cand_root / "val_users.csv"
    eval_users_csv = cand_root / "eval_users.csv"
    cand_root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"uid": val_users_with_gt}).write_csv(val_users_csv)
    pl.DataFrame({"uid": eval_users}).write_csv(eval_users_csv)

    val_dir = cand_root / "val"
    eval_dir = cand_root / "eval"

    common_gen = [
        _hydra_override("data", cfg.data.size),
        _hydra_override("run_id", run_id),
        _hydra_override("artifacts_root", cfg.artifacts_root),
        _hydra_override("data.root", cfg.data.root),
        _hydra_override("suffix", ""),
        _hydra_override("filter_dislikes", bool(cfg.filter_dislikes)),
        _hydra_override("dislike_cutoff_ts", train_cutoff_ts),
    ]
    _run_phase("scripts/_phases/gen_candidates.py", common_gen + [
        _hydra_override("users_source", str(val_users_csv)),
        _hydra_override("output_dir_phase", str(val_dir)),
    ])
    _run_phase("scripts/_phases/gen_candidates.py", common_gen + [
        _hydra_override("users_source", str(eval_users_csv)),
        _hydra_override("output_dir_phase", str(eval_dir)),
    ])

    merged_train_path = val_dir / "merged.parquet"
    merged_eval_path = eval_dir / "merged.parquet"

    # CG-recall@∞ on val (upper bound for the ranker; logged for parity with
    # the previous pipeline).
    upper_bound = cg_recall(
        pl.read_parquet(merged_train_path),
        pl.read_parquet(gt_val_path),
    )
    log.info(
        "CG-recall@∞ on val (upper bound for ranker, ×1000 scale): %.2f",
        upper_bound,
    )

    # ── 4. Phase 3 — compute features × 2 (labeled train + unlabeled eval) ───
    features_dir = Path(cfg.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    feats_train_path = features_dir / f"{run_id}_train.parquet"
    feats_eval_path = features_dir / f"{run_id}_eval.parquet"

    common_feat = [
        _hydra_override("data", cfg.data.size),
        _hydra_override("run_id", run_id),
        _hydra_override("artifacts_root", cfg.artifacts_root),
        _hydra_override("data.root", cfg.data.root),
        _hydra_override("enable_embed_features", bool(cfg.get("enable_embed_features", True))),
        _hydra_override("cutoff_ts", train_cutoff_ts),
    ]
    _run_phase("scripts/_phases/compute_features.py", common_feat + [
        _hydra_override("merged_path", str(merged_train_path)),
        _hydra_override("output_path", str(feats_train_path)),
        _hydra_override("label_gt_path", str(gt_val_path)),
    ])
    _run_phase("scripts/_phases/compute_features.py", common_feat + [
        _hydra_override("merged_path", str(merged_eval_path)),
        _hydra_override("output_path", str(feats_eval_path)),
        # No label_gt_path — eval features are unlabeled; predict only.
    ])

    # ── 5. In-process: load labeled train features → train ranker ────────────
    log.info("loading labeled train features ← %s", feats_train_path)
    labeled = pl.read_parquet(feats_train_path)
    pos_rate = float(labeled["label"].mean())
    log.info(
        "labeled features: %d rows × %d cols | label rate: %.4f (neg_ratio ~%d:1)",
        len(labeled), len(labeled.columns), pos_rate,
        int(1 / pos_rate) if pos_rate > 0 else 0,
    )

    df_train, df_val_ranker = _split_for_ranker(labeled, cfg.seed)
    log.info("ranker train=%d  val=%d", len(df_train), len(df_val_ranker))
    del labeled

    ranker = RankerModel(**cfg.ranker)
    ranker.fit(df_train, df_val_ranker)
    del df_train, df_val_ranker

    # ── 6. Eval Recall@100 on val + test ─────────────────────────────────────
    log.info("loading eval features ← %s", feats_eval_path)
    feats_full = pl.read_parquet(feats_eval_path)
    log.info(
        "eval features: %d rows × %d cols",
        len(feats_full), len(feats_full.columns),
    )

    gt_val = pl.read_parquet(gt_val_path)
    gt_test = pl.read_parquet(gt_test_path)
    log.info(
        "val ground truth: %d pairs / %d users; test: %d pairs / %d users",
        len(gt_val), gt_val["uid"].n_unique(),
        len(gt_test), gt_test["uid"].n_unique(),
    )

    preds_val = ranker.predict(feats_full, n=cfg.top_k)
    score_val = recall_at_k(gt_val, preds_val, k=cfg.top_k)
    log.info("val  Recall@%d = %.2f", cfg.top_k, score_val)

    preds_test = ranker.predict(feats_full, n=cfg.top_k)
    score_test = recall_at_k(gt_test, preds_test, k=cfg.top_k)
    log.info("test Recall@%d = %.2f", cfg.top_k, score_test)

    # ── 7. Persist ranker + log results ──────────────────────────────────────
    ranker_dir = Path(cfg.ranker_dir)
    ranker_dir.mkdir(parents=True, exist_ok=True)
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
            "model": f"ranker[{cg_names}]",
            "dataset_size": cfg.data.size,
            "split": split_name,
            "score": round(score, 4),
            "config_path": "configs/ranker.yaml",
        })
    log.info("results appended to %s", results_path)

    # ── 8. Optional feature importance ───────────────────────────────────────
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
