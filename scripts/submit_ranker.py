"""Generate a submission from a trained ranker (subprocess-staged).

Same staged architecture as ``scripts/train_ranker.py`` (each phase in its
own subprocess so RSS resets at boundaries). Differences vs train:

  - Single dataset (no temporal split, no ground truth derivation).
  - CG cache slot ``_full`` (CGs fit on the full event table).
  - cutoff_ts = max(timestamp) over full listens, used uniformly for
    feature aggregates.
  - ``ranker.predict`` runs in-process, writes CSV in the contest format.

Usage:
    python -u scripts/submit_ranker.py data=50m run_id=003
"""
from __future__ import annotations

import logging
import pickle
import subprocess
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.inference.phases import load_eval_users_from_csv
from src.utils import setup_logging

log = logging.getLogger(__name__)


def _run_phase(script: str, overrides: list[str], config_name: str | None = None) -> None:
    """Same semantics as train_ranker._run_phase — propagates --config-name."""
    cmd = [sys.executable, "-u", script]
    if config_name is not None and config_name not in ("ranker", "submit_ranker"):
        cmd.append(f"--config-name={config_name}")
    cmd.extend(overrides)
    log.info("running phase: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _hydra_override(key: str, value) -> str:
    if value is None:
        return f"{key}=null"
    if isinstance(value, bool):
        return f"{key}={'true' if value else 'false'}"
    return f"{key}={value}"


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

    config_name = HydraConfig.get().job.config_name
    log.info("config_name=%s (passed to subprocess phases)", config_name)

    run_id = str(cfg.run_id)
    eval_users = load_eval_users_from_csv(cfg.data.users_csv)
    log.info("eval users: %d", len(eval_users))

    ranker_path = Path(cfg.ranker_dir) / f"ranker_{run_id}.pkl"
    if not ranker_path.exists():
        raise FileNotFoundError(
            f"ranker pickle not found: {ranker_path}. Run train_ranker.py first."
        )

    # ── 1. Phase 1 — fit / load CGs on FULL data (suffix='_full') ────────────
    fit_overrides = [
        _hydra_override("data", cfg.data.size),
        _hydra_override("run_id", run_id),
        _hydra_override("artifacts_root", cfg.artifacts_root),
        _hydra_override("data.root", cfg.data.root),
        _hydra_override("force_refit_cg", bool(cfg.force_refit_cg)),
        _hydra_override("suffix", "_full"),
        _hydra_override("train_cutoff_ts", None),  # full data
    ]
    _run_phase("scripts/_phases/fit_cgs.py", fit_overrides, config_name=config_name)

    # ── 2. Phase 2 — generate candidates for all eval users ──────────────────
    cand_root = Path(cfg.candidates_dir) / f"{run_id}_full"
    cand_root.mkdir(parents=True, exist_ok=True)
    eval_users_csv = cand_root / "eval_users.csv"
    pl.DataFrame({"uid": eval_users}).write_csv(eval_users_csv)

    full_dir = cand_root / "full"
    gen_overrides = [
        _hydra_override("data", cfg.data.size),
        _hydra_override("run_id", run_id),
        _hydra_override("artifacts_root", cfg.artifacts_root),
        _hydra_override("data.root", cfg.data.root),
        _hydra_override("suffix", "_full"),
        _hydra_override("filter_dislikes", bool(cfg.filter_dislikes)),
        _hydra_override("dislike_cutoff_ts", None),  # full inference, no cutoff
        _hydra_override("users_source", str(eval_users_csv)),
        _hydra_override("output_dir_phase", str(full_dir)),
    ]
    _run_phase("scripts/_phases/gen_candidates.py", gen_overrides, config_name=config_name)
    merged_path = full_dir / "merged.parquet"

    # ── 3. Phase 3 — compute features ────────────────────────────────────────
    # cutoff_ts = max timestamp across full listens (no train/val split).
    cutoff_ts = int(
        pl.scan_parquet(cfg.data.listens)
        .select(pl.col("timestamp").max())
        .collect()
        .row(0)[0]
    )
    log.info("submit cutoff_ts=%d (full data max)", cutoff_ts)

    features_dir = Path(cfg.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    feats_path = features_dir / f"{run_id}_submit.parquet"

    feat_overrides = [
        _hydra_override("data", cfg.data.size),
        _hydra_override("run_id", run_id),
        _hydra_override("artifacts_root", cfg.artifacts_root),
        _hydra_override("data.root", cfg.data.root),
        _hydra_override("enable_embed_features", bool(cfg.get("enable_embed_features", True))),
        _hydra_override("feature_chunk_size", int(cfg.get("feature_chunk_size", 0) or 0)),
        _hydra_override("merged_path", str(merged_path)),
        _hydra_override("output_path", str(feats_path)),
        _hydra_override("cutoff_ts", cutoff_ts),
        _hydra_override("label_gt_path", None),  # submission has no labels
    ]
    _run_phase("scripts/_phases/compute_features.py", feat_overrides, config_name=config_name)

    # ── 4. In-process: predict + write submission CSV ────────────────────────
    log.info("loading ranker ← %s", ranker_path)
    with open(ranker_path, "rb") as f:
        ranker = pickle.load(f)

    log.info("loading submission features ← %s", feats_path)
    feats = pl.read_parquet(feats_path)
    log.info(
        "submission features: %d rows × %d cols",
        len(feats), len(feats.columns),
    )

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
