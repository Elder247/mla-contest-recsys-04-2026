"""Generate a submission by mean-blending ranker_score across N seeds.

Loads the N ranker pkls produced by ``scripts/train_ranker_multiseed.py``
and scores them on the same submission features. The submit-time feature
build (CG fit on full data + gen_candidates + compute_features) is reused
from ``scripts/submit_ranker.py`` — same Hydra phases, same feature
parquet path.

Usage:
    python -u scripts/submit_ranker_multiseed.py \\
        --config-name=ranker_v2_top1 data=500m \\
        run_id=v2_top1_ms \\
        +seed_list=[42,43,44,45,46] \\
        +submission_name=ranker_ms

Pre-condition: ``train_ranker_multiseed.py`` was run with the same
``run_id`` and ``seed_list``, producing ``ranker_{run_id}_seed{S}.pkl``
for every S in seed_list.

The generated CSV is mean-blended at the score level, then top-K per uid.
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


def _format_submission(top_k_df: pl.DataFrame) -> pl.DataFrame:
    """Group sorted top-K rows into the contest CSV format."""
    return (
        top_k_df
        .group_by("uid", maintain_order=True)
        .agg(pl.col("item_id").cast(pl.Utf8).str.join(delimiter=" ").alias("item_ids"))
        .sort("uid")
    )


@hydra.main(config_path="../configs", config_name="ranker", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    config_name = HydraConfig.get().job.config_name
    log.info("config_name=%s (passed to subprocess phases)", config_name)

    seeds = list(cfg.get("seed_list", [42, 43, 44, 45, 46]))
    if not seeds:
        raise ValueError("+seed_list must be a non-empty list of ints")
    run_id = str(cfg.run_id)
    submission_name = str(cfg.get("submission_name", "ranker_ms"))
    log.info("submit-multiseed: run_id=%s seeds=%s", run_id, seeds)

    # Verify all ranker pkls exist before running expensive feature phases.
    ranker_dir = Path(cfg.ranker_dir)
    ranker_paths = [ranker_dir / f"ranker_{run_id}_seed{s}.pkl" for s in seeds]
    missing = [p for p in ranker_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing ranker pkls: {missing}. Run scripts/train_ranker_multiseed.py "
            f"with run_id={run_id} and seed_list={seeds} first."
        )

    eval_users = load_eval_users_from_csv(cfg.data.users_csv)
    log.info("eval users: %d", len(eval_users))

    # ── 1. Phase 1 — fit / load CGs on FULL data (suffix='_full') ────────────
    fit_overrides = [
        _hydra_override("data", cfg.data.size),
        _hydra_override("run_id", run_id),
        _hydra_override("artifacts_root", cfg.artifacts_root),
        _hydra_override("data.root", cfg.data.root),
        _hydra_override("force_refit_cg", bool(cfg.force_refit_cg)),
        _hydra_override("suffix", "_full"),
        _hydra_override("train_cutoff_ts", None),
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
        _hydra_override("dislike_cutoff_ts", None),
        _hydra_override("users_source", str(eval_users_csv)),
        _hydra_override("output_dir_phase", str(full_dir)),
    ]
    _run_phase("scripts/_phases/gen_candidates.py", gen_overrides, config_name=config_name)
    merged_path = full_dir / "merged.parquet"

    # ── 3. Phase 3 — compute features (full data, no labels) ─────────────────
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
        _hydra_override("merged_path", str(merged_path)),
        _hydra_override("output_path", str(feats_path)),
        _hydra_override("cutoff_ts", cutoff_ts),
        _hydra_override("label_gt_path", None),
    ]
    _run_phase("scripts/_phases/compute_features.py", feat_overrides, config_name=config_name)

    # ── 4. In-process: score with each seed, blend, write CSV ────────────────
    log.info("loading submission features ← %s", feats_path)
    feats = pl.read_parquet(feats_path)
    log.info("submission features: %d rows × %d cols", len(feats), len(feats.columns))

    scored = feats.select(["uid", "item_id"])
    score_cols: list[str] = []
    for seed, ranker_path in zip(seeds, ranker_paths):
        log.info("loading ranker ← %s", ranker_path)
        with open(ranker_path, "rb") as f:
            ranker = pickle.load(f)
        log.info("scoring with seed=%d ranker", seed)
        s = ranker.score(feats).rename({"ranker_score": f"ranker_score_seed{seed}"})
        score_cols.append(f"ranker_score_seed{seed}")
        scored = scored.join(s, on=["uid", "item_id"], how="left")
        del ranker

    log.info("blending %d seeds via mean(ranker_score)", len(seeds))
    blended = scored.with_columns(
        pl.mean_horizontal(*[pl.col(c) for c in score_cols]).alias("score")
    )
    top_k_df = (
        blended.select(["uid", "item_id", "score"])
        .sort(["uid", "score"], descending=[False, True])
        .group_by("uid", maintain_order=True)
        .head(cfg.top_k)
    )

    submission = _format_submission(top_k_df)
    log.info("submission rows: %d", len(submission))
    missing_uids = set(eval_users) - set(submission["uid"].cast(pl.Int64).to_list())
    if missing_uids:
        log.warning("%d eval users have no predictions", len(missing_uids))

    sub_dir = Path(cfg.get("submission_dir", "submissions"))
    sub_dir.mkdir(parents=True, exist_ok=True)
    archive_path = sub_dir / f"sub_{run_id}_{submission_name}.csv"
    submission.write_csv(archive_path)
    submission.write_csv(sub_dir / "test.csv")
    log.info("submission saved to %s and %s", archive_path, sub_dir / "test.csv")


if __name__ == "__main__":
    main()
