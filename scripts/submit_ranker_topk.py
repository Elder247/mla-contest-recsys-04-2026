"""Generate a submission CSV for a top-K config using cached submit features.

Mirror of :mod:`scripts.refit_ranker_topk` for the inference half of the
top-K sweep. Reuses everything that ``submit_ranker.py run_id={base_run_id}``
already produced:

    {features_dir}/{base_run_id}_submit.parquet      full submit pool
    {ranker_dir}/lgbm_{base_run_id}.pkl              stage-1 LGBM (one per base)

Optional cache (created on first run, reused on subsequent top-K submits):

    {features_dir}/{base_run_id}_submit_lgbm.parquet  LGBM scores on submit pool

Per top-K it reads ``ranker_{run_id}.pkl`` (produced by ``refit_ranker_topk``)
and applies:

    1. ``n_cand_keep`` post-hoc filter using the per-config CG list.
    2. cascade cut to top-``n_ranker_eval`` per uid by ``lgbm_score``.
    3. CatBoost stage-2 ``ranker.predict`` → top-K per uid → CSV.

Wallclock per top-K: ~3-8 min on 500m (vs ~30+ min for full submit_ranker.py).

Usage:
    python -u scripts/submit_ranker_topk.py --config-name=ranker_v4_top3 \\
        data=500m run_id=v4_top3 +base_run_id=v4_features \\
        submission_name=v4_top3 \\
        2>&1 | tee /tmp/v4_top3_submit_topk.log

Outputs:
    submissions/sub_{run_id}_{submission_name}.csv
    submissions/test.csv  (overwritten each run — used by the contest UI)
    {features_dir}/{base_run_id}_submit_lgbm.parquet  (cached on first run)

The first run for a given ``base_run_id`` rescores LGBM on the full submit
pool and writes the cache (~5 min). Subsequent runs read the cache (<1 min).
"""
from __future__ import annotations

import logging
import pickle
from functools import reduce
from pathlib import Path
from typing import Any, Iterable

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf

from src.inference.phases import load_eval_users_from_csv
from src.models.lightgbm_ranker import LightGBMRanker  # noqa: F401  (pickle compat)
from src.utils import setup_logging

log = logging.getLogger(__name__)


def _apply_n_cand_keep(
    df: pl.DataFrame, cg_cfg_list: Iterable[Any],
) -> pl.DataFrame:
    """Mirror of :func:`merge_candidates.apply_n_cand_keep` on features parquet.

    Same OR semantics as the merge-time filter — see scripts.refit_ranker_topk
    for full docstring. Kept inline to keep this script standalone.
    """
    has_field = False
    keep_terms: list[pl.Expr] = []
    for cg in cg_cfg_list:
        if "n_cand_keep" not in cg:
            continue
        has_field = True
        n_keep = cg["n_cand_keep"]
        if n_keep is None or int(n_keep) <= 0:
            continue
        name = cg.get("name")
        rank_col = f"{name}_rank"
        if rank_col not in df.columns:
            raise ValueError(
                f"_apply_n_cand_keep: CG '{name}' has n_cand_keep={n_keep} "
                f"but '{rank_col}' is not in features columns"
            )
        keep_terms.append(
            pl.col(rank_col).is_not_null() & (pl.col(rank_col) <= int(n_keep))
        )
    if not has_field:
        log.info("_apply_n_cand_keep: no CG has the field — returning df unchanged")
        return df
    if not keep_terms:
        raise ValueError(
            "_apply_n_cand_keep: every CG with 'n_cand_keep' set was 0 — "
            "no rows would survive."
        )
    keep_expr = reduce(lambda a, b: a | b, keep_terms)
    before = len(df)
    filtered = df.filter(keep_expr)
    log.info(
        "_apply_n_cand_keep: filtered %d → %d rows (dropped %d) using %d active CGs",
        before, len(filtered), before - len(filtered), len(keep_terms),
    )
    return filtered


def _format_submission(preds: pl.DataFrame, top_k: int) -> pl.DataFrame:
    """Same formatting as scripts/submit_ranker.py."""
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


@hydra.main(config_path="../configs", config_name="ranker", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    base_run_id = cfg.get("base_run_id")
    if base_run_id is None:
        raise ValueError(
            "+base_run_id=<existing run with cached submit features> is required. "
            "Run scripts/submit_ranker.py with that run_id first to populate "
            "{features_dir}/{base_run_id}_submit.parquet and "
            "{ranker_dir}/lgbm_{base_run_id}.pkl."
        )
    base_run_id = str(base_run_id)
    run_id = str(cfg.run_id)
    submission_name = str(cfg.get("submission_name", "ranker"))
    log.info(
        "submit-topk: base=%s out=%s submission_name=%s",
        base_run_id, run_id, submission_name,
    )

    features_dir = Path(cfg.features_dir)
    submit_feats_path = features_dir / f"{base_run_id}_submit.parquet"
    submit_lgbm_path = features_dir / f"{base_run_id}_submit_lgbm.parquet"
    ranker_dir = Path(cfg.ranker_dir)
    base_lgbm_path = ranker_dir / f"lgbm_{base_run_id}.pkl"
    ranker_path = ranker_dir / f"ranker_{run_id}.pkl"

    if not submit_feats_path.exists():
        raise FileNotFoundError(
            f"missing prerequisite: {submit_feats_path}\n"
            f"Run scripts/submit_ranker.py with run_id={base_run_id} first."
        )
    if not base_lgbm_path.exists():
        raise FileNotFoundError(
            f"missing LGBM stage-1: {base_lgbm_path}\n"
            f"Run scripts/train_ranker.py with run_id={base_run_id} first."
        )
    if not ranker_path.exists():
        raise FileNotFoundError(
            f"missing ranker for this top-K: {ranker_path}\n"
            f"Run scripts/refit_ranker_topk.py with run_id={run_id} first."
        )

    eval_users = load_eval_users_from_csv(cfg.data.users_csv)
    log.info("eval users: %d", len(eval_users))

    # ── 1. Load submit features (full pool) ──────────────────────────────────
    log.info("loading submission features ← %s", submit_feats_path)
    feats = pl.read_parquet(submit_feats_path)
    log.info(
        "submission features (pre-filter): %d rows × %d cols",
        len(feats), len(feats.columns),
    )

    # ── 2. Apply per-config n_cand_keep filter (post-hoc) ────────────────────
    feats = _apply_n_cand_keep(feats, cfg.candidate_generators)
    log.info("submission features (post-filter): %d rows", len(feats))

    # ── 3. LGBM stage-1 — read cache or score + write cache ──────────────────
    # The LGBM scores depend only on FEATURE VALUES (not on which rows
    # survive ``n_cand_keep``). So one cache per ``base_run_id`` is reused
    # by every top-K submit pass. First run writes the cache (~5 min on
    # 500m); subsequent runs read it (<1 min).
    force_rescore = bool(cfg.get("force_rescore_lgbm", False))
    if submit_lgbm_path.exists() and not force_rescore:
        log.info("loading cached LGBM submit scores ← %s", submit_lgbm_path)
        lgbm_scores = pl.read_parquet(submit_lgbm_path)
    else:
        log.info(
            "no LGBM submit cache (or force_rescore_lgbm=true) — scoring with %s",
            base_lgbm_path,
        )
        with open(base_lgbm_path, "rb") as f:
            lgbm = pickle.load(f)
        # Score the FULL submit pool (pre-filter) once, so the cache is
        # reusable across top-K configs with different n_cand_keep filters.
        log.info(
            "loading FULL submit pool for LGBM scoring (cache write) ← %s",
            submit_feats_path,
        )
        feats_full = pl.read_parquet(submit_feats_path)
        lgbm_scores = lgbm.score(feats_full)
        del feats_full, lgbm
        lgbm_scores.write_parquet(submit_lgbm_path, compression="zstd")
        log.info("LGBM submit scores cached → %s", submit_lgbm_path)

    # ── 4. Cascade cut: top-n_ranker_eval per uid ────────────────────────────
    n_ranker_eval = int(cfg.get("n_ranker_eval", cfg.get("n_ranker", 1500)))
    feats_cut = (
        feats.join(lgbm_scores, on=["uid", "item_id"], how="left")
        .sort(["uid", "lgbm_score"], descending=[False, True])
        .group_by("uid", maintain_order=True)
        .head(n_ranker_eval)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("uid").cast(pl.Int32).alias("lgbm_rank")
        )
    )
    log.info(
        "cascade: %d → %d rows after top-%d cut",
        len(feats), len(feats_cut), n_ranker_eval,
    )
    n_null = int(feats_cut["lgbm_score"].is_null().sum())
    if n_null > 0:
        log.warning(
            "cascade: %d submit rows have NULL lgbm_score after join — "
            "missing from cached LGBM scores parquet (likely (uid, item_id) "
            "not in cache; will sort to bottom)",
            n_null,
        )
    del feats, lgbm_scores

    # ── 5. CatBoost stage-2 → top-K per user → CSV ───────────────────────────
    log.info("loading ranker ← %s", ranker_path)
    with open(ranker_path, "rb") as f:
        ranker = pickle.load(f)

    preds = ranker.predict(feats_cut, n=cfg.top_k)
    submission = _format_submission(preds, top_k=cfg.top_k)
    log.info("submission rows: %d", len(submission))

    missing = set(eval_users) - set(submission["uid"].cast(pl.Int64).to_list())
    if missing:
        log.warning("%d eval users have no predictions (cold users)", len(missing))

    sub_dir = Path(cfg.get("submission_dir", "submissions"))
    sub_dir.mkdir(parents=True, exist_ok=True)
    archive_path = sub_dir / f"sub_{run_id}_{submission_name}.csv"
    submission.write_csv(archive_path)
    submission.write_csv(sub_dir / "test.csv")
    log.info("submission saved to %s and %s", archive_path, sub_dir / "test.csv")


if __name__ == "__main__":
    main()
