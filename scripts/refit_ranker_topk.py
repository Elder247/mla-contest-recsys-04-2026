"""Train a top-K CatBoost ranker from cached features + LGBM scores of a
``base_run_id`` warmup run, applying per-config ``n_cand_keep`` post-hoc.

The orthodox path (``train_ranker.py --config-name=ranker_v4_topN``) re-runs
the entire pipeline per top-K config:

    fit CGs → gen candidates (with apply_n_cand_keep at merge time)
    → compute features  ← ~30-40 min on 500m
    → fit LGBM          ← ~25-40 min on 500m
    → fit CatBoost      ← ~10-15 min

For a sweep over the Optuna top-K (top1..top5) where only ``n_cand_keep_X``
and the CatBoost hyperparams differ, the first three phases are wasted work
— the CG outputs and features only depend on ``n_cand`` (= 800 for every
top-K) and the CG list (identical across top-K configs), and the LGBM scores
only depend on the labeled features.

This script reuses everything that's safe to share:

    {features_dir}/{base_run_id}_train.parquet           full n_cand=800 pool
    {features_dir}/{base_run_id}_eval.parquet            full n_cand=800 pool
    {features_dir}/{base_run_id}_train_lgbm.parquet      cached OOF LGBM scores
    {features_dir}/{base_run_id}_eval_lgbm.parquet       cached final-fit LGBM scores
    {gt_dir}/{base_run_id}/gt_{val,test}.parquet          ground truth

and only redoes:

    n_cand_keep filter (post-features, on {cg}_rank columns)
    cascade cut (top-n_ranker_train / top-n_ranker_eval per uid)
    CatBoost.fit (per-config catboost params from cfg.ranker)
    eval recall + persist ranker pickle as ranker_{run_id}.pkl

Wallclock per top-K: ~5-15 min on 500m (vs ~1.5-2h with full pipeline).

Pre-condition: ``train_ranker.py run_id={base_run_id}`` already produced the
five parquets above (warmup run, e.g. base ranker.yaml on 9 CGs n_cand=800).

Usage:
    python -u scripts/refit_ranker_topk.py --config-name=ranker_v4_top3 \\
        data=500m run_id=v4_top3 +base_run_id=v4_features \\
        2>&1 | tee /tmp/v4_top3_refit_topk.log

Outputs:
    artifacts/ranker_{run_id}.pkl                  CatBoost ranker
    artifacts/feature_importance_{run_id}.csv      (when compute_feature_importance)
    artifacts/results.csv                          appended val/test rows

Run scripts/submit_ranker_topk.py afterwards to produce the submission CSV.

Notes:

* ``cg_mean_score_norm`` is computed in ``generate_phase`` using the full
  pool's min/max. After post-hoc ``n_cand_keep`` filter the column is a
  *mild* approximation (top-rank rows survive, so min/max stay close).
  Importance in v4_top1 was ~0.07% — not worth recomputing here.

* The semantics of ``_apply_n_cand_keep`` MUST match
  :func:`src.inference.merge_candidates.apply_n_cand_keep`: a row survives
  iff at least one CG has ``{cg}_rank.is_not_null() & {cg}_rank <= keep``.
  Tested in tests/test_refit_topk.py.
"""
from __future__ import annotations

import csv
import logging
import pickle
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Iterable

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupShuffleSplit

from src.evaluation.metrics import recall_at_k
from src.models.catboost_ranker import RankerModel
from src.models.lightgbm_ranker import LightGBMRanker  # noqa: F401  (pickle compat)
from src.utils import setup_logging

log = logging.getLogger(__name__)


def _split_for_ranker(
    df: pl.DataFrame, seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """80/20 group-by-uid split — same logic as scripts/train_ranker.py."""
    uids = df["uid"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(np.zeros(len(df)), groups=uids))
    return df[train_idx], df[val_idx]


def _apply_n_cand_keep(
    df: pl.DataFrame, cg_cfg_list: Iterable[Any],
) -> pl.DataFrame:
    """Mirror of :func:`merge_candidates.apply_n_cand_keep` for features parquet.

    The features parquet retains every ``{cg}_rank`` column from the merge
    step (``add_features`` only joins; it doesn't drop or overwrite them).
    A row survives iff ``rank.is_not_null() & rank <= keep`` for at least
    one CG — same OR semantics as the merge-time filter.

    Edge cases:
      * No CG has ``n_cand_keep`` set → no-op (returns df unchanged).
      * Some CGs missing the field → those CGs are not gating.
      * Every present ``n_cand_keep`` is 0 / None → ValueError (would empty
        the pool).
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
        log.info(
            "_apply_n_cand_keep: no CG has the field — returning df unchanged",
        )
        return df

    if not keep_terms:
        raise ValueError(
            "_apply_n_cand_keep: every CG with 'n_cand_keep' set was 0 — "
            "no rows would survive. Set n_cand_keep > 0 for at least one CG.",
        )

    keep_expr = reduce(lambda a, b: a | b, keep_terms)
    before = len(df)
    filtered = df.filter(keep_expr)
    log.info(
        "_apply_n_cand_keep: filtered %d → %d rows (dropped %d) using %d active CGs",
        before, len(filtered), before - len(filtered), len(keep_terms),
    )
    return filtered


def _cascade_cut(
    df_feat: pl.DataFrame, df_lgbm: pl.DataFrame, n: int,
) -> pl.DataFrame:
    """Same cascade cut as scripts/train_ranker.py + multiseed siblings.

    Joins LGBM scores, sorts each user's pool by ``lgbm_score`` desc, keeps
    top-n, and adds a dense ``lgbm_rank`` (1-indexed) column for CatBoost.
    """
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

    base_run_id = cfg.get("base_run_id")
    if base_run_id is None:
        raise ValueError(
            "+base_run_id=<existing run with cached features> is required. "
            "Run scripts/train_ranker.py first to populate "
            "{features_dir}/{base_run_id}_*.parquet and lgbm_{base_run_id}.pkl."
        )
    base_run_id = str(base_run_id)
    run_id = str(cfg.run_id)
    log.info("refit-topk: base=%s out=%s", base_run_id, run_id)

    features_dir = Path(cfg.features_dir)
    feats_train_path = features_dir / f"{base_run_id}_train.parquet"
    feats_eval_path = features_dir / f"{base_run_id}_eval.parquet"
    train_lgbm_path = features_dir / f"{base_run_id}_train_lgbm.parquet"
    eval_lgbm_path = features_dir / f"{base_run_id}_eval_lgbm.parquet"
    gt_dir = Path(cfg.gt_dir) / base_run_id
    gt_val_path = gt_dir / "gt_val.parquet"
    gt_test_path = gt_dir / "gt_test.parquet"

    required = [
        feats_train_path, feats_eval_path,
        train_lgbm_path, eval_lgbm_path,
        gt_val_path, gt_test_path,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"missing prerequisite: {p}\n"
                f"Run scripts/train_ranker.py with run_id={base_run_id} first."
            )

    # ── 1. Load cached labeled features + apply n_cand_keep ──────────────────
    log.info("loading labeled train features ← %s", feats_train_path)
    labeled = pl.read_parquet(feats_train_path)
    log.info(
        "labeled features (pre-filter): %d rows × %d cols",
        len(labeled), len(labeled.columns),
    )
    labeled = _apply_n_cand_keep(labeled, cfg.candidate_generators)
    pos_rate = float(labeled["label"].mean())
    log.info(
        "labeled features (post-filter): %d rows | label rate %.4f (~%d:1)",
        len(labeled), pos_rate, int(1 / pos_rate) if pos_rate > 0 else 0,
    )

    # ── 2. Load cached eval features + apply n_cand_keep ─────────────────────
    log.info("loading eval features ← %s", feats_eval_path)
    feats_eval = pl.read_parquet(feats_eval_path)
    log.info(
        "eval features (pre-filter): %d rows × %d cols",
        len(feats_eval), len(feats_eval.columns),
    )
    feats_eval = _apply_n_cand_keep(feats_eval, cfg.candidate_generators)
    log.info("eval features (post-filter): %d rows", len(feats_eval))

    # ── 3. Load cached LGBM scores (semi-join to surviving rows is implicit
    #       in the cascade cut's left-join + top-N: any row not present in
    #       the cached scores parquet would get NULL lgbm_score and sort
    #       to the bottom. Defensive sanity-check below.) ─────────────────────
    log.info(
        "loading cached LGBM scores ← train=%s  eval=%s",
        train_lgbm_path, eval_lgbm_path,
    )
    labeled_lgbm = pl.read_parquet(train_lgbm_path)
    eval_lgbm = pl.read_parquet(eval_lgbm_path)

    # ── 4. Cascade cuts ──────────────────────────────────────────────────────
    n_ranker_train = int(cfg.get("n_ranker_train", 1023))
    n_ranker_eval = int(cfg.get("n_ranker_eval", 1500))
    log.info(
        "cascade: train top-%d / eval top-%d per user (LGBM stage-1)",
        n_ranker_train, n_ranker_eval,
    )
    labeled = _cascade_cut(labeled, labeled_lgbm, n_ranker_train)
    feats_eval = _cascade_cut(feats_eval, eval_lgbm, n_ranker_eval)
    log.info(
        "after cascade: labeled=%d rows, eval=%d rows",
        len(labeled), len(feats_eval),
    )
    n_null = int(labeled["lgbm_score"].is_null().sum())
    if n_null > 0:
        log.warning(
            "cascade: %d labeled rows have NULL lgbm_score after join — "
            "missing from cached scores parquet, will sort to bottom of cascade",
            n_null,
        )
    del labeled_lgbm, eval_lgbm

    # ── 5. CatBoost stage-2 fit on cascaded labeled ──────────────────────────
    df_train, df_val = _split_for_ranker(labeled, cfg.seed)
    log.info("ranker train=%d  val=%d", len(df_train), len(df_val))
    del labeled

    ranker = RankerModel(**cfg.ranker)
    ranker.fit(df_train, df_val)
    del df_train, df_val

    # ── 6. Eval Recall@100 on val + test ─────────────────────────────────────
    gt_val = pl.read_parquet(gt_val_path)
    gt_test = pl.read_parquet(gt_test_path)
    log.info(
        "val ground truth: %d pairs / %d users; test: %d pairs / %d users",
        len(gt_val), gt_val["uid"].n_unique(),
        len(gt_test), gt_test["uid"].n_unique(),
    )

    preds = ranker.predict(feats_eval, n=cfg.top_k)
    score_val = recall_at_k(gt_val, preds, k=cfg.top_k)
    log.info("val  Recall@%d = %.2f", cfg.top_k, score_val)
    score_test = recall_at_k(gt_test, preds, k=cfg.top_k)
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
        _append_results(
            results_path,
            {
                "run_id": run_id_ts,
                "model": f"ranker_topk[{cg_names}]",
                "dataset_size": cfg.data.size,
                "split": split_name,
                "score": round(float(score), 4),
                "config_path": "configs/ranker.yaml",
            },
        )
    log.info("results appended to %s", results_path)

    # ── 8. Optional feature importance ───────────────────────────────────────
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
        "DONE (refit-topk). base=%s run_id=%s val=%.2f test=%.2f. "
        "Run scripts/submit_ranker_topk.py to generate the submission CSV.",
        base_run_id, run_id, score_val, score_test,
    )


if __name__ == "__main__":
    main()
