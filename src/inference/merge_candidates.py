"""Union candidates from multiple candidate generators.

Each CG returns a DataFrame with the schema:
    uid: Int64, item_id: Int64, score: Float32/64, {name}_rank: Int32

merge_candidates() outer-joins all of them on (uid, item_id) and renames
``score`` to ``{name}_score`` so each CG's contribution is preserved as a
distinct feature column. Missing entries become NULL — CatBoost can handle
that natively via ``nan_mode="Min"``.

apply_n_cand_keep() is an OPTIONAL post-merge row filter that emulates the
``n_cand_keep`` semantics learnt by the joint_v2 Optuna study:

  - Each CG generates a *pool* of candidates (``n_cand`` in the yaml).
  - After the outer-join, rows are kept iff at least one CG's rank is ≤
    that CG's ``n_cand_keep`` value (which is ≤ ``n_cand``).
  - CGs with ``n_cand_keep == 0`` contribute no unique rows but their
    ``{name}_rank``/``{name}_score`` columns still enrich rows that other
    CGs put through. This matches the keep_expr used by Optuna trials and
    lets the production ranker see the same dense feature distribution.

The filter is a no-op when no CG block has the ``n_cand_keep`` field, so
existing pipelines (and configs without the field) keep working unchanged.
"""
from __future__ import annotations

import logging
from functools import reduce
from typing import Any, Iterable

import polars as pl

log = logging.getLogger(__name__)


def merge_candidates(
    cg_dataframes: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Outer-join candidate DataFrames from multiple CGs.

    Args:
        cg_dataframes: mapping of ``cg_name -> recommend()`` output. Each input
            DataFrame must contain ``uid``, ``item_id``, ``score`` and
            ``{cg_name}_rank``.

    Returns:
        DataFrame with columns:
            uid, item_id, {name}_score, {name}_rank for each name in input.
        Within a single CG, duplicate (uid, item_id) rows are reduced to the
        best (lowest) rank before joining.
    """
    if not cg_dataframes:
        raise ValueError("merge_candidates: empty cg_dataframes")

    normalized = []
    for name, df in cg_dataframes.items():
        rank_col = f"{name}_rank"
        score_col = f"{name}_score"
        if rank_col not in df.columns:
            raise ValueError(
                f"merge_candidates: CG '{name}' missing required column '{rank_col}'. "
                f"Got columns: {df.columns}"
            )
        if "score" not in df.columns:
            raise ValueError(
                f"merge_candidates: CG '{name}' missing 'score' column. "
                f"Got columns: {df.columns}"
            )

        deduped = (
            df
            .with_columns([
                pl.col("uid").cast(pl.Int64),
                pl.col("item_id").cast(pl.Int64),
                pl.col("score").alias(score_col),
                pl.col(rank_col).cast(pl.Int32),
            ])
            .group_by(["uid", "item_id"], maintain_order=False)
            .agg([
                pl.col(rank_col).min(),
                pl.col(score_col).max(),
            ])
            .select(["uid", "item_id", score_col, rank_col])
        )
        log.info(
            "merge_candidates: CG '%s' contributes %d (uid, item_id) pairs",
            name, len(deduped),
        )
        normalized.append(deduped)

    merged = reduce(
        lambda left, right: left.join(right, on=["uid", "item_id"], how="full", coalesce=True),
        normalized,
    )

    log.info(
        "merge_candidates: merged %d (uid, item_id) pairs from %d CGs",
        len(merged), len(cg_dataframes),
    )
    return merged


def apply_n_cand_keep(
    merged: pl.DataFrame,
    cg_cfg_list: Iterable[Any],
) -> pl.DataFrame:
    """Drop rows where no CG has rank ≤ its ``n_cand_keep``.

    Acts as a post-merge replacement for "what if each CG had been called
    with ``n_cand=n_cand_keep`` instead of ``n_cand=n_cand``". The pool is
    still big (so per-CG rank/score columns are densely populated and the
    ranker sees rich features), but rows that wouldn't have made it into
    any CG's top-``n_cand_keep`` are dropped — matching joint_v2 Optuna's
    keep_expr semantics.

    Args:
        merged: outer-joined candidate DataFrame from :func:`merge_candidates`.
            Must contain ``{name}_rank`` for every CG in ``cg_cfg_list``.
        cg_cfg_list: iterable of CG config blocks (Hydra DictConfig or plain
            dict). Each block may have an optional ``n_cand_keep`` integer:
              - field absent on every block → no-op (returns ``merged``).
              - ``n_cand_keep > 0`` → row survives if ``{name}_rank`` ≤ value.
              - ``n_cand_keep == 0`` → CG contributes no unique rows (its
                rank/score columns still enrich rows kept by other CGs).
              - field absent on some blocks but set on others → the absent
                blocks contribute no row-keep predicate either (treated as
                "this CG doesn't gate the pool"). To opt back in, set
                ``n_cand_keep`` equal to ``n_cand``.

    Returns:
        Filtered DataFrame with the same columns as ``merged``. When the
        field is set on every CG and at least one row survives every CG's
        check, output is a strict subset of ``merged`` rows.
    """
    has_field = False
    keep_terms = []
    for cg_cfg in cg_cfg_list:
        if "n_cand_keep" not in cg_cfg:
            continue
        has_field = True
        n_keep = cg_cfg["n_cand_keep"]
        if n_keep is None or n_keep <= 0:
            # CG with n_cand_keep=0 contributes no unique rows — its rank
            # column still survives in the dataframe to enrich features
            # on rows kept by other CGs (which is the whole point).
            continue
        name = cg_cfg.get("name")
        rank_col = f"{name}_rank"
        if rank_col not in merged.columns:
            raise ValueError(
                f"apply_n_cand_keep: CG '{name}' has n_cand_keep={n_keep} "
                f"but '{rank_col}' is not in merged columns "
                f"({list(merged.columns)})"
            )
        keep_terms.append(
            pl.col(rank_col).is_not_null() & (pl.col(rank_col) <= int(n_keep))
        )

    if not has_field:
        log.info(
            "apply_n_cand_keep: no CG has the field — returning merged unchanged"
        )
        return merged

    if not keep_terms:
        # At least one CG had the field, but every value was 0 / None.
        # That asks for an empty pool, which is never useful.
        raise ValueError(
            "apply_n_cand_keep: every CG with 'n_cand_keep' set was 0 — "
            "no rows would survive. Set n_cand_keep > 0 for at least one CG."
        )

    keep_expr = reduce(lambda a, b: a | b, keep_terms)
    before = len(merged)
    filtered = merged.filter(keep_expr)
    log.info(
        "apply_n_cand_keep: filtered %d → %d rows (dropped %d) using %d active CGs",
        before, len(filtered), before - len(filtered), len(keep_terms),
    )
    return filtered


def compute_cg_aggregates(
    merged: pl.DataFrame,
    cg_cfg_list: Iterable[Any],
) -> pl.DataFrame:
    """Append per-row aggregate features over the CG ``{name}_rank`` /
    ``{name}_score`` columns. Adds 2 Float32/Int32 columns:

      - ``cg_count``           — number of CGs that contributed this row
      - ``cg_mean_score_norm`` — mean of MinMax-normalized scores (per-CG
                                 normalization on the visible pool; only
                                 the aggregate is normalized — original
                                 ``{name}_score`` columns are untouched)

    NOTE on removed rank-derived aggregates (``cg_min_rank``,
    ``cg_max_rank``, ``cg_mean_rank``, ``cg_rrf_score``): they used to be
    here but were a textbook **sample-selection leak**. ``RankerModel.fit``
    on GPU caps train+val pools to 1023 rows/uid keyed by
    ``sum(1/(60+rank_i))`` over every ``*_rank`` column — which is
    ``cg_rrf_score`` verbatim, and ``cg_min_rank`` correlates with it
    monotonically. Adding those as explicit features handed CatBoost the
    exact selection criterion in one column → it learned the inverse of
    the true label distribution (low ``cg_rrf_score`` items survive in
    train *only* if labelled positive, so the model predicted "low
    ``cg_rrf_score`` → positive"; at inference the eval pool has the
    full long tail of low-RRF items, all of which the model then
    incorrectly ranks high → top-100 pollution → recall ≈ random).
    Empirically: feature_importance v3_features (with rank aggregates):
    cg_rrf_score=21.9, cg_min_rank=14.0, cg_mean_rank=10.7,
    cg_max_rank=2.8, with val/test Recall@100 ≈ 10 (vs 319 baseline
    without these features). ``cg_count`` and ``cg_mean_score_norm`` do
    not mirror the trim formula and are kept.

    Polars' ``*_horizontal`` reductions skip nulls natively; the count is
    derived from the same masks for consistency. Aggregates are intentionally
    computed *after* :func:`apply_n_cand_keep` so feature distributions
    match what the production ranker will see.
    """
    cg_names = [cg.get("name") for cg in cg_cfg_list]
    rank_cols = [f"{n}_rank" for n in cg_names if f"{n}_rank" in merged.columns]
    score_cols = [f"{n}_score" for n in cg_names if f"{n}_score" in merged.columns]
    if not rank_cols:
        log.info("compute_cg_aggregates: no rank columns in merged — skipping")
        return merged

    log.info(
        "compute_cg_aggregates: %d rank cols, %d score cols on %d rows",
        len(rank_cols), len(score_cols), len(merged),
    )

    not_null_terms = [pl.col(c).is_not_null().cast(pl.Int32) for c in rank_cols]
    out = merged.with_columns([
        pl.sum_horizontal(*not_null_terms).cast(pl.Int32).alias("cg_count"),
    ])

    # ── MinMax-normalised score aggregate ────────────────────────────────────
    # Per-CG: (score - min) / (max - min), null preserved. Then mean across CGs
    # ignoring nulls. Skip CGs whose score column is fully null or constant
    # (no meaningful normalisation).
    norm_terms: list[pl.Expr] = []
    for sc in score_cols:
        s_min = out[sc].min()
        s_max = out[sc].max()
        if s_min is None or s_max is None or s_min == s_max:
            continue
        norm_terms.append(
            (pl.col(sc).cast(pl.Float64) - float(s_min))
            / (float(s_max) - float(s_min))
        )
    if norm_terms:
        out = out.with_columns(
            pl.mean_horizontal(*norm_terms).alias("cg_mean_score_norm")
        ).with_columns(pl.col("cg_mean_score_norm").cast(pl.Float32))
    else:
        out = out.with_columns(
            pl.lit(None).cast(pl.Float32).alias("cg_mean_score_norm")
        )

    return out


def cg_recall(
    candidates: pl.DataFrame,
    ground_truth: pl.DataFrame,
) -> float:
    """Recall@∞ — share of GT items covered by the union of candidates.

    Returned in the contest scale (x 1000) so it lines up with Recall@100
    figures in experiment-log.md. This is the upper bound the downstream
    ranker can achieve: if it is low, the bottleneck is candidate
    generation, not the ranker.

    Args:
        candidates: DataFrame with at least ``uid``, ``item_id``.
        ground_truth: DataFrame with ``uid``, ``item_id`` representing
            held-out positives.

    Returns:
        Mean per-user recall x 1000 on uids that appear in ground_truth.
    """
    cand = (
        candidates
        .select(["uid", "item_id"])
        .with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
        .unique()
    )
    gt = (
        ground_truth
        .select(["uid", "item_id"])
        .with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
        .unique()
    )
    per_user = (
        gt
        .join(
            cand.with_columns(pl.lit(1, dtype=pl.Int8).alias("hit")),
            on=["uid", "item_id"],
            how="left",
        )
        .with_columns(pl.col("hit").fill_null(0))
        .group_by("uid")
        .agg([
            pl.col("hit").sum().alias("n_hits"),
            pl.len().alias("n_gt"),
        ])
        .with_columns(
            (pl.col("n_hits").cast(pl.Float64) / pl.col("n_gt").cast(pl.Float64)).alias("recall")
        )
    )
    return float(per_user["recall"].mean()) * 1000.0
