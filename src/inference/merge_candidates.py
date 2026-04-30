"""Union candidates from multiple candidate generators.

Each CG returns a DataFrame with the schema:
    uid: Int64, item_id: Int64, score: Float32/64, {name}_rank: Int32

merge_candidates() outer-joins all of them on (uid, item_id) and renames
``score`` to ``{name}_score`` so each CG's contribution is preserved as a
distinct feature column. Missing entries become NULL — CatBoost can handle
that natively via ``nan_mode="Min"``.
"""
import logging
from functools import reduce

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
