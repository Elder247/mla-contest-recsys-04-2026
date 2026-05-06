"""Blend multiple submission CSVs via Reciprocal Rank Fusion (RRF).

For each (uid, item_id), score = Σ w_i / (k + rank_i + 1) over every
input submission that contains it (rank 0-indexed within a row). Per uid
we take the top-K items by aggregated score and write a new submission.

Usage:
    python scripts/blend_submissions.py \\
        --inputs submissions/sub_server_005_ranker.csv:0.5 \\
                 submissions/sub_server_004_ranker.csv:0.3 \\
                 submissions/sub_server_003_ranker.csv:0.2 \\
        --output submissions/sub_blend_v1.csv

Always prints per-pair overlap stats (Jaccard on top-K item sets) — if two
inputs overlap >70% they're near-duplicates, and blending is unlikely to
help. Validates the output against ``submissions/users.csv`` before exit.

Notes:
- ``rank`` is the position within the comma-separated list (0-based);
  the ``rrf_k`` offset (default 60) flattens the head of the curve.
- Missing items in some submissions don't break anything — they simply
  contribute 0 to the score for that submission.
- Weight is multiplicative on the per-sub score; absolute scale doesn't
  matter (top-K is invariant to a global rescale).
"""
from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import polars as pl

from src.inference.validate_submission import validate_submission
from src.utils import setup_logging

log = logging.getLogger(__name__)


def _parse_input(spec: str) -> tuple[Path, float]:
    """Parse ``path:weight`` form. Defaults to weight 1.0 when omitted."""
    if ":" in spec:
        path_str, weight_str = spec.rsplit(":", 1)
        return Path(path_str), float(weight_str)
    return Path(spec), 1.0


def _read_sub_long(path: Path) -> pl.DataFrame:
    """Read submission CSV and return long-format ``(uid, item_id, rank)``."""
    df = pl.read_csv(path, schema_overrides={"uid": pl.Int64})
    return (
        df
        .with_columns(pl.col("item_ids").cast(pl.Utf8).str.split(" ").alias("_items"))
        .with_row_index("_uid_idx")
        .explode("_items")
        .with_columns([
            pl.col("_items").cast(pl.Int64).alias("item_id"),
            pl.int_range(0, pl.len()).over("_uid_idx").cast(pl.Int32).alias("rank"),
        ])
        .select(["uid", "item_id", "rank"])
    )


def _overlap_stats(subs: list[tuple[Path, pl.DataFrame]]) -> None:
    """Log per-pair Jaccard overlap on top-K item sets."""
    sets = {p.name: set(zip(d["uid"].to_list(), d["item_id"].to_list())) for p, d in subs}
    log.info("pairwise (uid, item_id) Jaccard overlap:")
    for (a, sa), (b, sb) in combinations(sets.items(), 2):
        inter = len(sa & sb)
        union = len(sa | sb)
        jacc = inter / union if union else 0.0
        log.info("  %-50s vs %-50s  J=%.3f (|∩|=%d, |∪|=%d)", a, b, jacc, inter, union)


def blend(
    inputs: list[tuple[Path, float]],
    rrf_k: int = 60,
    top_k: int = 100,
) -> pl.DataFrame:
    """RRF-blend ``inputs`` and return a submission DataFrame ``(uid, item_ids)``."""
    parts = []
    parsed = []
    for path, weight in inputs:
        if not path.exists():
            raise FileNotFoundError(f"submission not found: {path}")
        long = _read_sub_long(path)
        log.info("loaded %s (rows=%d, weight=%.3f)", path, long["uid"].n_unique(), weight)
        parsed.append((path, long))
        parts.append(
            long.with_columns(
                (weight / (rrf_k + pl.col("rank").cast(pl.Float64) + 1.0)).alias("score")
            )
        )

    _overlap_stats(parsed)

    pooled = (
        pl.concat(parts)
        .group_by(["uid", "item_id"], maintain_order=False)
        .agg(pl.col("score").sum())
    )

    top = (
        pooled
        .sort(by=["uid", "score"], descending=[False, True])
        .group_by("uid", maintain_order=True)
        .head(top_k)
    )

    return (
        top
        .group_by("uid", maintain_order=True)
        .agg(pl.col("item_id").cast(pl.Utf8).str.join(" ").alias("item_ids"))
        .sort("uid")
    )


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="space-separated list of ``path:weight`` (weight defaults to 1.0)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--users-csv", type=Path, default=Path("submissions/users.csv"),
        help="path to users.csv for output validation",
    )
    args = parser.parse_args()

    inputs = [_parse_input(s) for s in args.inputs]
    log.info("blending %d submissions into %s (rrf_k=%d, top_k=%d)",
             len(inputs), args.output, args.rrf_k, args.top_k)

    out = blend(inputs, rrf_k=args.rrf_k, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.output)
    log.info("wrote %s (%d rows)", args.output, len(out))

    report = validate_submission(args.output, args.users_csv, max_items=args.top_k)
    log.info("\n%s", report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
