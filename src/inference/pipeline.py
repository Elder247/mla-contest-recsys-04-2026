"""Shared helpers for the multi-CG ranker pipeline.

Used by both ``scripts/train_ranker.py`` (offline train + eval) and
``scripts/submit_ranker.py`` (submission generation). Feature engineering
itself lives in ``src/data/features.py`` (LazyFrame-based).
"""
from __future__ import annotations

import logging
from typing import Iterable

import polars as pl

from src.models.base import BaseModel

log = logging.getLogger(__name__)


def load_eval_users(users_csv: str) -> list[int]:
    return pl.read_csv(users_csv).get_column("uid").cast(pl.Int64).to_list()


def generate_candidates(
    cgs: Iterable[BaseModel],
    eval_users: list[int],
) -> dict[str, pl.DataFrame]:
    """Run ``recommend()`` on each CG; each CG decides its own ``n_cand``.

    Returns:
        Dict ``cg.name -> DataFrame`` ready to feed into ``merge_candidates``.
    """
    out: dict[str, pl.DataFrame] = {}
    for cg in cgs:
        n_cand = getattr(cg, "n_cand", 100)
        log.info("CG '%s': generating top-%d for %d users", cg.name, n_cand, len(eval_users))
        df = cg.recommend(eval_users, n=n_cand)
        log.info("CG '%s': %d candidate rows", cg.name, len(df))
        out[cg.name] = df
    return out


def apply_exclude_filter(
    candidates: pl.DataFrame,
    exclude: pl.DataFrame,
) -> pl.DataFrame:
    """Drop ``(uid, item_id)`` pairs that appear in ``exclude``.

    Used as a hard guard against recommending disliked items, which the
    ranker score-feature alone cannot guarantee. Anti-join is cheap:
    dislikes table is ~1M rows on 50M dataset.
    """
    return candidates.join(
        exclude.select(["uid", "item_id"]).with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ]).unique(),
        on=["uid", "item_id"],
        how="anti",
    )
