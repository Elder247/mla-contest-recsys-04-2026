"""Shared helpers for the multi-CG ranker pipeline.

Used by both ``scripts/train_ranker.py`` (offline train + eval) and
``scripts/submit_ranker.py`` (submission generation). Phase A1 keeps the
feature set minimal — A2 will swap ``add_basic_features`` for the rich
LazyFrame implementation in ``src/data/features.py``.
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


def add_basic_features(candidates: pl.DataFrame, train: pl.DataFrame) -> pl.DataFrame:
    """Phase A1 placeholder — joins user_n_listens + item_pop only.

    A2 will replace this with the LazyFrame-based ``add_features`` from
    ``src/data/features.py`` (~50 features).
    """
    user_feats = (
        train
        .group_by("uid")
        .agg(pl.len().alias("user_n_listens"))
        .with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("user_n_listens").cast(pl.Int32),
        ])
    )
    item_feats = (
        train
        .group_by("item_id")
        .agg(pl.len().alias("item_pop"))
        .with_columns([
            pl.col("item_id").cast(pl.Int64),
            pl.col("item_pop").cast(pl.Int32),
        ])
    )
    return (
        candidates
        .join(user_feats, on="uid", how="left")
        .join(item_feats, on="item_id", how="left")
        .with_columns([
            pl.col("user_n_listens").fill_null(0).cast(pl.Int32),
            pl.col("item_pop").fill_null(0).cast(pl.Int32),
        ])
    )
