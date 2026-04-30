"""Shared preprocessing utilities for CF models (ALS, ItemKNN)."""
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


def build_id_maps(
    df: pl.DataFrame,
    user_col: str = "uid",
    item_col: str = "item_id",
) -> tuple[dict, dict, dict, dict]:
    """Build integer index mappings for users and items.

    Returns:
        uid_map: uid -> row index
        item_map: item_id -> col index
        inv_uid_map: row index -> uid
        inv_item_map: col index -> item_id
    """
    uids = sorted(df[user_col].unique().to_list())
    items = sorted(df[item_col].unique().to_list())
    uid_map = {u: i for i, u in enumerate(uids)}
    item_map = {it: i for i, it in enumerate(items)}
    inv_uid_map = {i: u for u, i in uid_map.items()}
    inv_item_map = {i: it for it, i in item_map.items()}
    return uid_map, item_map, inv_uid_map, inv_item_map


def add_engagement_weights(
    df: pl.DataFrame,
    high: float = 3.0,
    mid: float = 1.0,
    low: float = 0.0,
    mid_threshold: float = 50.0,
    high_threshold: float = 80.0,
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Append a per-row interaction weight derived from played_ratio_pct.

    Three tiers:
        played_ratio_pct >  high_threshold  → ``high`` (default 3.0)
        mid_threshold < ratio ≤ high       → ``mid``  (default 1.0)
        ratio ≤ mid_threshold              → ``low``  (default 0.0 → row dropped)

    If ``low == 0``, low-engagement listens get zero confidence — equivalent
    to filtering them out. Set ``low > 0`` (typically 0.1–0.5) to feed weak
    positive signal from skipped tracks into ALS, instead of discarding.

    Negative ``low`` is rejected: implicit ALS treats values as confidence
    multipliers and negative entries break the PSD assumption of the solver.
    """
    if low < 0:
        raise ValueError(f"low engagement weight must be ≥ 0, got {low}")
    return df.with_columns(
        pl.when(pl.col("played_ratio_pct") > high_threshold)
        .then(pl.lit(high, dtype=pl.Float32))
        .when(pl.col("played_ratio_pct") > mid_threshold)
        .then(pl.lit(mid, dtype=pl.Float32))
        .otherwise(pl.lit(low, dtype=pl.Float32))
        .alias(weight_col)
    )


def build_csr_matrix(
    df: pl.DataFrame,
    uid_map: dict,
    item_map: dict,
    user_col: str = "uid",
    item_col: str = "item_id",
    weight_col: str | None = None,
) -> csr_matrix:
    """Build a user-item interaction matrix.

    If ``weight_col`` is given, that column drives the matrix values.
    Otherwise all observed interactions get value 1.0 (binary).

    The implicit library scales confidence as C = 1 + alpha * matrix,
    so non-uniform weights translate into per-interaction confidence.
    Duplicate (user, item) rows are summed by scipy's COO→CSR conversion.
    """
    rows = np.array([uid_map[u] for u in df[user_col].to_list()], dtype=np.int32)
    cols = np.array([item_map[it] for it in df[item_col].to_list()], dtype=np.int32)
    if weight_col is None:
        data = np.ones(len(rows), dtype=np.float32)
    else:
        data = df[weight_col].to_numpy().astype(np.float32)
    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(uid_map), len(item_map)),
    )
