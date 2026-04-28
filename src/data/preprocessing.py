"""Shared preprocessing utilities for CF models (ALS, BPR, LightFM)."""
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


def build_csr_matrix(
    df: pl.DataFrame,
    uid_map: dict,
    item_map: dict,
    user_col: str = "uid",
    item_col: str = "item_id",
) -> csr_matrix:
    """Build a binary user-item interaction matrix.

    All observed interactions get value 1.0.
    The implicit library scales confidence as C = 1 + alpha * matrix,
    so alpha controls engagement strength.
    """
    rows = np.array([uid_map[u] for u in df[user_col].to_list()], dtype=np.int32)
    cols = np.array([item_map[it] for it in df[item_col].to_list()], dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(uid_map), len(item_map)),
    )
