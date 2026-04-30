from pathlib import Path

import polars as pl

DATA_ROOT = Path("data")


# ---------------------------------------------------------------------------
# Loaders — accept either an explicit path or (size, data_root) pair
# ---------------------------------------------------------------------------

def load_listens(size: str = "50m", path: str | Path | None = None) -> pl.DataFrame:
    return pl.read_parquet(path or DATA_ROOT / size / "listens.parquet")


def load_likes(size: str = "50m", path: str | Path | None = None) -> pl.DataFrame:
    return pl.read_parquet(path or DATA_ROOT / size / "likes.parquet")


def load_dislikes(size: str = "50m", path: str | Path | None = None) -> pl.DataFrame:
    return pl.read_parquet(path or DATA_ROOT / size / "dislikes.parquet")


def load_unlikes(size: str = "50m", path: str | Path | None = None) -> pl.DataFrame:
    return pl.read_parquet(path or DATA_ROOT / size / "unlikes.parquet")


def load_undislikes(size: str = "50m", path: str | Path | None = None) -> pl.DataFrame:
    return pl.read_parquet(path or DATA_ROOT / size / "undislikes.parquet")


def load_multi_event(size: str = "50m", path: str | Path | None = None) -> pl.DataFrame:
    """All event types in one file: listens + likes + dislikes + unlikes + undislikes."""
    return pl.read_parquet(path or DATA_ROOT / size / "multi_event.parquet")


def load_embeddings(path: str | Path | None = None) -> pl.DataFrame:
    """Audio embeddings — shared across all dataset sizes."""
    return pl.read_parquet(path or DATA_ROOT / "embeddings.parquet")


def load_album_item_mapping(path: str | Path | None = None) -> pl.DataFrame:
    return pl.read_parquet(path or DATA_ROOT / "album_item_mapping.parquet")


def load_artist_item_mapping(path: str | Path | None = None) -> pl.DataFrame:
    return pl.read_parquet(path or DATA_ROOT / "artist_item_mapping.parquet")


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def positive_listens(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only listens where played_ratio_pct > 50."""
    return df.filter(pl.col("played_ratio_pct") > 50)


def effective_dislikes(
    dislikes: pl.DataFrame,
    undislikes: pl.DataFrame,
) -> pl.DataFrame:
    """Active dislikes only — those NOT reverted by a later undislike.

    Per (uid, item_id) pair, take max dislike_ts and max undislike_ts.
    If undislike_ts ≥ dislike_ts → user changed their mind, drop the pair.

    Returns a DataFrame with two columns: ``uid``, ``item_id``.
    Use this for the dislike post-merge filter so we don't suppress tracks
    the user has explicitly reactivated.
    """
    last_dis = (
        dislikes
        .group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").max().alias("dislike_ts"))
    )
    last_undis = (
        undislikes
        .group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").max().alias("undislike_ts"))
    )
    return (
        last_dis
        .join(last_undis, on=["uid", "item_id"], how="left")
        .filter(
            pl.col("undislike_ts").is_null()
            | (pl.col("dislike_ts") > pl.col("undislike_ts"))
        )
        .select(["uid", "item_id"])
    )


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def to_sequential(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    extra_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Convert flat interactions to per-user chronological sequences.

    Returns a DataFrame with one row per user:
        uid | item_ids | timestamps | [extra_cols as lists]

    Useful for SASRec / GRU4Rec which expect sequential input.
    """
    agg = [
        pl.col("item_id").alias("item_ids"),
        pl.col(timestamp_col).alias("timestamps"),
    ]
    if extra_cols:
        agg += [pl.col(c) for c in extra_cols]

    return (
        df
        .sort(["uid", timestamp_col])
        .group_by("uid")
        .agg(agg)
        .sort("uid")
    )
