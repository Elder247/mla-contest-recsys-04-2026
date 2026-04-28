import polars as pl
from pathlib import Path

DATA_ROOT = Path("data")


def load_listens(size: str = "50m") -> pl.DataFrame:
    return pl.read_parquet(DATA_ROOT / size / "listens.parquet")

def load_likes(size: str = "50m") -> pl.DataFrame:
    return pl.read_parquet(DATA_ROOT / size / "likes.parquet")

def load_dislikes(size: str = "50m") -> pl.DataFrame:
    return pl.read_parquet(DATA_ROOT / size / "dislikes.parquet")

def load_unlikes(size: str = "50m") -> pl.DataFrame:
    return pl.read_parquet(DATA_ROOT / size / "unlikes.parquet")

def load_undislikes(size: str = "50m") -> pl.DataFrame:
    return pl.read_parquet(DATA_ROOT / size / "undislikes.parquet")

def load_multi_event(size: str = "50m") -> pl.DataFrame:
    """Все события в одном файле: listens + likes + dislikes + unlikes + undislikes."""
    return pl.read_parquet(DATA_ROOT / size / "multi_event.parquet")

def load_embeddings() -> pl.DataFrame:
    """Аудио-эмбеддинги треков — общие для всех размеров датасета."""
    return pl.read_parquet(DATA_ROOT / "embeddings.parquet")

def positive_listens(df: pl.DataFrame) -> pl.DataFrame:
    """Фильтрует положительные прослушивания: played_ratio_pct > 50."""
    return df.filter(pl.col("played_ratio_pct") > 50)
