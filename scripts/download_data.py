"""Download Yambda dataset files for a given dataset_size.

Usage:
    python scripts/download_data.py dataset_size=50m
"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.utils import setup_logging

HF_BASE = "https://huggingface.co/datasets/yandex/yambda/resolve/main"

SIZE_FILES = [
    "listens.parquet",
    "likes.parquet",
    "dislikes.parquet",
    "unlikes.parquet",
    "undislikes.parquet",
    "multi_event.parquet",
]

COMMON_FILES = [
    "embeddings.parquet",
    "album_item_mapping.parquet",
    "artist_item_mapping.parquet",
]

log = logging.getLogger(__name__)


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("already exists, skipping: %s", dest)
        return
    log.info("downloading %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    log.info("done: %s", dest)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    size: str = cfg.data.size
    data_root = Path(cfg.data.root)

    for fname in SIZE_FILES:
        url = f"{HF_BASE}/flat/{size}/{fname}"
        dest = data_root / size / fname
        _download(url, dest)

    for fname in COMMON_FILES:
        url = f"{HF_BASE}/{fname}"  # common files are at repo root, not /flat/
        dest = data_root / fname
        _download(url, dest)

    log.info("all files downloaded for dataset_size=%s", size)


if __name__ == "__main__":
    main()
