"""Train a candidate generation model.

Usage:
    python scripts/train.py model=pop data=50m
    python scripts/train.py model=als data=50m
"""
import logging
import pickle
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import load_listens, positive_listens
from src.data.splits import temporal_split
from src.utils import setup_logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    # Load and filter to positive listens
    log.info("loading listens from %s", cfg.data.listens)
    listens = load_listens(path=cfg.data.listens)
    listens = positive_listens(listens)
    log.info("positive listens: %d rows", len(listens))

    # Temporal split — train on everything before val window
    split = temporal_split(
        listens,
        val_days=cfg.split.val_days,
        gap_days=cfg.split.gap_days,
        timestamp_col=cfg.split.timestamp_col,
    )
    log.info(
        "split sizes — train: %d  val: %d  test: %d",
        len(split.train), len(split.val), len(split.test),
    )

    # Instantiate and train model
    model = hydra.utils.instantiate(cfg.model)
    log.info("training %s", type(model).__name__)
    model.fit(split.train)
    log.info("training done")

    # Persist model
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = cfg.model.get("name", type(model).__name__.lower())
    artifact_path = output_dir / f"{model_name}_{cfg.data.size}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(model, f)
    log.info("model saved to %s", artifact_path)


if __name__ == "__main__":
    main()
