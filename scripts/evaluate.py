"""Evaluate a trained model on val and/or test splits.

Usage:
    python scripts/evaluate.py model=pop data=50m
"""
import csv
import logging
import pickle
from datetime import datetime
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import load_listens, positive_listens
from src.data.splits import temporal_split
from src.evaluation.metrics import recall_at_k
from src.utils import setup_logging

log = logging.getLogger(__name__)


def _load_users(users_csv: str) -> list[int]:
    return (
        pl.read_csv(users_csv)
        .get_column("uid")
        .cast(pl.Int64)
        .to_list()
    )


def _ground_truth(df: pl.DataFrame, users: list[int]) -> pl.DataFrame:
    return (
        df
        .select(["uid", "item_id"])
        .with_columns([
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
        ])
        .filter(pl.col("uid").is_in(users))
        .unique()
    )


def _append_results(results_path: Path, row: dict) -> None:
    exists = results_path.exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@hydra.main(config_path="../configs", config_name="evaluate", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    # Load model
    model_name = cfg.model.get("name", "model")
    artifact_path = Path(cfg.output_dir) / f"{model_name}_{cfg.data.size}.pkl"
    log.info("loading model from %s", artifact_path)
    with open(artifact_path, "rb") as f:
        model = pickle.load(f)

    # Load data
    listens = positive_listens(load_listens(cfg.data.size))
    split = temporal_split(
        listens,
        val_days=cfg.split.val_days,
        gap_days=cfg.split.gap_days,
        timestamp_col=cfg.split.timestamp_col,
    )

    eval_users = _load_users(cfg.data.users_csv)
    top_k: int = cfg.top_k

    results_path = Path(cfg.output_dir) / "results.csv"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for split_name in cfg.splits:
        split_df: pl.DataFrame = getattr(split, split_name)
        ground_truth = _ground_truth(split_df, eval_users)

        preds = model.recommend(users=eval_users, n=top_k)
        score = recall_at_k(ground_truth, preds, k=top_k)
        log.info("%s  Recall@%d = %.2f", split_name, top_k, score)

        _append_results(results_path, {
            "run_id": run_id,
            "model": model_name,
            "dataset_size": cfg.data.size,
            "split": split_name,
            "val_score": score if split_name == "val" else "",
            "test_score": score if split_name == "test" else "",
            "config_path": f"configs/model/{model_name}.yaml",
        })

    log.info("results appended to %s", results_path)


if __name__ == "__main__":
    main()
