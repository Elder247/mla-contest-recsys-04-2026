"""Generate a submission CSV from a trained model.

Output format:
    uid,item_ids
    100,6 7 6767

Usage:
    python scripts/make_submission.py model=pop data=50m run_id=001
"""
import logging
import pickle
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import load_listens, positive_listens
from src.utils import setup_logging

log = logging.getLogger(__name__)


def _load_users(users_csv: str) -> list[int]:
    return (
        pl.read_csv(users_csv)
        .get_column("uid")
        .cast(pl.Int64)
        .to_list()
    )


def _format_submission(preds: pl.DataFrame, top_k: int) -> pl.DataFrame:
    """Convert uid/item_id/score DataFrame to submission format."""
    if "score" in preds.columns:
        preds = preds.sort("score", descending=True)

    return (
        preds
        .group_by("uid")
        .head(top_k)
        .sort(["uid", "score"] if "score" in preds.columns else ["uid"])
        .group_by("uid")
        .agg(
            pl.col("item_id")
            .cast(pl.Utf8)
            .str.concat(delimiter=" ")
            .alias("item_ids")
        )
        .sort("uid")
    )


@hydra.main(config_path="../configs", config_name="submit", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    model_name = cfg.model.get("name", "model")
    artifact_path = Path(cfg.output_dir) / f"{model_name}_{cfg.data.size}.pkl"
    log.info("loading model from %s", artifact_path)
    with open(artifact_path, "rb") as f:
        model = pickle.load(f)

    eval_users = _load_users(cfg.data.users_csv)
    log.info("generating recommendations for %d users", len(eval_users))

    preds = model.recommend(users=eval_users, n=cfg.top_k)

    submission = _format_submission(preds, top_k=cfg.top_k)
    log.info("submission rows: %d", len(submission))

    # Validate format
    missing = set(eval_users) - set(submission["uid"].cast(pl.Int64).to_list())
    if missing:
        log.warning("%d users have no predictions", len(missing))

    run_id = str(cfg.run_id)
    out_path = Path(cfg.submission_dir) / f"sub_{run_id}_{model_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(out_path)
    log.info("submission saved to %s", out_path)


if __name__ == "__main__":
    main()
