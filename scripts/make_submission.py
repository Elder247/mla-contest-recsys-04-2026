"""Generate a submission CSV from a model trained on ALL available data.

Trains from scratch on all positive listens (no temporal split), matching the
notebook baseline approach. This gives the best possible submission score.

Output format:
    uid,item_ids
    100,6 7 6767

Usage:
    python scripts/make_submission.py model=pop data=50m run_id=001
"""
import logging
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
        preds = preds.sort(["uid", "score"], descending=[False, True])

    return (
        preds
        .group_by("uid")
        .head(top_k)
        .group_by("uid")
        .agg(
            pl.col("item_id")
            .cast(pl.Utf8)
            .str.join(delimiter=" ")
            .alias("item_ids")
        )
        .sort("uid")
    )


@hydra.main(config_path="../configs", config_name="submit", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    # Train on ALL data — matches notebook, gives best submission score
    log.info("loading all positive listens from %s", cfg.data.size)
    listens = positive_listens(load_listens(path=cfg.data.listens))
    log.info("total positive listens: %d", len(listens))

    model = hydra.utils.instantiate(cfg.model)
    model_name: str = cfg.model.get("name", type(model).__name__.lower())
    log.info("training %s on full dataset", model_name)
    model.fit(listens)

    eval_users = _load_users(cfg.data.users_csv)
    log.info("generating recommendations for %d users", len(eval_users))

    preds = model.recommend(users=eval_users, n=cfg.top_k)

    submission = _format_submission(preds, top_k=cfg.top_k)
    log.info("submission rows: %d", len(submission))

    missing = set(eval_users) - set(submission["uid"].cast(pl.Int64).to_list())
    if missing:
        log.warning("%d users have no predictions", len(missing))

    sub_dir = Path(cfg.submission_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(cfg.run_id)
    archive_path = sub_dir / f"sub_{run_id}_{model_name}.csv"
    submission.write_csv(archive_path)
    log.info("submission saved to %s", archive_path)

    test_path = sub_dir / "test.csv"
    submission.write_csv(test_path)
    log.info("test.csv updated → %s", test_path)


if __name__ == "__main__":
    main()
