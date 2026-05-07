"""Fit a single candidate generator on FULL event data and pickle it
under the ``_full`` cache slot.

Usage:
    python -u scripts/fit_cg_full.py model=esasrec data=500m

Reads the CG config from ``configs/model/{name}.yaml`` (the same files used
by ``scripts/train.py``) but, unlike train.py:
  * does NOT apply a temporal split — the model sees every positive listen
    (or every like, when ``data_source: likes`` is on the CG config);
  * writes to ``artifacts/cg/{name}_{size}_full.pkl`` — the cache slot
    consumed by ``scripts/submit_ranker.py`` at submission time.

Use this whenever you need to refresh just one CG's full-data pickle
without re-fitting the whole pipeline (e.g. after honest eSASRec retrain).

Add ``+force_refit=true`` to ignore an existing pickle.

Note on ``data_source``: configs/ranker.yaml carries ``data_source: likes``
on RecentLikesModel, but configs/model/recent_likes.yaml does NOT. If you
ever fit recent_likes via this script, pass ``+data_source=likes`` on the
CLI explicitly. eSASRec / ALS / ItemKNN / pop / artist_pop / audio_knn /
repeat all use listens (the default) — no extra flag needed.
"""
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import load_likes, load_listens, positive_listens
from src.training.cg_cache import cg_cache_path, fit_or_load_cg
from src.utils import setup_logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    name = cfg.model.get("name")
    if name is None:
        raise ValueError("cfg.model is missing required field 'name'")

    # Allow override via either the CG config (configs/ranker.yaml carries
    # ``data_source: likes`` on RecentLikesModel) or a top-level Hydra flag
    # (``+data_source=likes`` on the CLI when the model yaml lacks it).
    data_source = cfg.get("data_source") or cfg.model.get("data_source", "listens")
    if data_source == "listens":
        train_df = positive_listens(load_listens(path=cfg.data.listens))
    elif data_source == "likes":
        train_df = load_likes(path=cfg.data.likes)
    else:
        raise ValueError(
            f"unsupported data_source '{data_source}' on CG '{name}'; "
            f"expected one of: listens, likes"
        )
    log.info("loaded full %s for '%s': %d rows", data_source, name, len(train_df))

    target_path = cg_cache_path(name, cfg.data.size, "_full", "/home/astrofimuk/dc-remote/artifacts/cg")
    log.info("fitting CG '%s' → %s", name, target_path)

    fit_or_load_cg(
        cg_cfg=cfg.model,
        train_df=train_df,
        size=cfg.data.size,
        suffix="_full",
        force_refit=bool(cfg.get("force_refit", False)),
        cache_dir="/home/astrofimuk/dc-remote/artifacts/cg",
    )
    log.info("done — %s ready", target_path)


if __name__ == "__main__":
    main()
