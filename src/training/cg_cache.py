"""Pickle cache for fitted candidate generators.

Two cache slots per CG:
- ``artifacts/cg/{name}_{size}.pkl``      — fitted on split.train (train_ranker)
- ``artifacts/cg/{name}_{size}_full.pkl`` — fitted on full data   (submit_ranker)

Re-using these avoids the dominant cost of repeated experiments
(eg. running train_ranker.py with different ranker hyperparams should not
re-fit ALS each time).
"""
import logging
import pickle
from pathlib import Path

import polars as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.models.base import BaseModel

log = logging.getLogger(__name__)

# Keys that live alongside model __init__ args in candidate-generator yamls
# but are pipeline-level metadata, not model parameters. Stripped before
# Hydra instantiate so the model class doesn't see them as kwargs.
_META_KEYS = {"data_source"}


def cg_cache_path(
    name: str,
    size: str,
    suffix: str = "",
    cache_dir: str | Path = "artifacts/cg",
) -> Path:
    """Filesystem location for a CG pickle.

    suffix=""       → split.train fit
    suffix="_full"  → full-data fit (used by submission)
    """
    return Path(cache_dir) / f"{name}_{size}{suffix}.pkl"


def fit_or_load_cg(
    cg_cfg: DictConfig,
    train_df: pl.DataFrame,
    size: str,
    suffix: str = "",
    force_refit: bool = False,
    cache_dir: str | Path = "artifacts/cg",
) -> BaseModel:
    """Either load a pickled CG from cache or fit a fresh one.

    The CG name is read from ``cg_cfg.name``; this must match between the
    config and the model class (each BaseModel exposes ``self.name``).

    Args:
        cg_cfg: Hydra config for the CG (must include ``_target_`` and ``name``).
        train_df: DataFrame to fit on if cache miss / force_refit.
        size: Dataset size ("50m" / "500m" / "5b") — kept in the filename so
            artifacts from different scales never collide.
        suffix: ``""`` for split.train fit, ``"_full"`` for full-data fit.
        force_refit: If True, ignore any existing cache and refit.

    Returns:
        Fitted BaseModel instance.
    """
    name = cg_cfg.get("name")
    if name is None:
        raise ValueError(f"cg_cfg missing 'name' field: {cg_cfg}")

    path = cg_cache_path(name, size, suffix, cache_dir)
    if path.exists() and not force_refit:
        log.info("loading cached CG '%s' from %s", name, path)
        with open(path, "rb") as f:
            return pickle.load(f)

    log.info("fitting CG '%s' (cache %s, force_refit=%s)",
             name, "miss" if not path.exists() else "ignored", force_refit)
    cfg_for_init = OmegaConf.create({
        k: v for k, v in OmegaConf.to_container(cg_cfg, resolve=True).items()
        if k not in _META_KEYS
    })
    cg: BaseModel = instantiate(cfg_for_init)
    cg.fit(train_df)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cg, f)
    log.info("saved CG '%s' to %s", name, path)
    return cg
