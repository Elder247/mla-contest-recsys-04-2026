"""Phase 1 entrypoint — fit / load every CG into the pickle cache.

This is run as a subprocess by ``scripts/train_ranker.py`` and
``scripts/submit_ranker.py`` so the OS reclaims RSS at exit. Direct
invocation is also supported for debugging:

    python -u scripts/_phases/fit_cgs.py data=500m suffix='' train_cutoff_ts=12345
    python -u scripts/_phases/fit_cgs.py data=500m suffix=_full   # full data
"""
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference.phases import fit_phase
from src.utils import setup_logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="ranker", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("fit_cgs config:\n%s", OmegaConf.to_yaml(cfg))

    suffix = cfg.get("suffix", "")
    train_cutoff_ts = cfg.get("train_cutoff_ts", None)
    if train_cutoff_ts is not None:
        train_cutoff_ts = int(train_cutoff_ts)

    fit_phase(
        cg_cfg_list=cfg.candidate_generators,
        listens_path=cfg.data.listens,
        likes_path=cfg.data.likes,
        cache_dir=cfg.cg_cache_dir,
        size=cfg.data.size,
        suffix=suffix,
        force_refit=bool(cfg.get("force_refit_cg", False)),
        train_cutoff_ts=train_cutoff_ts,
    )


if __name__ == "__main__":
    main()
