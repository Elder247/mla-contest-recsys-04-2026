"""Phase 2 entrypoint — generate candidates per CG and merge.

Reads cached CG pickles (suffix selects "" or "_full"), calls ``recommend``
once per CG, writes per-CG parquet to ``output_dir`` and finally a single
``merged.parquet``. Optional dislike anti-join is applied before the merge
write.

Direct invocation:

    python -u scripts/_phases/gen_candidates.py \\
        data=500m suffix='' kind=val output_dir=artifacts/candidates/run_001/val \\
        users_source=eval_users.csv [filter_dislikes=true dislike_cutoff_ts=12345]

CLI overrides:

- ``users_source``: CSV path with a ``uid`` column. Required.
- ``output_dir``:    where per-CG parquets and merged.parquet go.
- ``suffix``:        CG cache slot ("" or "_full").
- ``filter_dislikes`` (bool): apply the dislike anti-join.
- ``dislike_cutoff_ts`` (int|null): if set, only dislikes/undislikes with
  ``timestamp <= cutoff`` are considered (train_ranker offline); leave
  null for full inference (submit_ranker).
"""
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference.phases import generate_phase, load_eval_users_from_csv
from src.utils import setup_logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="ranker", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info("gen_candidates config:\n%s", OmegaConf.to_yaml(cfg))

    users_source = cfg.get("users_source")
    if users_source is None:
        raise ValueError("gen_candidates: 'users_source' (path to CSV) must be set")
    output_dir = cfg.get("output_dir_phase")
    if output_dir is None:
        raise ValueError("gen_candidates: 'output_dir_phase' must be set")
    suffix = cfg.get("suffix", "")
    dislike_cutoff_ts = cfg.get("dislike_cutoff_ts", None)
    if dislike_cutoff_ts is not None:
        dislike_cutoff_ts = int(dislike_cutoff_ts)

    eval_users = load_eval_users_from_csv(users_source)
    log.info("gen_candidates: %d eval users from %s", len(eval_users), users_source)

    generate_phase(
        cg_cfg_list=cfg.candidate_generators,
        eval_users=eval_users,
        cache_dir=cfg.cg_cache_dir,
        size=cfg.data.size,
        suffix=suffix,
        output_dir=output_dir,
        dislikes_path=cfg.data.dislikes,
        undislikes_path=cfg.data.undislikes,
        dislike_cutoff_ts=dislike_cutoff_ts,
        filter_dislikes=bool(cfg.get("filter_dislikes", False)),
    )


if __name__ == "__main__":
    main()
