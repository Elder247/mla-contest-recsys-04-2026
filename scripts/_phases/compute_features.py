"""Phase 3 entrypoint — compute the feature parquet for a candidate set.

Direct invocation:

    python -u scripts/_phases/compute_features.py \\
        data=500m \\
        merged_path=artifacts/candidates/run_001/val/merged.parquet \\
        output_path=artifacts/features/run_001_train.parquet \\
        cutoff_ts=12345 \\
        [label_gt_path=artifacts/gt/gt_val.parquet]

CLI overrides:

- ``merged_path``  (required): per-phase candidate parquet from
  ``gen_candidates.py``.
- ``output_path``  (required): destination feature parquet.
- ``cutoff_ts``    (required): timestamp upper bound for feature aggregates.
- ``label_gt_path`` (optional): if set, append a ``label`` Int8 column.
"""
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference.phases import features_phase
from src.utils import setup_logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="ranker", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    log.info(
        "compute_features phase: data=%s run_id=%s cutoff_ts=%s enable_embed_features=%s merged_path=%s output_path=%s label_gt_path=%s",
        cfg.data.size, cfg.run_id, cfg.cutoff_ts, cfg.enable_embed_features,
        cfg.merged_path, cfg.output_path, cfg.label_gt_path,
    )


    merged_path = cfg.get("merged_path")
    output_path = cfg.get("output_path")
    cutoff_ts = cfg.get("cutoff_ts")
    if merged_path is None or output_path is None or cutoff_ts is None:
        raise ValueError(
            "compute_features: merged_path, output_path, cutoff_ts are all required"
        )
    cutoff_ts = int(cutoff_ts)
    label_gt_path = cfg.get("label_gt_path", None)

    enable_embed = bool(cfg.get("enable_embed_features", True))
    embeddings_path = cfg.data.embeddings if enable_embed else None

    chunk_size = cfg.get("feature_chunk_size", 0)
    chunk_size_uids = int(chunk_size) if chunk_size else None

    features_phase(
        merged_path=merged_path,
        listens_path=cfg.data.listens,
        likes_path=cfg.data.likes,
        dislikes_path=cfg.data.dislikes,
        unlikes_path=cfg.data.unlikes,
        undislikes_path=cfg.data.undislikes,
        artist_map_path=cfg.data.artist_item_mapping,
        album_map_path=cfg.data.album_item_mapping,
        cutoff_ts=cutoff_ts,
        output_path=output_path,
        embeddings_path=embeddings_path,
        label_gt_path=label_gt_path,
        chunk_size_uids=chunk_size_uids,
    )


if __name__ == "__main__":
    main()
