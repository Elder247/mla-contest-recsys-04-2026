"""Generate ``configs/ranker_optuna.yaml`` from Optuna best-params JSONs.

Reads:
    artifacts/optuna/cg_{name}_best.json   (Phase 1A — per-CG hyperparams)
    artifacts/optuna/n_cand_best.json      (Phase 1B — per-CG n_cand)
    artifacts/optuna/ranker_best.json      (Phase 1C — CatBoost params)

Writes:
    configs/ranker_optuna.yaml             (drop-in for --config-name=ranker_optuna)

The output is a copy of ``configs/ranker.yaml`` with hyperparam values from
the JSONs spliced in. Missing JSONs are tolerated — those slots keep their
ranker.yaml defaults so partial Optuna runs still produce a valid config.

Usage:
    python scripts/apply_optuna.py
    python scripts/apply_optuna.py base=configs/ranker.yaml \\
        out=configs/ranker_optuna.yaml \\
        optuna_dir=artifacts/optuna
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

from src.utils import setup_logging

log = logging.getLogger(__name__)


def _load_best(path: Path) -> dict | None:
    if not path.exists():
        log.warning("missing %s — skipping", path)
        return None
    with open(path) as f:
        return json.load(f)


def _apply_cg_best(cg_block, optuna_dir: Path) -> tuple[bool, bool]:
    """Mutate one CG entry in-place; returns (cg_params_applied, n_cand_applied)."""
    name = cg_block.get("name")
    if not name:
        return False, False

    cg_params_applied = False
    cg_best = _load_best(optuna_dir / f"cg_{name}_best.json")
    if cg_best and "best_params" in cg_best:
        for k, v in cg_best["best_params"].items():
            if k in cg_block:
                old = cg_block[k]
                cg_block[k] = v
                log.info("  %s.%s: %s → %s", name, k, old, v)
            else:
                # Param is in search space but not declared in ranker.yaml CG
                # block (e.g. defaults). Inject it.
                cg_block[k] = v
                log.info("  %s.%s: + %s (new)", name, k, v)
        cg_params_applied = True

    return cg_params_applied, False  # n_cand handled below from one shared file


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="configs/ranker.yaml")
    parser.add_argument("--out", default="configs/ranker_optuna.yaml")
    parser.add_argument("--optuna-dir", default="artifacts/optuna")
    args = parser.parse_args()

    base_path = Path(args.base)
    out_path = Path(args.out)
    optuna_dir = Path(args.optuna_dir)

    cfg = OmegaConf.load(base_path)
    log.info("loaded base config: %s", base_path)
    log.info("optuna dir: %s", optuna_dir)
    log.info("output: %s", out_path)

    # 1A — per-CG hyperparams
    log.info("--- Phase 1A: per-CG hyperparams ---")
    for cg_block in cfg.candidate_generators:
        _apply_cg_best(cg_block, optuna_dir)

    # 1B+1C — joint ranker + n_cand has priority over separate phases.
    # joint_best.json contains both ``n_cand_{name}`` and ranker keys
    # (iterations, depth, learning_rate, l2_leaf_reg). When present, use it
    # exclusively and skip the legacy separate-phase JSONs.
    joint_best = _load_best(optuna_dir / "joint_best.json")
    if joint_best and "best_params" in joint_best:
        log.info("--- Phase 1B+1C: joint ranker + n_cand (from joint_best.json) ---")
        params = joint_best["best_params"]

        # n_cand allocation
        for cg_block in cfg.candidate_generators:
            name = cg_block.get("name")
            key = f"n_cand_{name}"
            if key in params:
                v = int(params[key])
                if v <= 0:
                    log.warning(
                        "  %s.n_cand: optuna selected 0 — keeping baseline (%s)",
                        name, cg_block.get("n_cand"),
                    )
                    continue
                old = cg_block.get("n_cand")
                cg_block["n_cand"] = v
                log.info("  %s.n_cand: %s → %s", name, old, v)
        total = sum(int(cg.get("n_cand", 0)) for cg in cfg.candidate_generators)
        log.info("  total n_cand budget: %d", total)

        # Ranker hyperparams (only those that exist in cfg.ranker — joint
        # study omits early_stopping_rounds; keep ranker.yaml's value).
        for k, v in params.items():
            if k.startswith("n_cand_"):
                continue
            if k in cfg.ranker:
                old = cfg.ranker[k]
                cfg.ranker[k] = v
                log.info("  ranker.%s: %s → %s", k, old, v)
    else:
        # Legacy: separate n_cand_best.json + ranker_best.json
        log.info("--- Phase 1B: n_cand allocation ---")
        n_cand_best = _load_best(optuna_dir / "n_cand_best.json")
        if n_cand_best and "best_params" in n_cand_best:
            for cg_block in cfg.candidate_generators:
                name = cg_block.get("name")
                key = f"n_cand_{name}"
                if key in n_cand_best["best_params"]:
                    v = int(n_cand_best["best_params"][key])
                    if v <= 0:
                        log.warning(
                            "  %s.n_cand: optuna selected 0 — keeping baseline (%s)",
                            name, cg_block.get("n_cand"),
                        )
                        continue
                    old = cg_block.get("n_cand")
                    cg_block["n_cand"] = v
                    log.info("  %s.n_cand: %s → %s", name, old, v)
            total = sum(int(cg.get("n_cand", 0)) for cg in cfg.candidate_generators)
            log.info("  total n_cand budget: %d", total)

        log.info("--- Phase 1C: ranker hyperparams ---")
        ranker_best = _load_best(optuna_dir / "ranker_best.json")
        if ranker_best and "best_params" in ranker_best:
            for k, v in ranker_best["best_params"].items():
                if k in cfg.ranker:
                    old = cfg.ranker[k]
                    cfg.ranker[k] = v
                    log.info("  ranker.%s: %s → %s", k, old, v)

    # Write with a header comment so it's clear this file is generated
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_str = OmegaConf.to_yaml(cfg)
    header = (
        "# AUTO-GENERATED by scripts/apply_optuna.py.\n"
        "# Do not edit by hand — re-run apply_optuna.py to refresh.\n"
        f"# Base: {base_path}\n"
        f"# Optuna source: {optuna_dir}\n\n"
    )
    with open(out_path, "w") as f:
        f.write(header + yaml_str)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
