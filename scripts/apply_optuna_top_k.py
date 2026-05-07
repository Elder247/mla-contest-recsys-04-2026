"""Generate K ranker configs from the top-K trials of an Optuna study.

Counterpart to ``scripts/apply_optuna.py`` but keyed on `n_cand_keep`
instead of `n_cand`. Each output config has every CG running at a fixed
**pool size** (`n_cand`, default 500) so the merged candidate pool is
densely populated, plus a per-CG `n_cand_keep` row-filter equal to the
trial's `n_cand_{name}` value. This matches the keep_expr semantics that
joint_v2 Optuna trials trained on — see
:func:`src.inference.merge_candidates.apply_n_cand_keep`.

Outputs N yaml files named ``{out_prefix}{i}.yaml`` (i = 1..K), one per
trial, sorted by descending Optuna value. Each is a drop-in for
``--config-name=ranker_v2_top{i}`` on train_ranker.py / submit_ranker.py.

Usage:
    python scripts/apply_optuna_top_k.py \\
        --study-name joint_v2 \\
        --storage sqlite:///artifacts/optuna/joint_v2.db \\
        --base configs/ranker.yaml \\
        --out-prefix configs/ranker_v2_top \\
        --top-k 5 \\
        --pool-size 500

Pre-condition: ``--base`` must declare every CG named in the trial params
(``n_cand_{name}``). Comment-out CGs that aren't in the trial's search
space, or pass a base file that has them all uncommented.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import optuna
from omegaconf import OmegaConf

from src.utils import setup_logging

log = logging.getLogger(__name__)

_RANKER_KEYS = ("iterations", "depth", "learning_rate", "l2_leaf_reg")


def _load_top_k_trials(
    study_name: str, storage: str, top_k: int,
) -> list[optuna.trial.FrozenTrial]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name, storage=storage)
    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise RuntimeError(f"study '{study_name}' has no completed trials")
    complete.sort(key=lambda t: t.value, reverse=True)
    return complete[:top_k]


def _apply_trial_to_config(cfg, trial: optuna.trial.FrozenTrial, pool_size: int):
    """Mutate ``cfg`` in place: set CG ``n_cand``+``n_cand_keep`` and ranker hyperparams."""
    cg_names_in_cfg = {cg.get("name") for cg in cfg.candidate_generators}
    cg_names_in_trial = {
        k.removeprefix("n_cand_") for k in trial.params if k.startswith("n_cand_")
    }
    missing = cg_names_in_trial - cg_names_in_cfg
    if missing:
        raise ValueError(
            f"trial #{trial.number}: trial's CGs {sorted(missing)} are not "
            f"present in base config CGs {sorted(cg_names_in_cfg)}. "
            f"Uncomment the missing CG blocks in --base."
        )

    for cg in cfg.candidate_generators:
        name = cg.get("name")
        key = f"n_cand_{name}"
        n_keep = trial.params.get(key)
        if n_keep is None:
            log.warning(
                "  trial #%d has no '%s' — skipping CG '%s' (likely outside search space)",
                trial.number, key, name,
            )
            continue
        cg["n_cand"] = pool_size
        cg["n_cand_keep"] = int(n_keep)
        log.info("  %s: n_cand=%d, n_cand_keep=%d", name, pool_size, int(n_keep))

    for k in _RANKER_KEYS:
        if k in trial.params and k in cfg.ranker:
            old = cfg.ranker[k]
            cfg.ranker[k] = trial.params[k]
            log.info("  ranker.%s: %s → %s", k, old, trial.params[k])


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--storage", required=True,
                        help="optuna storage URI, e.g. sqlite:///path/to.db")
    parser.add_argument("--base", default="configs/ranker.yaml",
                        help="base ranker yaml; must declare all CGs named in the trials")
    parser.add_argument("--out-prefix", default="configs/ranker_v2_top",
                        help="output prefix; files will be {prefix}{i}.yaml for i=1..K")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=500,
                        help="fixed n_cand (pool size) for every CG in every output")
    args = parser.parse_args()

    base_path = Path(args.base)
    log.info("loading top-%d trials from %s @ %s", args.top_k, args.study_name, args.storage)
    trials = _load_top_k_trials(args.study_name, args.storage, args.top_k)
    log.info("got %d trials", len(trials))
    for t in trials:
        log.info("  #%d value=%.4f", t.number, t.value)

    written = []
    for i, trial in enumerate(trials, start=1):
        cfg = OmegaConf.load(base_path)
        log.info("--- trial #%d (rank %d, value=%.4f) ---", trial.number, i, trial.value)
        _apply_trial_to_config(cfg, trial, args.pool_size)

        out_path = Path(f"{args.out_prefix}{i}.yaml")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# AUTO-GENERATED by scripts/apply_optuna_top_k.py.\n"
            f"# Do not edit by hand — re-run apply_optuna_top_k.py to refresh.\n"
            f"# Base: {base_path}\n"
            f"# Study: {args.study_name} @ {args.storage}\n"
            f"# Trial: #{trial.number} (rank {i}/{args.top_k}), value={trial.value:.4f}\n"
            f"# Pool size: {args.pool_size}\n\n"
        )
        with open(out_path, "w") as f:
            f.write(header + OmegaConf.to_yaml(cfg))
        written.append(out_path)
        log.info("wrote %s", out_path)

    log.info("done — wrote %d configs:", len(written))
    for p in written:
        log.info("  %s", p)


if __name__ == "__main__":
    main()
