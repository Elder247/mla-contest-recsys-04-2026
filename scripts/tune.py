"""Optuna entrypoint — three phases of tuning (Phase D / B4).

Usage:

    # Phase 1: standalone CG tuning (Recall@N_max coverage metric)
    python -u scripts/tune.py phase=cg cg_name=decaypop n_trials=30 run_id=008

    # Phase 2: ranker tuning on cached features
    #   pre-req: train_ranker.py was run with cache_features=true
    python -u scripts/tune.py phase=ranker n_trials=30 run_id=008

    # Phase 3: n_cand allocation under budget
    #   pre-req: same as ranker phase + ranker pickle exists
    python -u scripts/tune.py phase=n_cand n_trials=50 run_id=008 \
        total_budget=1500 budget_step=25

Studies persist to ``artifacts/optuna/{study_name}.db`` (SQLite). Resume
by re-invoking with the same ``study_name`` — Optuna will pick up where it
left off.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import hydra
import optuna
import polars as pl
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import load_likes, load_listens, positive_listens
from src.data.splits import temporal_split
from src.evaluation.metrics import recall_at_k
from src.inference.pipeline import load_eval_users
from src.models.als import ALSModel
from src.models.artist_pop import ArtistAlbumPopModel
from src.models.audio_knn import AudioEmbedKNNModel
from src.models.itemknn import ItemKNNModel
from src.models.pop import DecayPop
from src.models.recent_likes import RecentLikesModel
from src.models.repeat import RepeatListenModel
from src.training.tune import (
    default_als_space,
    default_artist_album_pop_space,
    default_audio_knn_space,
    default_decaypop_space,
    default_itemknn_space,
    default_recent_likes_space,
    default_repeat_space,
    make_storage,
    tune_candidate_generator,
    tune_n_cand,
    tune_ranker,
)
from src.utils import setup_logging

log = logging.getLogger(__name__)


# ── CG factory registry ─────────────────────────────────────────────────────
# Each entry maps a ``cg_name`` to (model_cls, default_space, fixed_kwargs).
# fixed_kwargs are merged with sampled params and passed to ``model_cls(...)``.

def _cg_registry(n_max: int) -> dict[str, tuple]:
    return {
        "decaypop": (
            DecayPop, default_decaypop_space,
            dict(name="decaypop", n_cand=n_max),
        ),
        "repeat": (
            RepeatListenModel, default_repeat_space,
            dict(name="repeat", n_cand=n_max),
        ),
        "recent_likes": (
            RecentLikesModel, default_recent_likes_space,
            dict(name="recent_likes", n_cand=n_max),
        ),
        "als": (
            ALSModel, default_als_space,
            dict(name="als", n_cand=n_max, random_state=42),
        ),
        "itemknn": (
            ItemKNNModel, default_itemknn_space,
            dict(name="itemknn", n_cand=n_max),
        ),
        "artist_pop": (
            ArtistAlbumPopModel, default_artist_album_pop_space,
            dict(name="artist_pop", entity="artist", n_cand=n_max),
        ),
        "album_pop": (
            ArtistAlbumPopModel, default_artist_album_pop_space,
            dict(name="album_pop", entity="album", n_cand=n_max),
        ),
        "audio_knn": (
            AudioEmbedKNNModel, default_audio_knn_space,
            dict(name="audio_knn", n_cand=n_max),
        ),
    }


def _build_cg_factory(cg_name: str, n_max: int):
    registry = _cg_registry(n_max)
    if cg_name not in registry:
        raise ValueError(
            f"Unknown cg_name='{cg_name}'. Supported: {sorted(registry)}"
        )
    model_cls, space_fn, fixed_kwargs = registry[cg_name]

    def factory(trial: optuna.Trial):
        sampled = space_fn(trial)
        return model_cls(**fixed_kwargs, **sampled)

    return factory


# ── Baseline param extraction ──────────────────────────────────────────────
# Search-space keys per CG (must match ``trial.suggest_*`` names in
# ``src/training/tune.py:default_*_space``). Used to filter the ranker.yaml
# CG block down to just the params Optuna is sampling.
_CG_SPACE_KEYS = {
    "decaypop": ["half_life_units"],
    "repeat": ["half_life_units"],
    "recent_likes": ["half_life_units"],
    "artist_pop": ["top_entities", "half_life_units"],
    "album_pop": ["top_entities", "half_life_units"],
    "als": [
        "factors", "iterations", "regularization", "alpha",
        "low_engagement_weight", "high_engagement_weight",
    ],
    "itemknn": ["k"],
    "audio_knn": ["user_history_k", "hnsw_m", "ef_construction", "ef_search"],
}

_RANKER_SPACE_KEYS = [
    "iterations", "depth", "learning_rate", "l2_leaf_reg", "early_stopping_rounds",
]


def _load_ranker_yaml(path: str = "configs/ranker.yaml") -> DictConfig:
    return OmegaConf.load(path)


def _baseline_for_cg(cg_name: str, ranker_yaml_path: str = "configs/ranker.yaml") -> dict:
    """Pull the current ranker.yaml hyperparams matching this CG's search space."""
    keys = _CG_SPACE_KEYS.get(cg_name)
    if keys is None:
        return {}
    cfg = _load_ranker_yaml(ranker_yaml_path)
    for cg_cfg in cfg.candidate_generators:
        if cg_cfg.get("name") == cg_name:
            return {k: cg_cfg[k] for k in keys if k in cg_cfg}
    return {}


def _baseline_for_ranker(ranker_yaml_path: str = "configs/ranker.yaml") -> dict:
    cfg = _load_ranker_yaml(ranker_yaml_path)
    return {k: cfg.ranker[k] for k in _RANKER_SPACE_KEYS if k in cfg.ranker}


def _baseline_for_n_cand(
    cg_names: list[str],
    ranker_yaml_path: str = "configs/ranker.yaml",
) -> dict:
    """Match ``trial.suggest_int`` names in tune_n_cand: ``n_cand_{name}``."""
    cfg = _load_ranker_yaml(ranker_yaml_path)
    name_to_n = {cg.get("name"): cg.get("n_cand") for cg in cfg.candidate_generators}
    return {f"n_cand_{n}": int(name_to_n[n]) for n in cg_names if n in name_to_n}


def _save_best_json(
    study: optuna.Study,
    output_path: Path,
    extra: dict | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_name": study.study_name,
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        **(extra or {}),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("saved best params → %s", output_path)


def _resolve_n_trials(cfg: DictConfig) -> int:
    """Resolve n_trials: explicit ``cfg.n_trials`` wins, else per-CG/per-phase default.

    Default lookup uses ``n_trials_per_cg`` (keyed by cg_name) for phase=cg
    and ``n_trials_per_phase`` (keyed by 'ranker' / 'n_cand') otherwise.
    Set ``n_trials: null`` in tune.yaml so this works as expected; pass
    ``n_trials=N`` on the CLI to force a smoke-test count.
    """
    if cfg.get("n_trials") is not None:
        return int(cfg.n_trials)
    if cfg.phase == "cg":
        per_cg = cfg.get("n_trials_per_cg") or {}
        if cfg.cg_name in per_cg:
            return int(per_cg[cfg.cg_name])
    else:
        per_phase = cfg.get("n_trials_per_phase") or {}
        if cfg.phase in per_phase:
            return int(per_phase[cfg.phase])
    raise ValueError(
        f"could not resolve n_trials: phase={cfg.phase} cg_name={cfg.get('cg_name')}"
    )


# ── Phase implementations ──────────────────────────────────────────────────


def _ground_truth(df: pl.DataFrame, users: list[int]) -> pl.DataFrame:
    return (
        df.select(["uid", "item_id"])
        .with_columns([pl.col("uid").cast(pl.Int64), pl.col("item_id").cast(pl.Int64)])
        .filter(pl.col("uid").is_in(users))
        .unique()
    )


def _phase_cg(cfg: DictConfig, storage: str) -> optuna.Study:
    log.info("phase=cg | cg_name=%s | n_max=%d | data_source=%s",
             cfg.cg_name, cfg.n_max, cfg.data_source)

    listens = positive_listens(load_listens(path=cfg.data.listens))
    split = temporal_split(
        listens,
        val_days=cfg.split.val_days,
        gap_days=cfg.split.gap_days,
        timestamp_col=cfg.split.timestamp_col,
    )
    log.info("train=%d  val=%d  test=%d", len(split.train), len(split.val), len(split.test))

    eval_users = load_eval_users(cfg.data.users_csv)
    gt_val = _ground_truth(split.val, eval_users)
    val_users_with_gt = gt_val["uid"].unique().to_list()
    log.info("gt_val: %d pairs / %d users", len(gt_val), len(val_users_with_gt))

    if cfg.data_source == "listens":
        train_data = split.train
    elif cfg.data_source == "likes":
        train_max_ts = float(split.train["timestamp"].max())
        train_data = (
            load_likes(path=cfg.data.likes)
            .filter(pl.col("timestamp") <= train_max_ts)
        )
    else:
        raise ValueError(f"data_source must be 'listens' or 'likes', got {cfg.data_source}")
    log.info("CG train data: %d rows (source=%s)", len(train_data), cfg.data_source)

    factory = _build_cg_factory(cfg.cg_name, cfg.n_max)
    baseline = _baseline_for_cg(cfg.cg_name)
    n_trials = _resolve_n_trials(cfg)

    return tune_candidate_generator(
        model_factory=factory,
        train=train_data,
        eval_users=val_users_with_gt,
        gt_val=gt_val,
        n_max=cfg.n_max,
        n_trials=n_trials,
        study_name=cfg.study_name,
        storage=storage,
        seed=cfg.seed,
        baseline_params=baseline,
    )


def _load_gt_val(cfg: DictConfig) -> pl.DataFrame:
    listens = positive_listens(load_listens(path=cfg.data.listens))
    split = temporal_split(
        listens,
        val_days=cfg.split.val_days,
        gap_days=cfg.split.gap_days,
        timestamp_col=cfg.split.timestamp_col,
    )
    eval_users = load_eval_users(cfg.data.users_csv)
    return _ground_truth(split.val, eval_users)


def _phase_ranker(cfg: DictConfig, storage: str) -> optuna.Study:
    labeled_path = Path(cfg.labeled_features_path)
    eval_path = Path(cfg.eval_features_path)
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"labeled features not cached: {labeled_path}\n"
            f"Run: python -u scripts/train_ranker.py data={cfg.data.size} "
            f"run_id={cfg.run_id} cache_features=true"
        )
    if not eval_path.exists():
        raise FileNotFoundError(
            f"eval features not cached: {eval_path}\n"
            f"Run train_ranker.py with cache_features=true (same run_id)"
        )

    log.info("loading cached features: labeled=%s eval=%s", labeled_path, eval_path)
    labeled_df = pl.read_parquet(labeled_path)
    eval_features_df = pl.read_parquet(eval_path)
    log.info(
        "labeled=%d×%d  eval=%d×%d",
        len(labeled_df), len(labeled_df.columns),
        len(eval_features_df), len(eval_features_df.columns),
    )

    gt_val = _load_gt_val(cfg)
    log.info("gt_val: %d pairs / %d users", len(gt_val), gt_val["uid"].n_unique())

    return tune_ranker(
        labeled_df=labeled_df,
        eval_features_df=eval_features_df,
        gt_val=gt_val,
        n_trials=_resolve_n_trials(cfg),
        k=cfg.top_k,
        study_name=cfg.study_name,
        storage=storage,
        seed=cfg.seed,
        baseline_params=_baseline_for_ranker(),
    )


def _phase_n_cand(cfg: DictConfig, storage: str) -> optuna.Study:
    eval_path = Path(cfg.eval_features_path)
    ranker_path = Path(cfg.ranker_path)
    if not eval_path.exists():
        raise FileNotFoundError(
            f"eval features not cached: {eval_path}\n"
            f"Run train_ranker.py with cache_features=true (same run_id)"
        )
    if not ranker_path.exists():
        raise FileNotFoundError(f"ranker pickle not found: {ranker_path}")

    log.info("loading eval features: %s", eval_path)
    eval_features_df = pl.read_parquet(eval_path)
    log.info("loading ranker: %s", ranker_path)
    with open(ranker_path, "rb") as f:
        ranker = pickle.load(f)

    cg_names = list(cfg.cg_names_list)
    missing_ranks = [
        f"{n}_rank" for n in cg_names if f"{n}_rank" not in eval_features_df.columns
    ]
    if missing_ranks:
        raise ValueError(
            f"eval features missing rank columns for {missing_ranks}; "
            f"either drop those CGs from cg_names_list or rebuild eval features."
        )

    log.info("scoring eval features once (chunked)")
    scored_uid_item = ranker.score(eval_features_df)
    # Re-attach the {name}_rank columns from eval_features_df via row-wise concat
    # — score() preserves order, so a horizontal concat is safe.
    rank_cols = [f"{n}_rank" for n in cg_names]
    scored_df = pl.concat(
        [scored_uid_item, eval_features_df.select(rank_cols)],
        how="horizontal",
    )
    log.info("scored merged candidates: %d rows × %d cols",
             len(scored_df), len(scored_df.columns))

    gt_val = _load_gt_val(cfg)

    # Clamp baseline to the search ceiling so enqueue_trial doesn't reject
    # values outside the suggest_int range (e.g. als baseline 350 vs n_max 350).
    baseline = {
        k: min(int(v), int(cfg.n_max_per_cg))
        for k, v in _baseline_for_n_cand(cg_names).items()
    }

    return tune_n_cand(
        scored_df=scored_df,
        gt_val=gt_val,
        cg_names=cg_names,
        n_max_per_cg=cfg.n_max_per_cg,
        total_budget=cfg.total_budget,
        n_trials=_resolve_n_trials(cfg),
        k=cfg.top_k,
        step=cfg.budget_step,
        study_name=cfg.study_name,
        storage=storage,
        seed=cfg.seed,
        baseline_params=baseline,
    )


# ── Main ────────────────────────────────────────────────────────────────────


@hydra.main(config_path="../configs", config_name="tune", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    # Auto-expand the default ``study_name=${phase}`` to ``cg_${cg_name}``
    # when running per-CG tuning, so each of the 8 CG runs gets its own
    # SQLite study without forcing the user to pass it on the CLI.
    if cfg.phase == "cg" and cfg.study_name == "cg":
        cfg.study_name = f"cg_{cfg.cg_name}"
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    storage = make_storage(Path(cfg.storage_dir) / f"{cfg.study_name}.db")
    log.info("optuna storage: %s", storage)

    if cfg.phase == "cg":
        study = _phase_cg(cfg, storage)
    elif cfg.phase == "ranker":
        study = _phase_ranker(cfg, storage)
    elif cfg.phase == "n_cand":
        study = _phase_n_cand(cfg, storage)
    else:
        raise ValueError(f"phase must be 'cg' | 'ranker' | 'n_cand', got '{cfg.phase}'")

    log.info("=" * 70)
    log.info("study '%s' done: %d trials", cfg.study_name, len(study.trials))
    log.info("best value (Recall@%s x1000): %.4f", cfg.top_k if cfg.phase != "cg" else cfg.n_max, study.best_value)
    log.info("best params:")
    for k, v in study.best_params.items():
        log.info("  %s = %s", k, v)
    log.info("storage: %s", storage)

    # Persist best params as JSON so apply_optuna.py / experiment-log can
    # consume them without opening the SQLite study.
    extra = {"phase": cfg.phase}
    if cfg.phase == "cg":
        extra["cg_name"] = cfg.cg_name
        extra["n_max"] = int(cfg.n_max)
    elif cfg.phase == "n_cand":
        extra["total_budget"] = int(cfg.total_budget)
        extra["n_max_per_cg"] = int(cfg.n_max_per_cg)
    _save_best_json(
        study,
        Path(cfg.storage_dir) / f"{cfg.study_name}_best.json",
        extra=extra,
    )


if __name__ == "__main__":
    main()
