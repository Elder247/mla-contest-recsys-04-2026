"""Quick summary of a training run: ranker hyperparams, top features, CG list, submissions.

Usage:
    python scripts/inspect_run.py 008
    python scripts/inspect_run.py 008 --top 20
    python scripts/inspect_run.py 007 --artifacts-dir artifacts --submissions-dir submissions

Reads the artifacts that ``train_ranker.py`` / ``submit_ranker.py`` write
under ``artifacts/`` and ``submissions/``; nothing here triggers training
or data loading.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import polars as pl

from src.inference.validate_submission import validate_submission


def _cg_names_from_feature_cols(feat_cols: list[str]) -> list[str]:
    """A CG is detected whenever both ``{name}_score`` and ``{name}_rank`` exist."""
    score_names = {c[:-len("_score")] for c in feat_cols if c.endswith("_score")}
    rank_names = {c[:-len("_rank")] for c in feat_cols if c.endswith("_rank")}
    return sorted(score_names & rank_names)


def _format_kb(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def inspect_run(
    run_id: str,
    artifacts_dir: Path,
    submissions_dir: Path,
    top: int,
    users_csv: Path | None,
) -> int:
    rid = str(run_id)
    ranker_path = artifacts_dir / f"ranker_{rid}.pkl"
    fi_path = artifacts_dir / f"feature_importance_{rid}.csv"
    cg_dir = artifacts_dir / "cg"

    print(f"=== run_id={rid} ===")

    # ── Ranker pickle ────────────────────────────────────────────────────────
    if not ranker_path.exists():
        print(f"  [missing] {ranker_path}")
        return 1
    with open(ranker_path, "rb") as f:
        ranker = pickle.load(f)
    cb = ranker._model
    feats = ranker._feature_cols
    print(f"  ranker      : {ranker_path} ({_format_kb(ranker_path.stat().st_size)})")
    print(f"  hyperparams : iter={ranker.iterations} depth={ranker.depth} "
          f"lr={ranker.learning_rate} l2={ranker.l2_leaf_reg} "
          f"early_stop={ranker.early_stopping_rounds}")
    if cb is not None:
        try:
            best_iter = cb.get_best_iteration()
        except Exception:
            best_iter = "?"
        try:
            best_score = cb.get_best_score()
        except Exception:
            best_score = "?"
        print(f"  fitted      : best_iter={best_iter} best_score={best_score}")
    print(f"  features    : n={len(feats)}")

    cg_names = _cg_names_from_feature_cols(feats)
    print(f"  CGs (from feature cols, {len(cg_names)}): {', '.join(cg_names)}")

    # ── Feature importance ──────────────────────────────────────────────────
    if fi_path.exists():
        fi = pl.read_csv(fi_path)
        # CatBoost prettified columns: "Feature Id" + "Importances"
        if "Feature Id" in fi.columns and "Importances" in fi.columns:
            fi = fi.sort("Importances", descending=True).head(top)
            print(f"\n  top-{top} features (PredictionValuesChange):")
            width = max(len(str(c)) for c in fi["Feature Id"].to_list())
            for fid, imp in zip(fi["Feature Id"].to_list(), fi["Importances"].to_list()):
                print(f"    {str(fid).ljust(width)}  {imp:6.2f}%")
        else:
            print(f"\n  feature_importance: unexpected cols {fi.columns}")
    else:
        print(f"\n  feature_importance: [missing] {fi_path}")

    # ── CG cache files ──────────────────────────────────────────────────────
    if cg_dir.exists():
        cg_files = sorted(cg_dir.glob("*.pkl"))
        if cg_files:
            print(f"\n  CG cache ({len(cg_files)} files in {cg_dir}):")
            width = max(len(p.name) for p in cg_files)
            for p in cg_files:
                print(f"    {p.name.ljust(width)}  {_format_kb(p.stat().st_size)}")
    else:
        print(f"\n  CG cache: [missing dir] {cg_dir}")

    # ── Submissions matching this run_id ────────────────────────────────────
    sub_paths = sorted(submissions_dir.glob(f"sub_{rid}_*.csv"))
    if sub_paths:
        print(f"\n  submissions ({len(sub_paths)} matching sub_{rid}_*.csv):")
        for p in sub_paths:
            print(f"    {p.name}  {_format_kb(p.stat().st_size)}")
            if users_csv is not None and users_csv.exists():
                report = validate_submission(p, users_csv)
                status = "OK" if report.ok else "FAIL"
                print(f"      [{status}] rows={report.n_rows} "
                      f"items/row min={report.items_per_row_min} "
                      f"max={report.items_per_row_max} "
                      f"mean={report.items_per_row_mean:.1f} "
                      f"short={report.n_short_rows}")
                for w in report.warnings:
                    print(f"      warn: {w}")
                for e in report.errors:
                    print(f"      err : {e}")
    else:
        print(f"\n  submissions: none matching sub_{rid}_*.csv in {submissions_dir}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="run id (e.g. 008)")
    parser.add_argument("--artifacts-dir", default="artifacts", type=Path)
    parser.add_argument("--submissions-dir", default="submissions", type=Path)
    parser.add_argument("--users-csv", default="submissions/users.csv", type=Path)
    parser.add_argument("--top", type=int, default=15, help="top-N features to show")
    args = parser.parse_args()

    return inspect_run(
        run_id=args.run_id,
        artifacts_dir=args.artifacts_dir,
        submissions_dir=args.submissions_dir,
        top=args.top,
        users_csv=args.users_csv,
    )


if __name__ == "__main__":
    sys.exit(main())
