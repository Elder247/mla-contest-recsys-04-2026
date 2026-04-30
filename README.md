# Yandex Music RecSys contest — Моя Волна

Two-stage candidate generation + CatBoost YetiRank reranker for Yandex Music's
"My Wave" recommendation contest. Predicts up to 100 tracks per user from a
10 000-user evaluation set; metric is `Recall@100 × 1000`.

Architecture, hygiene, and current CG bank are documented in
[CLAUDE.md](CLAUDE.md). Detailed roadmap and per-run history live in
[docs/roadmap.md](docs/roadmap.md) and [docs/experiment-log.md](docs/experiment-log.md).

---

## Quickstart (server, fresh clone)

Assumes Linux + NVIDIA GPU. Adjust the `data.root` / `artifacts_root` paths to
your persistent disk if applicable (e.g. `~/dc-remote/...`).

```bash
# 1. Clone
git clone <repo-url> mla_contest && cd mla_contest

# 2. Python env (3.11 or 3.12)
python3.12 -m venv .venv && source .venv/bin/activate

# 3. Install. CUDA torch first, then the rest:
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.6.0
pip install -r requirements.txt

# 4. Download data to persistent disk
mkdir -p ~/dc-remote/{data,artifacts}
python -u scripts/download_data.py dataset_size=500m data.root=~/dc-remote/data
# (50m / 5b sizes work the same — change dataset_size=)
# embeddings.parquet (~13 GB, shared across sizes) is downloaded automatically

# 5. Verify the install
pytest tests/ -v

# 6. First baseline on 500m
python -u scripts/train_ranker.py \
    data=500m run_id=server_001 \
    data.root=~/dc-remote/data \
    artifacts_root=~/dc-remote/artifacts \
    cache_features=true 2>&1 | tee /tmp/run.log

# 7. Build the submission CSV
python -u scripts/submit_ranker.py \
    data=500m run_id=server_001 \
    data.root=~/dc-remote/data \
    artifacts_root=~/dc-remote/artifacts \
    submission_name=server_baseline

# 8. (optional) Validate the CSV format before uploading
python scripts/validate_submission.py submissions/sub_server_001_*.csv
```

Local development on a Mac follows the same flow with the defaults — no
need for the `data.root` / `artifacts_root` overrides; data lives in
`./data`, artifacts in `./artifacts`. CPU torch is fine; faiss-cpu is the
default on Apple Silicon.

---

## Layout

```
configs/      Hydra configs (data, model, ranker, tune, submit_ranker)
src/          Library code: data loaders, models, features, evaluation,
              inference, training utilities
scripts/      Hydra entrypoints (train_ranker, submit_ranker, tune, ...)
              + plain CLIs (validate_submission, inspect_run)
tests/        pytest suite
docs/         Roadmap, dataset / data dictionary, experiment log
submissions/  Output CSVs (sub_*.csv) and the eval users list
artifacts/    CG / ranker pickles, feature_importance, results.csv,
              optuna sqlite — redirect via ``artifacts_root=...``
data/         Yambda parquets (gitignored) — redirect via ``data.root=...``
```

---

## Common commands

```bash
# Standalone CG (debug only — main entrypoint is train_ranker.py)
python -u scripts/train.py model=als data=50m
python -u scripts/evaluate.py model=als data=50m

# Multi-CG → ranker pipeline (main)
python -u scripts/train_ranker.py data=50m run_id=NNN
python -u scripts/submit_ranker.py data=50m run_id=NNN submission_name=NAME

# Force-refit all CGs (ignore pickle cache)
python -u scripts/train_ranker.py data=50m run_id=NNN force_refit_cg=true

# Optuna tuning (3 phases)
python -u scripts/tune.py phase=cg cg_name=als n_trials=50 n_max=500
python -u scripts/tune.py phase=ranker n_trials=30 run_id=NNN
python -u scripts/tune.py phase=n_cand n_trials=50 run_id=NNN total_budget=1500

# Diagnostics
python scripts/inspect_run.py NNN [--top 15]
python scripts/validate_submission.py submissions/sub_NNN_*.csv

# Tests
pytest tests/ -v
```

See [CLAUDE.md](CLAUDE.md) for code style, hygiene rules
(`temporal_split` invariants, CG cache semantics), and coding conventions
followed by both human contributors and AI agents.
