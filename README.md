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

Clone
```bash
git clone https://github.com/Elder247/mla-contest-recsys-04-2026 mla_contest && cd mla_contest
```

Python env (3.11 or 3.12)
```bash
python3.12 -m venv .venv && source .venv/bin/activate
export PYTHONPATH=$(pwd)
```

Install. CUDA torch first, then the rest:
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.6.0
pip install -r requirements.txt
```

Download 50m data to persistent disk
```bash
mkdir -p ~/dc-remote/{data,artifacts}
python3 -u scripts/download_data.py +dataset_size=50m data.root=/home/astrofimuk/dc-remote/data model=pop
```

Verify the install
```bash
python3 -m pytest tests/ -v
```

Baseline on 50m (CPU)
```bash
python -u scripts/train_ranker.py data=50m run_id=server_001 \
    data.root=/home/astrofimuk/dc-remote/data \
    artifacts_root=/home/astrofimuk/dc-remote/artifacts \
    cache_features=true 2>&1 | tee /tmp/run.log
```

Baseline on 50m (GPU)
```bash
python -u scripts/train_ranker.py data=50m run_id=server_001_gpu \
    data.root=/home/astrofimuk/dc-remote/data \
    artifacts_root=/home/astrofimuk/dc-remote/artifacts \
    ranker.task_type=GPU ranker.devices='0' \
    cache_features=true 2>&1 | tee /tmp/run_gpu.log
```

Build the submission CSV
```bash
python -u scripts/submit_ranker.py \
    data=50m run_id=server_001_gpu \
    data.root=/home/astrofimuk/dc-remote/data \
    artifacts_root=/home/astrofimuk/dc-remote/artifacts \
    submission_name=server_gpu_baseline
```

Validate the CSV format before uploading (optional)
```bash
python scripts/validate_submission.py submissions/sub_server_001_*.csv
```

Local development on a Mac follows the same flow with the defaults — no
need for the `data.root` / `artifacts_root` overrides; data lives in
`./data`, artifacts in `./artifacts`. CPU torch is fine; faiss-cpu is the
default on Apple Silicon.


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
