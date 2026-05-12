# RecSys — Моя Волна

Внутренний контест Яндекса: **candidate generation** — до 100 `item_id` на пользователя. Оценка только на 10k uid из `submissions/users.csv`. Метрика: `Recall@100 × 1000`. Формат сабмита: `uid,item_ids`.

Архитектура: 9 CG → merge → фичи → LightGBM (top-1023) → CatBoost YetiRank (top-100).
Подробнее — [CLAUDE.md](CLAUDE.md).

---

## Структура

| Путь | Назначение |
|------|------------|
| `configs/` | Hydra: `data/{50m,500m}.yaml`, `model/*.yaml`, `ranker.yaml`, `tune.yaml`, `submit_ranker.yaml`, оверлеи `ranker_v4_top{1..5}.yaml` из Optuna |
| `src/` | Данные, модели CG, фичи, merge, метрики, фазы пайплайна |
| `scripts/` | Точки входа: `train_ranker`, `submit_ranker`, `tune`, `refit_ranker`, multiseed, `blend_submissions`, `apply_optuna_top_k`, `train`/`evaluate` (один CG), `fit_cg_full`, `download_data`, `validate_submission`, `inspect_run` |
| `tests/` | pytest |
| `docs/` | `experiment-log.md`, прочие заметки |
| `submissions/` | `users.csv`, `sub_*.csv` |
| `data/` | Parquet датасета (gitignore) |
| `artifacts/` | CG/ranker pickle, `features/*.parquet`, `optuna/*.db`, `results.csv` (gitignore; путь задаётся `artifacts_root`) |

---

## Окружение

```bash
git clone https://github.com/Elder247/mla-contest-recsys-04-2026.git
cd mla-contest-recsys-04-2026

python3.10 -m venv .venv
source .venv/bin/activate
export PYTHONPATH=$(pwd)
```

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.0
pip install -r requirements.txt
```

**Данные** (Hugging Face `yandex/yambda`), `model=pop` нужен Hydra для конфига `train`:

```bash
python -u scripts/download_data.py data=50m model=pop data.root=/ABS/PATH/data
# 500m: data=500m
# 5b: data=5b
```

Опции для переопределения путей при запусках (если нужны не дефолтные значения из `configs/data/*.yaml` и `ranker.yaml`):

```bash
data.root=/ABS/PATH/data artifacts_root=/ABS/PATH/artifacts
```

---

## Тесты

```bash
pytest tests/ -v
```

---

## Один CG (train на сплите / full pickle для submit)

```bash
python -u scripts/train.py model=als data=50m
python -u scripts/evaluate.py model=als data=50m
# полный датасет → artifacts/cg/{name}_{size}_full.pkl (submit_ranker)
python -u scripts/fit_cg_full.py model=esasrec data=500m
```

---

## Финальный пайплайн (500m, joint_v4)

Имена `run_id`/`submission_name` — примеры

**1. Один раз:** фичи + LGBM-кэш скоров (в `ranker.yaml` проставить везде `n_cand` = `n_max_per_cg` из следующего шага, например, 800):

```bash
python -u scripts/train_ranker.py data=500m run_id=v4_features \
  feature_chunk_size=5000 enable_embed_features=true \
  2>&1 | tee /tmp/v4_features.log
```

**2. Optuna** (`phase=joint_v2`, имя стади `joint_v4`; 9 CG, включая `esasrec`; `n_max_per_cg` = `n_cand` при построении фич, здесь 800):

```bash
python -u scripts/tune.py phase=joint_v2 data=500m run_id=v4_features \
  n_max_per_cg=800 n_cand_min=0 \
  cg_names_list='[decaypop,als,repeat,itemknn,artist_pop,album_pop,recent_likes,audio_knn,esasrec]' \
  study_name=joint_v4 n_trials=80 \
  2>&1 | tee /tmp/joint_v4.log
```

**3. Создать YAML конфиги для Top-k запусков из optuna:

```bash
python scripts/apply_optuna_top_k.py --study-name joint_v4 \
  --storage sqlite:////ABS/artifacts/optuna/joint_v4.db \
  --base configs/ranker.yaml --out-prefix configs/ranker_v4_top --top-k 5 \
  --pool-size 800
```

**4. Обучение + сабмит**:

```bash
python -u scripts/train_ranker.py --config-name=ranker_v4_top1 \
  data=500m run_id=v4_top1 feature_chunk_size=5000 \
  2>&1 | tee /tmp/v4_top1_train.log

python -u scripts/submit_ranker.py --config-name=ranker_v4_top1 \
  data=500m run_id=v4_top1 submission_name=v4_top1 \
  2>&1 | tee /tmp/v4_top1_submit.log
```

**5. Проверка**

```bash
python scripts/validate_submission.py submissions/sub_v4_top1_v4_top1.csv
python scripts/inspect_run.py v4_top1
```

**6. Multiseed CatBoost** (те же фичи/LGBM, другой `run_id`, `+base_run_id` на кэш шага 4):

```bash
python -u scripts/train_ranker_multiseed.py --config-name=ranker_v4_top1 \
  data=500m run_id=v4_top1_ms +base_run_id=v4_top1 \
  +seed_list=[42,43,44,45,46] \
  2>&1 | tee /tmp/v4_top1_ms_train.log

python -u scripts/submit_ranker_multiseed.py --config-name=ranker_v4_top1 \
  data=500m run_id=v4_top1_ms +base_run_id=v4_top1 \
  +seed_list=[42,43,44,45,46] submission_name=v4_top1_ms \
  2>&1 | tee /tmp/v4_top1_ms_submit.log
```

**7. Переобучить только LGBM+CatBoost** без пересчёта фич: [scripts/refit_ranker.py](scripts/refit_ranker.py) — те же `data`, `run_id`, при оверлее `--config-name=ranker_v4_top1`.

```bash
python -u scripts/refit_ranker.py --config-name=ranker_v4_top1 \\
    data=500m run_id=v4_top1
```

**8. Весовой / RRF блендинг CSV**
Со стандартным rrf-k=60:
```bash
python -u scripts/blend_submissions.py \
  --inputs submissions/sub_v4_top1_ms_v4_top1_ms.csv:0.52 submissions/sub_v2_top1_v2_top1.csv:0.48 \
  --output submissions/sub_blend_ms_v2t1.csv
python scripts/validate_submission.py submissions/sub_blend_ms_v2t1.csv
```

rrf-k=45:
```bash
.venv/bin/python -u scripts/blend_submissions.py --rrf-k 45 \
  --inputs submissions/sub_v4_top1_ms_v4_top1_ms.csv:0.52 submissions/sub_v2_top1_v2_top1.csv:0.48 \
  --output submissions/sub_blend_ms_v2t1_k45.csv
```

---

## Остальные команды

```bash
python -u scripts/tune.py phase=cg cg_name=als n_trials=20 data=50m
python -u scripts/tune.py phase=ranker n_trials=25 run_id=RUN data=500m  # deprecated
python -u scripts/tune.py phase=n_cand n_trials=15 run_id=RUN data=500m  # deprecated
```

```bash
python -u scripts/train_ranker.py data=500m run_id=RUN force_refit_cg=true
```
