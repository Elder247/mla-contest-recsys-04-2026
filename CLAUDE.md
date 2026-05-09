# RecSys Contest — Яндекс Музыка (Моя Волна)

## Цель и метрика
Внутренний контест Яндекса. Задача: **candidate generation** — предсказать до 100 треков на пользователя.
Метрика: `Recall@100 × 1000`. Оцениваем только на юзерах из `submissions/users.csv` (10 000).

**Submission format** (`submissions/sub_XXX_name.csv`):
```
uid,item_ids
100,6 7 6767
```

## Архитектура решения (cascade)

```
┌─ 9 CGs (n_cand=800) ─────┐
│  decaypop, als, repeat,  │
│  itemknn, artist_pop,    │
│  album_pop, recent_likes,│
│  audio_knn, esasrec      │
└────────┬─────────────────┘
         ▼ outer-join (~25M строк, 9 {name}_rank/_score)
┌─ apply_n_cand_keep ──────┐  optuna: n_cand_keep_X ∈ [0, 800]
│  drops rows outside any  │  (CG=0 → не вносит уникальные строки,
│  CG's keep range         │   но score/rank колонки остаются)
└────────┬─────────────────┘
         ▼ ~14M строк
┌─ compute_cg_aggregates ──┐ → cg_count, cg_mean_score_norm
└────────┬─────────────────┘
         ▼
┌─ compute_features ───────┐ → user/item/pair/embed/cross (~85 cols)
│ chunked по uid (5000)    │
└────────┬─────────────────┘
         ▼
┌─ LightGBM 1-stage ───────┐  fixed lambdarank, fit ONCE
│  score → top-N per uid   │  train: n_ranker_train=1023 (fixed)
│                          │  eval:  n_ranker_eval ∈ [1000,2000] step 50 (optuna)
└────────┬─────────────────┘
         ▼ <= n_ranker_eval × n_users
┌─ CatBoost 2-stage ───────┐  YetiRank GPU, 1023-cap по RRF
│  score → top-100         │  optuna: iter/depth/lr/l2_leaf_reg/
│                          │          bagging_temperature/random_strength
└────────┬─────────────────┘
         ▼ submission CSV
```

Активные CG (9) в `configs/ranker.yaml`:

| CG | Файл | Назначение |
|----|------|------------|
| `decaypop` | `pop.py` (DecayPop) | global popular fallback (cold users) |
| `als` | `als.py` (3-tier weighted) | CF, low/mid/high engagement |
| `repeat` | `repeat.py` (RepeatListenModel) | re-listen из истории |
| `itemknn` | `itemknn.py` (CosineRecommender) | item-item KNN |
| `artist_pop` | `artist_pop.py` (entity=artist) | top tracks favorite artists |
| `album_pop` | `artist_pop.py` (entity=album) | top tracks favorite albums |
| `recent_likes` | `recent_likes.py` (likes-based) | likes × decay |
| `audio_knn` | `audio_knn.py` (FAISS HNSW32) | acoustic similarity |
| `esasrec` | `esasrec.py` (PyTorch SASRec) | sequential CF |

**Никогда не убирать esasrec из `cg_names_list`** — он в топ-10 importance.

### Two-stage ranker
- **LightGBM** (`src/models/lightgbm_ranker.py`) — lambdarank loss, CPU. Фиксированные гиперпараметры в коде с агрессивной регуляризацией (subsample + lambdarank_truncation_level=200). Обучается один раз с **negative subsampling 10:1** (≈10× быстрее, без потерь recall). Скоры кэшируются (`{run_id}_{train,eval}_lgbm.parquet`). Первая стадия фильтрации:
  - **`n_ranker_train=1023` (фиксировано)** — равно GPU YetiRank cap внутри `RankerModel.fit()`. Любое большее значение вырезается RRF-эвристикой при fit, что игнорирует LGBM-сигнал внизу пула.
  - **`n_ranker_eval` (тюнится в [1000, 2000])** — для inference важен recall headroom; больше кандидатов = больше шансов на positive в top-100 после CatBoost.
- Обе колонки **`lgbm_score` (Float32) + `lgbm_rank` (Int32)** идут как фичи в CatBoost — rank инвариантен к scale.
- **CatBoost** (`src/models/catboost_ranker.py`) — YetiRank GPU, 1023-cap внутри `fit()` через RRF. Финальный top-100. Поддерживает дополнительные knobs `bagging_temperature`, `random_strength` (None → CatBoost default).

### Optuna (joint_v4)
Поверхность поиска (~16 dim):
- 9 × `n_cand_keep_X` ∈ [0, 800] step 50.
- `n_ranker_eval` ∈ [1000, 2000] step 50. **`n_ranker_train` зафиксирован 1023** (== GPU YetiRank cap, см. ниже).
- CatBoost: `iterations` ∈ [1500, 4000] step 250, `depth` ∈ [4, 8], `learning_rate` ∈ [0.02, 0.15] log, `l2_leaf_reg` ∈ [1, 20] log, **`bagging_temperature` ∈ [0.0, 3.0]**, **`random_strength` ∈ [0.5, 5.0] log**.
- Objective: `0.5 × (recall_val + recall_test)`, multivariate TPE, `n_startup_trials=20`.
- LightGBM hyperparams **зафиксированы** в коде (см. `src/models/lightgbm_ranker.py` docstring) — `lambdarank_truncation_level=200`, `feature_fraction=0.8 / bagging_fraction=0.8 / freq=1`, lr=0.03, `n_estimators=3000`, `negative_ratio=10` (subsample при fit).

## Данные (Yambda)
Размеры: `50m` / `500m` / `5b`. **Разрабатывать на 50m, прогоны на 500m.**
- Положительное прослушивание = `played_ratio_pct > 50` (>100 при перемотке).
- Временные метки бинированы в 5-секундные юниты (1 день = 17 280).

## Стек
Polars 1.31, PyArrow, Pandas (только CatBoost.Pool), NumPy 2.2 / SciPy 1.15, implicit 0.7, catboost 1.2, torch 2.5, faiss-cpu 1.13, lightgbm (см. requirements.txt), optuna 4.3, hydra-core 1.3, pytest 8.3.

## Структура репо
```
configs/
  data/             50m.yaml  500m.yaml
  model/            pop|als|repeat|itemknn|artist_pop|album_pop|recent_likes|audio_knn|esasrec.yaml
  split/            temporal.yaml
  ranker.yaml             # 9 CGs + cascade pipeline (n_cand=800)
  ranker_v4_top{1..5}.yaml  # auto-generated by apply_optuna_top_k.py
  submit_ranker.yaml      # adds submission_dir/_name on top of ranker.yaml
  tune.yaml               # optuna phases: cg | ranker | n_cand | joint | joint_v2
src/
  data/             dataset.py, splits.py, preprocessing.py
                    features.py                 # ~85 фич (multi-window embed + cross-ratio)
  models/           base.py, pop.py, als.py, repeat.py, itemknn.py, artist_pop.py
                    recent_likes.py, audio_knn.py, esasrec.py
                    catboost_ranker.py          # 2-stage: score / top_k_per_user / predict
                    lightgbm_ranker.py          # 1-stage cascade filter
  evaluation/       metrics.py
  inference/        merge_candidates.py         # merge + apply_n_cand_keep + compute_cg_aggregates
                    pipeline.py
                    phases.py                   # fit / generate / features (chunked)
                    validate_submission.py
  training/         cg_cache.py, tune.py        # tune_ranker_and_n_cand_v2 + cascade variant
scripts/            train_ranker.py, submit_ranker.py, tune.py
                    train.py, evaluate.py, make_submission.py, fit_cg_full.py
                    apply_optuna_top_k.py       # writes ranker_*_top{N}.yaml
                    blend_submissions.py        # RRF blend CSVs
                    train_ranker_multiseed.py, submit_ranker_multiseed.py
artifacts/          results.csv, ranker_{run_id}.pkl, lgbm_{run_id}.pkl
                    feature_importance_{run_id}.csv
                    cg/{name}_{size}{,_full}.pkl
                    features/{run_id}_{train,eval,submit}.parquet
                    features/{run_id}_{train,eval}_lgbm.parquet  # cached LGBM scores
                    optuna/{study_name}.db
data/               (gitignored) 50m/  500m/  5b/  embeddings.parquet
```

## Команды (стандартный pipeline в порядке выполнения)

Допущения:
- `artifacts_root` = `/home/astrofimuk/dc-remote/artifacts` (см. `configs/ranker.yaml`). Под Linux/macOS sqlite-URI выходит вида `sqlite:////home/astrofimuk/dc-remote/artifacts/optuna/joint_v4.db` (4 слэша = абсолютный путь). Если у тебя другой путь — подставь его в `--storage`.
- Все нижеследующие `n_ranker_train=1023` и `n_ranker_eval=1500` — defaults из `configs/ranker.yaml`; явно прописывать НЕ нужно. `optuna joint_v4` тюнит только `n_ranker_eval`.

```bash
# 0) Tests (быстро, локально)
pytest tests/ -v

# 1) Прогрев features + LGBM (ОДИН раз; пишет _train/_eval/_lgbm parquets + lgbm_{run_id}.pkl)
#    Параметры из ranker.yaml: n_cand=800, n_ranker_train=1023, n_ranker_eval=1500.
#    Этот же run_id используется как base для Optuna joint_v2 ниже.
python -u scripts/train_ranker.py data=500m run_id=v4_features \
  feature_chunk_size=5000 enable_embed_features=true \
  2>&1 | tee /tmp/v4_features.log

# 2) Optuna joint (cascade на cached features + LGBM scores из шага 1)
#    n_ranker_eval тюнится в [1000, 2000] step 50; defaults из tune.yaml/ranker.yaml.
python -u scripts/tune.py phase=joint_v2 data=500m run_id=v4_features \
  n_max_per_cg=800 n_cand_min=0 \
  cg_names_list='[decaypop,als,repeat,itemknn,artist_pop,album_pop,recent_likes,audio_knn,esasrec]' \
  study_name=joint_v4 n_trials=80 2>&1 | tee /tmp/joint_v4.log

# 3) Top-K config gen — ВАЖНО: --pool-size 800 чтобы n_cand в сгенерённых yaml-ах
#    совпадал с тем, на чём тюнили (defaults: 500 → лучше явно).
python scripts/apply_optuna_top_k.py --study-name joint_v4 \
  --storage sqlite:////home/astrofimuk/dc-remote/artifacts/optuna/joint_v4.db \
  --base configs/ranker.yaml --out-prefix configs/ranker_v4_top --top-k 5 \
  --pool-size 800

# 4) Train + submit одного конфига (повторить для топ-5).
#    submission_name=v4_top1 → файл будет sub_v4_top1_v4_top1.csv (без — был бы sub_v4_top1_ranker.csv).
python -u scripts/train_ranker.py --config-name=ranker_v4_top1 \
  data=500m run_id=v4_top1 feature_chunk_size=5000 \
  2>&1 | tee /tmp/v4_top1_train.log
python -u scripts/submit_ranker.py --config-name=ranker_v4_top1 \
  data=500m run_id=v4_top1 submission_name=v4_top1 \
  2>&1 | tee /tmp/v4_top1_submit.log

# 5) Validate + inspect
python scripts/validate_submission.py submissions/sub_v4_top1_v4_top1.csv
python scripts/inspect_run.py v4_top1
```

### Multi-seed averaging (страховка для финального сабмита)
Готовая пара скриптов: [scripts/train_ranker_multiseed.py](scripts/train_ranker_multiseed.py) + [scripts/submit_ranker_multiseed.py](scripts/submit_ranker_multiseed.py). Они **переиспользуют** уже посчитанные features parquets, LGBM pickle и кэш LGBM-скоров от обычного `train_ranker.py` — новой feature-генерации **не происходит**. CatBoost обучается N раз с разными `random_state`, scores усредняются по seeds, top-100 берётся из mean-blend.

Стандартный сценарий — поверх лучшего optuna-trial'а (`ranker_v4_top1`). Шаги 0-3 ниже идут **после** train_ranker для `v4_top1` (т.е. шаг 4 в основном pipeline уже выполнен — есть `lgbm_v4_top1.pkl` + features-кэш).

```bash
# 1) Multi-seed train: тот же config-name, новый run_id, +base_run_id берёт кэши из v4_top1.
#    Cascade применяется внутри: train top-1023, eval top-n_ranker_eval (из ranker_v4_top1.yaml).
python -u scripts/train_ranker_multiseed.py --config-name=ranker_v4_top1 \
  data=500m run_id=v4_top1_ms \
  +base_run_id=v4_top1 \
  +seed_list=[42,43,44,45,46] \
  2>&1 | tee /tmp/v4_top1_ms_train.log

# 2) Submit с blend-усреднением (mean(ranker_score) по seeds, потом top-100 per uid).
#    submission_name перезаписывается на v4_top1_ms (был "ranker" из apply_optuna_top_k).
python -u scripts/submit_ranker_multiseed.py --config-name=ranker_v4_top1 \
  data=500m run_id=v4_top1_ms \
  +base_run_id=v4_top1 \
  +seed_list=[42,43,44,45,46] \
  submission_name=v4_top1_ms \
  2>&1 | tee /tmp/v4_top1_ms_submit.log

# 3) Validate
python scripts/validate_submission.py submissions/sub_v4_top1_ms_v4_top1_ms.csv
```

Что и куда пишется:
- `artifacts/ranker_v4_top1_ms_seed{S}.pkl` — N catboost'ов, по одному на seed.
- `artifacts/results.csv` — строки `model="ranker_multiseed_5seeds[...]"` для val/test blended recall.
- `submissions/sub_v4_top1_ms_v4_top1_ms.csv` — итоговый CSV.

Замечания:
- `seed_list` имеет смысл из 3-7 значений; больше — diminishing returns. 5 seeds — sweet spot.
- Каскад идентичный обычному пайплайну: `n_ranker_train=1023` для tying split, `n_ranker_eval` (берётся из `ranker_v4_top1.yaml` — т.е. оптимум из Optuna) для inference.
- LGBM **не переобучается** (один на `base_run_id=v4_top1`). Cascade применяется до per-seed CatBoost fit — все seeds видят одинаковый отфильтрованный пул.
- Ранее multi-seed давал +0.5-1.5 на public LB поверх top-1 single-seed. После добавления каскада ожидаемый эффект тот же.

## Запуск долгих команд
**Обязательно `python -u`** для unbuffered stdout (CatBoost progress в `tee /tmp/x.log` иначе появится только после завершения).

## Стиль кода
Polars — method chaining с переносом строки на каждую трансформацию. Type hints на все функции. `logging` вместо `print`. Никаких эмодзи в коде/PR.

LazyFrame для feature engineering (rules в `src/data/features.py` docstrings):
1. `pl.scan_parquet` (не read).
2. Predicate pushdown через `cutoff_ts`.
3. Pair-features: semi-join с candidates **до** group_by.
4. Materialization: `.collect(streaming=True)` или `.sink_parquet()`.
5. Type discipline: Int32/UInt32 счётчики, Float32 ratios, `.shrink_dtype()` в конце.

CatBoost Pool принимает только pandas/numpy → `.collect()` ровно один раз перед fit.

## Валидация
Темпоральный сплит: `temporal_split(df, val_days=7, gap_days=1)` в `src/data/splits.py`.
- val ≈ public test window. test ≈ private (с +1 day shift).
- Метрика только на юзерах из `submissions/users.csv`.
- **НЕ менять сплит между экспериментами**.

## MUST
- Каждый скрипт — `@hydra.main` с конфигом из `configs/`.
- Все гиперпараметры в YAML, никогда хардкод; `seed: 42`.
- Новый CG → YAML в `configs/model/` + класс реализует `BaseModel`.
- После каждого запуска → строка в `docs/experiment-log.md` (val/test/public).
- esasrec ВСЕГДА в `cg_names_list` для optuna. Без него — −5 на public.

## NEVER
- НЕ трогать `data/` напрямую — только через `src/data/dataset.py`.
- НЕ использовать `print()` — только `logging`.
- НЕ коммитить данные и веса моделей.
- НЕ менять валидационный сплит.
- НЕ добавлять rank-derived aggregates (`cg_min_rank`, `cg_rrf_score` etc.) — leak с 1023-cap (см. [merge_candidates.py](src/inference/merge_candidates.py) docstring).

## Кэш CG
- `artifacts/cg/{name}_{size}.pkl` (split.train fit, для train_ranker).
- `artifacts/cg/{name}_{size}_full.pkl` (full fit, для submit_ranker).
- При смене гиперпараметра CG → `force_refit_cg=true` или удалить пикл вручную (имя файла не содержит config hash).
- Audio_knn (FAISS) использует `__getstate__`/`__setstate__`: индекс не пиклится, перестраивается при load.

## Subprocess phases
`train_ranker.py` / `submit_ranker.py` запускают phases как subprocess (RSS reset). Они ОБЯЗАТЕЛЬНО передают `--config-name` через `_run_phase`. Без этого subprocess грузит базовый `ranker.yaml` и теряет `n_cand_keep`/etc.

## Memory hygiene на 500m
- `feature_chunk_size: 5000` — chunked features build, иначе 120 GB OOM. `build_embed_features` рано дропает eager dataframes. LightGBM на ~14M × 85 фич: ~10-15 GB peak, predict — chunked.
- LGBM `negative_ratio=10` (default) — fit-стадия использует ~1/10 размер тренировочного пула (только positives + 10× sample). На 500m это разница между ~3 минутами fit и ~30 минутами; для cache build / итераций тюнинга критично.

## Внешние референсы
https://github.com/antklen/recsys_challenge_2025 · NVIDIA 4-stage recommender blueprint (Merlin)
