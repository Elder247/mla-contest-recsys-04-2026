# RecSys Contest — Яндекс Музыка (Моя Волна)

## Цель и метрика
Внутренний контест Яндекса. Задача: **candidate generation** — предсказать до 100 треков на пользователя.
Метрика: `Recall@100 × 1000`. Оцениваем только на юзерах из `submissions/users.csv` (10 000).

**Submission format** (`submissions/sub_XXX_name.csv`):
```
uid,item_ids
100,6 7 6767
```

## Архитектура решения
Двухэтапная: **candidate generators** (~700-1000/юзер после dedup) → **CatBoost Ranker** (top-100).

Активные CG (8) в `configs/ranker.yaml`:

| CG | Файл | n_cand | Назначение |
|----|------|--------|------------|
| `decaypop` | `pop.py` (DecayPop) | 100 | global popular fallback (cold users) |
| `als` | `als.py` (3-tier weighted) | 500 | CF, low/mid/high engagement |
| `repeat` | `repeat.py` (RepeatListenModel) | 200 | re-listen из истории |
| `itemknn` | `itemknn.py` (CosineRecommender) | 200 | item-item KNN |
| `artist_pop` | `artist_pop.py` (entity=artist) | 100 | top tracks favorite artists |
| `album_pop` | `artist_pop.py` (entity=album) | 100 | top tracks favorite albums |
| `recent_likes` | `recent_likes.py` (data_source=likes) | 100 | likes × decay |
| `audio_knn` | `audio_knn.py` (FAISS HNSW32) | 100 | acoustic similarity, cold-cohort |

`eSASRec` — stub, реализуем на сервере (Phase C). Подробный план: [docs/roadmap.md](docs/roadmap.md), история прогонов: [docs/experiment-log.md](docs/experiment-log.md).

### RankerModel — split inference (важно для Optuna и 5B)
`src/models/catboost_ranker.py` экспортирует три точки входа:

- `score(df, chunk_size=500_000) → (uid, item_id, ranker_score)` — чанкует pandas-материализацию по N строк, сохраняет порядок входа. Память ≈ `chunk_size × n_features × 8B` независимо от размера. Используй для тюнинга n_cand allocation (Optuna): scoring делается ОДИН раз, top-k режется десятки раз с разными бюджетами без повторного предикта.
- `RankerModel.top_k_per_user(df, k, score_col="ranker_score")` — `@staticmethod`, чистая функция от scored DataFrame, режет per-user top-k.
- `predict(df, n=100, chunk_size=500_000)` — композиция `score → top_k_per_user`. Сигнатура и контракт прежние, train_ranker.py / submit_ranker.py не меняются.

Это рефакторинг D1+D2 из roadmap. **На 500m/5B и в Optuna trials используй `score()` напрямую** — иначе будет OOM или N-кратный пересчёт.

## Данные (Yambda)
Три размера: `50m` / `500m` / `5b`. **Разрабатывать на `50m` первым.**
Схемы всех файлов: [docs/data-dictionary.md](docs/data-dictionary.md)
Описание датасета: [docs/dataset-description.md](docs/dataset-description.md)

Скачивание: `python scripts/download_data.py dataset_size=50m`

Ключевые факты:
- Положительное прослушивание = `played_ratio_pct > 50` (может быть > 100 при перемотке)
- Временные метки бинированы в 5-секундные юниты (1 день = 17 280 единиц)
- Файлы отсортированы по `(uid, timestamp)`

## Стек
- **Данные**: Polars (primary), PyArrow, Pandas
- **Модели**: implicit (ALS/BPR), lightfm-next, PyTorch, transformers, catboost, optuna
- **ANN-индекс**: `faiss-cpu==1.13.2` (HNSW32 / IndexFlatIP) для AudioEmbedKNN
- **Config**: Hydra (`hydra-core==1.3.2`), конфиги под `configs/`
- **Тесты**: pytest

## Структура репо
```
configs/
  data/             50m.yaml  500m.yaml
  model/            pop.yaml ✅  user_pop.yaml ✅  als.yaml ✅  repeat.yaml ✅
                    itemknn.yaml ✅  artist_pop.yaml ✅  album_pop.yaml ✅
                    recent_likes.yaml ✅  audio_knn.yaml ✅
  split/            temporal.yaml
  train.yaml  evaluate.yaml  submit.yaml
  ranker.yaml ✅            # multi-CG + ranker pipeline (8 CG активных)
  submit_ranker.yaml ✅     # submission from saved ranker
src/
  data/             dataset.py ✅  splits.py ✅  preprocessing.py ✅
                    features.py ✅                       # LazyFrame FE, ~70 фич
  models/           base.py ✅  pop.py ✅  als.py ✅  repeat.py ✅
                    itemknn.py ✅  artist_pop.py ✅  recent_likes.py ✅
                    audio_knn.py ✅                      # FAISS HNSW32, cold cohort
                    catboost_ranker.py ✅                # score/top_k_per_user/predict
                    esasrec.py (stub → server)
  evaluation/       metrics.py ✅
  inference/        merge_candidates.py ✅  pipeline.py ✅
                    validate_submission.py ✅            # формат-чекер
  training/         cg_cache.py ✅
  utils/            logging.py ✅
scripts/            train.py ✅  evaluate.py ✅  make_submission.py ✅
                    train_ranker.py ✅      # train multi-CG + ranker, NO submission
                    submit_ranker.py ✅     # submission from saved ranker
                    inspect_run.py ✅       # сводка run: hyperparams, top features, CGs, subs
                    validate_submission.py ✅  # CLI вокруг src.inference.validate_submission
                    download_data.py ✅
notebooks/          00_baseline.ipynb  01_eda.ipynb
docs/               roadmap.md  dataset-description.md  data-dictionary.md  experiment-log.md
submissions/        sub_XXX_name.csv  users.csv
artifacts/          results.csv
                    cg/{name}_{size}.pkl              # fitted on split.train
                    cg/{name}_{size}_full.pkl         # fitted on full data
                    ranker_{run_id}.pkl
                    feature_importance_{run_id}.csv   # CatBoost native
                    features/{run_id}_{split}.parquet # optional cache
data/               (gitignored) 50m/  500m/  5b/  embeddings.parquet
```

## Команды
```bash
# Данные
python scripts/download_data.py dataset_size=50m

# Одиночная модель (кандидат-генератор) — для отладки
python scripts/train.py model=als data=50m
python scripts/evaluate.py model=als data=50m
python scripts/make_submission.py model=als data=50m run_id=001

# Multi-CG → CatBoost Ranker (основной): train сначала, submit потом
python scripts/train_ranker.py data=50m run_id=003
python scripts/submit_ranker.py data=50m run_id=003

# Принудительно переобучить все CG (игнорировать кэш)
python scripts/train_ranker.py data=50m run_id=003 force_refit_cg=true

# Сводка run'а: ranker hyperparams, best_iter, top features, CG-кэши, валидация сабмитов
python scripts/inspect_run.py 008 [--top 15]

# Валидация формата сабмита (10k uid, ≤100 item_ids/row, no dups)
python scripts/validate_submission.py submissions/sub_008_*.csv

# Тесты
pytest tests/ -v
```

## Запуск долгих команд (обучение)
Все долгие bash-команды (train, evaluate, make_submission) запускать в фоне с логом в файл и **обязательным флагом `-u`** для unbuffered stdout:
```bash
python -u scripts/train_ranker.py data=50m 2>&1 | tee /tmp/train.log
```
Использовать `run_in_background=true` в Bash-инструменте.

**Зачем `-u`**: CatBoost (и любой `print`/tqdm) пишет в stdout. При пайпе `| tee` Python автоматически переключает stdout на block-buffered (~8KB) → прогресс по итерациям и tqdm-бары появляются в логе только после завершения процесса. Флаг `-u` (или env `PYTHONUNBUFFERED=1`) отключает буферизацию, чтобы строки лились в `tee` сразу. Для `logging` это не нужно (оно идёт в stderr и unbuffered), но для CatBoost `verbose=N` — критично.

**Пользователю:** чтобы следить за прогрессом в реальном времени, открой терминал и выполни:
```bash
tail -f /tmp/train.log
```

## Стиль кода
Polars — method chaining с переносом строки на каждую трансформацию:
```python
return (
    df
    .filter(pl.col("played_ratio_pct") > 50)
    .group_by(["uid", "item_id"])
    .agg([pl.col("timestamp").max().alias("last_listen")])
    .with_columns([pl.col("last_listen").shrink_dtype()])
)
```
- Type hints на все функции; `logging` вместо `print`

### Lazy / streaming для feature engineering (масштаб 500m / 5B)
Все агрегаты пользователя/айтема/пары в `src/data/features.py` пишем как `pl.LazyFrame` chains. Правила:

1. **Загрузка**: `pl.scan_parquet(path)` (не `read_parquet`) — Polars пушит filter+projection в parquet reader, экономит RAM.
2. **Predicate pushdown через cutoff_ts**: фичестроители принимают `cutoff_ts: int`; внутри `lf.filter(pl.col("timestamp") <= cutoff_ts)`.
3. **Pair-features filter pushdown**: перед `group_by(["uid","item_id"])` делать `lf.join(candidates.select(["uid","item_id"]), on=..., how="semi")`. На 5B это разница между OK и OOM.
4. **Materialization**: `.collect(streaming=True)` или `.sink_parquet()` (для кэша) — никаких `.collect()` без streaming на больших данных.
5. **Type discipline**: `Int32` для счётчиков (или UInt32 на 5B), `Float32` для ratios, финальный `.shrink_dtype()` опц.
6. **Fold через left-join**: `add_features` orchestrator делает последовательные `.join(..., how="left")` — Polars optimizer фьюзит цепочку.

Не материализуем промежуточные DataFrame'ы. CatBoost Pool принимает только pandas/numpy → `.collect()` делаем один раз перед fit/predict.

## Валидация
Темпоральный сплит: `temporal_split(df, val_days=7, gap_days=1)` — реализован в `src/data/splits.py`.
- **val** имитирует публичный тест (7 дней после cutoff)
- **test** имитирует приватный (сдвиг +1 день, окна перекрываются — это ожидаемо)
- Метрика считается только на юзерах из `submissions/users.csv`
- **НЕ менять сплит между экспериментами**

## MUST
- Каждый скрипт — `@hydra.main` с конфигом из `configs/`
- Все гиперпараметры — в YAML, никогда хардкод; `seed: 42` в конфиге
- Новый кандидат-генератор → YAML в `configs/model/` + класс реализует `BaseModel`
- Новый pipeline-скрипт (несколько моделей) → отдельный YAML в `configs/` (напр. `configs/ranker.yaml`)
- После каждого запуска → записать val + test скор в `docs/experiment-log.md`

## NEVER
- НЕ трогать `data/` напрямую — только через `src/data/dataset.py`
- НЕ использовать `print()` — только `logging`
- НЕ коммитить данные и веса моделей (они в `.gitignore`)
- НЕ менять валидационный сплит между экспериментами

## Known limitations
- **Cold users**: ALS не знает юзеров без positive listens в train (~793/10k eval-юзеров). DecayPop в multi-CG пайплайне работает как fallback — он не требует истории.

## Кэш кандидат-генераторов
- Каждый CG, обученный через `train_ranker.py`, пиклится в `artifacts/cg/{name}_{size}.pkl`. При повторном запуске (например, ранкер с другими гиперпарамами) CG подгружается из кэша.
- `submit_ranker.py` использует параллельный кэш с суффиксом `_full` — модели обучены на полных данных без сплита.
- Принудительно переобучить: `force_refit_cg=true`.
- Кэш нельзя переиспользовать между разными размерами данных — это явно отражено в имени файла.
- **При изменении гиперпараметра CG (например, `half_life_units`) — ОБЯЗАТЕЛЬНО удалить соответствующий пикл** (`rm artifacts/cg/{name}_{size}{,_full}.pkl`) или запустить с `force_refit_cg=true`. Полей с hash'ом конфига в имени пикла нет — будет молча подгружен старый.
- **Для CG с тяжёлыми ANN-индексами** (audio_knn → FAISS) — не пиклить сам индекс. Реализуй `__getstate__` (выкидывает `_index = None`) и `__setstate__` (пересобирает из `_item_matrix` через `_build_index()`). Пример в `src/models/audio_knn.py`. Это избегает зависимости от faiss pickle compatibility и заметно ускоряет load.

## Внешние референсы
- https://github.com/antklen/recsys_challenge_2025
- NVIDIA 4-stage recommender blueprint (Merlin)
