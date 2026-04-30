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
Двухэтапная: **candidate generators** (~600-800/юзер) → **CatBoost Ranker** (top-100).
Активные генераторы: DecayPop, ALS (weighted), RepeatListen, ItemKNN, ArtistAlbumPop, AudioEmbedKNN, eSASRec.
Подробный план и текущий прогресс: [docs/roadmap.md](docs/roadmap.md)

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
- **Config**: Hydra (`hydra-core==1.3.2`), конфиги под `configs/`
- **Тесты**: pytest

## Структура репо
```
configs/
  data/             50m.yaml  500m.yaml
  model/            pop.yaml ✅  user_pop.yaml ✅  als.yaml ✅  repeat.yaml ✅
                    itemknn.yaml ✅  artist_pop.yaml ✅  recent_likes.yaml ✅
                    audio_knn.yaml (A3)
  split/            temporal.yaml
  train.yaml  evaluate.yaml  submit.yaml
  ranker.yaml ✅            # multi-CG + ranker pipeline
  submit_ranker.yaml ✅     # submission from saved ranker
src/
  data/             dataset.py ✅  splits.py ✅  preprocessing.py ✅
                    features.py (stub → A2.2)
  models/           base.py ✅  pop.py ✅  als.py ✅  repeat.py ✅
                    itemknn.py ✅  artist_pop.py ✅  recent_likes.py ✅
                    catboost_ranker.py ✅  esasrec.py (stub → server)
  evaluation/       metrics.py ✅
  inference/        merge_candidates.py ✅  pipeline.py ✅
  training/         cg_cache.py ✅
  utils/            logging.py ✅
scripts/            train.py ✅  evaluate.py ✅  make_submission.py ✅
                    train_ranker.py ✅      # train multi-CG + ranker, NO submission
                    submit_ranker.py ✅     # submission from saved ranker
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

## Внешние референсы
- https://github.com/antklen/recsys_challenge_2025
- NVIDIA 4-stage recommender blueprint (Merlin)
