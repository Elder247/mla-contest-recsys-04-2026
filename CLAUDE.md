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
Двухэтапная: **candidate generators** (~300-500/юзер) → **CatBoost Ranker** (top-100).
Генераторы: DecayPop, ALS, BPR, LightFM, SASRec, GRU4Rec, ContentKNN.
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
  data/           50m.yaml  500m.yaml
  model/          pop.yaml ✅  user_pop.yaml ✅  als.yaml ✅
  split/          temporal.yaml
  train.yaml      evaluate.yaml  submit.yaml  ranker.yaml ✅
src/
  data/           dataset.py ✅  splits.py ✅  preprocessing.py ✅  features.py (stub)
  models/         base.py ✅  pop.py ✅  als.py ✅  catboost_ranker.py ✅
                  lightfm.py (stub)  sasrec.py (stub)  gru4rec.py (stub)
  evaluation/     metrics.py ✅  (recall_at_k)
  inference/      merge_candidates.py (stub)
  training/       trainer.py (stub)
  utils/          logging.py ✅
scripts/          train.py ✅  evaluate.py ✅  make_submission.py ✅
                  train_ranker.py ✅  download_data.py ✅
notebooks/        00_baseline.ipynb  01_eda.ipynb
docs/             roadmap.md  dataset-description.md  data-dictionary.md  experiment-log.md
submissions/      sub_XXX_name.csv  users.csv
artifacts/        results.csv  (веса моделей — в .gitignore)
data/             (gitignored) 50m/  500m/  5b/  embeddings.parquet
```

## Команды
```bash
# Данные
python scripts/download_data.py dataset_size=50m

# Одиночная модель (кандидат-генератор)
python scripts/train.py model=als data=50m
python scripts/evaluate.py model=als data=50m
python scripts/make_submission.py model=als data=50m run_id=001

# Полный пайплайн ALS → CatBoost Ranker (основной)
python scripts/train_ranker.py data=50m run_id=002

# Тесты
pytest tests/ -v
```

## Запуск долгих команд (обучение)
Все долгие bash-команды (train, evaluate, make_submission) запускать в фоне с логом в файл:
```bash
python scripts/train_ranker.py data=50m 2>&1 | tee /tmp/train.log
```
Использовать `run_in_background=true` в Bash-инструменте.

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
- **Cold users**: ALS не знает юзеров без positive listens в train (~793/10k eval-юзеров). Они получают 0 предсказаний и тянут Recall вниз. Следующий шаг — добавить DecayPop fallback в `train_ranker.py` для cold users.

## Внешние референсы
- https://github.com/antklen/recsys_challenge_2025
- NVIDIA 4-stage recommender blueprint (Merlin)
