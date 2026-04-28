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
configs/          Hydra конфиги (data/, model/, split/, features/, experiment/)
src/
  data/           dataset.py ✅  splits.py ✅  preprocessing.py  features.py
  models/         base.py  als.py  lightfm.py  sasrec.py  gru4rec.py  catboost_ranker.py
  evaluation/     metrics.py ✅   (recall_at_k)
  inference/      merge_candidates.py
  training/       trainer.py
  utils/
scripts/          train.py  evaluate.py  make_submission.py  download_data.py
notebooks/        00_baseline.ipynb  01_eda.ipynb
docs/             roadmap.md  dataset-description.md  data-dictionary.md  experiment-log.md
submissions/      sub_XXX_name.csv  users.csv
artifacts/        results.csv  (веса моделей — в .gitignore)
data/             (gitignored) 50m/  500m/  5b/  embeddings.parquet
```

## Команды
```bash
python scripts/download_data.py dataset_size=50m
python scripts/train.py model=als data=50m
python scripts/evaluate.py model=als data=50m
python scripts/make_submission.py model=als data=50m
pytest tests/ -v
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
- Новая модель → новый YAML в `configs/model/` + реализует интерфейс `BaseModel`
- После каждой новой модели → запустить цикл train/evaluate/submit, записать в `docs/experiment-log.md`

## NEVER
- НЕ трогать `data/` напрямую — только через `src/data/dataset.py`
- НЕ использовать `print()` — только `logging`
- НЕ коммитить данные, веса моделей, `docs/roadmap.md`, `docs/TODO.md`
- НЕ менять валидационный сплит между экспериментами

## Внешние референсы
- https://github.com/antklen/recsys_challenge_2025
- NVIDIA 4-stage recommender blueprint (Merlin)
