# RecSys Contest — Яндекс Музыка (Моя Волна)

## Цель и метрика
Внутренний контест Яндекса. Задача: **candidate generation** — предсказать до 100 треков на пользователя.
Метрика: `Recall@100 × 1000` (усредняется по пользователям, ×1000 для лидерборда).
Описание контеста: [docs/contest-description.md](docs/contest-description.md)

**Submission format** (`submissions/sub_XXX_name.csv`):
```
uid,item_ids
100,6 7 6767
200,3 14 15 926
```
- `uid` — из `submissions/users.csv` (= `range(100, 1_000_001, 100)`, 10 000 юзеров)
- `item_ids` — space-separated track IDs, не более 100

## Архитектура решения
Двухэтапная:
1. **Candidate generators** (~300-500 кандидатов/юзер): DecayPop, ALS, BPR, LightFM, SASRec, GRU4Rec, ContentKNN
2. **CatBoost Ranker** (reranking кандидатов → топ-100): CF scores + popularity + user/item features + audio similarity

Текущие скоры-ориентиры:
| Модель | Recall@100×1000 |
|--------|----------------|
| DecayPop (baseline) | ~50 |
| ALS | ~65 |
| SASRec | ~75 |
| CatBoost Ranker (ensemble) | ~90+ |

## Данные (Yambda)
Три размера: `50m` / `500m` / `5b`. **Всегда разрабатывать на `50m` первым**.

Схема файлов (HuggingFace `yandex/yambda`):
```
flat/{size}/listens.parquet    uid, item_id, timestamp, is_organic, played_ratio_pct, track_length_seconds
flat/{size}/likes.parquet      uid, item_id, timestamp, is_organic
flat/{size}/multi_event.parquet  все события объединены (поле event_type)
embeddings.parquet             item_id, embed, normalized_embed  (аудио CNN)
```

**Положительное прослушивание** = `played_ratio_pct > 50` (может быть > 100 при перемотке).
**Временные метки** — бинированы в 5-секундные юниты, отсортированы по `(uid, timestamp)`.

Локальная раскладка (в .gitignore): `data/{size}/` — отдельная папка на каждый размер.

Скачивание (проверяет наличие):
```bash
FILENAME="listens.parquet" SIZE="50m"
URL="https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/${SIZE}/${FILENAME}"
[ -f data/${SIZE}/${FILENAME} ] || wget -nv -P data/${SIZE}/ ${URL}
```

## Стек
- **Данные**: Polars (primary), PyArrow, Pandas
- **Модели**: implicit (ALS/BPR), lightfm-next, PyTorch, transformers, catboost
- **Config**: Hydra (`hydra-core==1.3.2`), конфиги под `configs/`
- **Eval**: pytest, Recall@k реализован в `src/evaluation/metrics.py`
- **Загрузка данных**: HuggingFace `datasets`

## Структура репо
```
configs/          Hydra конфиги (data/, model/, split/, features/, experiment/)
src/
  data/           dataset.py, preprocessing.py, splits.py, features.py
  models/         base.py, als.py, lightfm.py, sasrec.py, gru4rec.py, catboost_ranker.py, content_knn.py
  evaluation/     metrics.py  (Recall@k на Polars)
  inference/      merge_candidates.py
  training/       trainer.py
  utils/
scripts/          train.py, evaluate.py, make_submission.py  (все через @hydra.main)
notebooks/        00_baseline.ipynb, 01_eda.ipynb
docs/             contest-description.md, dataset-description.md, experiment-log.md
submissions/      sub_XXX_name.csv, users.csv
artifacts/        результаты экспериментов
data/             (gitignored) raw/{size}/, processed/, embeddings/, features/
```

## Команды
```bash
python scripts/train.py model=als dataset_size=50m           # обучение
python scripts/train.py +experiment=exp001_sasrec            # полный эксперимент
python scripts/evaluate.py model=als                         # только оценка
python scripts/make_submission.py model=als                  # генерация сабмита
pytest tests/ -v                                             # тесты
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
- Type hints на все функции
- Docstring только если логика нетривиальна (одна строка)
- `logging` вместо `print`

## Сплит для валидации
Темпоральный: train / val / test по времени.
- **val** имитирует публичный тест (~7 дней после cutoff Yambda)
- **test** имитирует приватный (сдвиг +1 день относительно публичного)
- **НЕ менять сплит между экспериментами** — иначе несравнимые результаты

## MUST
- Каждый скрипт — `@hydra.main` с конфигом из `configs/`
- Все гиперпараметры — в YAML, никогда хардкод
- Seed фиксируется в конфиге (`seed: 42`)
- Новая модель → новый YAML в `configs/model/`
- Новый эксперимент → запись в `docs/experiment-log.md` (val_score, public_score, config)
- Новый кандидат-генератор реализует интерфейс `BaseModel` из `src/models/base.py`

## NEVER
- НЕ трогать `data/raw/` (read-only)
- НЕ использовать `print()` — только `logging`
- НЕ коммитить данные и веса моделей
- НЕ менять валидационный сплит между экспериментами
- НЕ создавать entry-point скрипты без Hydra

## Внешние референсы
- https://github.com/antklen/recsys_challenge_2025
- NVIDIA 4-stage recommender blueprint (Merlin)
