# Yambda Dataset

Источник: [huggingface.co/datasets/yandex/yambda](https://huggingface.co/datasets/yandex/yambda) · Apache 2.0 · [Препринт arxiv:2505.22238](https://arxiv.org/abs/2505.22238)

Промышленный датасет Яндекс Музыки: прослушивания, лайки/дизлайки, аудио-эмбеддинги.
Три размера: **50M / 500M / 5B**. Разрабатывать всегда начинаем на `50m`.

## Статистика

| Датасет | Юзеры | Треки | Прослушивания | Лайки | Дизлайки |
|---------|-------|-------|--------------|-------|----------|
| Yambda-50M | 10 000 | 934 057 | 46.5M | 881K | 107K |
| Yambda-500M | 100 000 | 3.0M | 466.5M | 9.0M | 1.1M |
| Yambda-5B | 1 000 000 | 9.39M | 4.65B | 89.3M | 11.6M |

Аудио-эмбеддинги доступны для **7.72M** треков (общие для всех размеров).

## Ключевые концепции

- **Положительное прослушивание** = `played_ratio_pct > 50` (трек прослушан больше половины)
- `played_ratio_pct > 100` — юзер перемотал и переслушал, это норма, не ошибка
- **Временные метки** — глобальный порядок, бинированы в 5-секундные юниты (1 день = 17 280 единиц)
- **`is_organic = 1`** — органическое взаимодействие (сам нашёл); `0` — рекомендательное (из персонального фида/плейлиста)
- Все файлы отсортированы по `(uid, timestamp)`

## Локальная раскладка (gitignored)

```
data/
  50m/
    listens.parquet
    likes.parquet
    dislikes.parquet
    unlikes.parquet
    undislikes.parquet
    multi_event.parquet
  500m/   ← аналогично
  5b/     ← аналогично
  embeddings.parquet          ← общий для всех размеров
  album_item_mapping.parquet  ← общий
  artist_item_mapping.parquet ← общий
```

Подробная схема каждого файла: [data-dictionary.md](data-dictionary.md)

## Скачивание

```bash
# Отдельный файл (проверяет наличие)
FILENAME="listens.parquet" SIZE="50m"
URL="https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/${SIZE}/${FILENAME}"
[ -f data/${SIZE}/${FILENAME} ] || wget -nv -P data/${SIZE}/ ${URL}

# Через scripts/download_data.py (рекомендуется)
python scripts/download_data.py dataset_size=50m
```

Через HuggingFace API:
```python
from datasets import load_dataset
ds = load_dataset("yandex/yambda", data_dir="flat/50m", data_files="listens.parquet")
```

## FAQ

**Есть ли тестовые треки в трейне?**
Не все — часть тестовых треков встречается в трейне, часть нет.

**Есть ли cold users в тесте?**
Нет, все тестовые пользователи присутствуют в трейне.

**Как сгенерированы аудио-эмбеддинги?**
CNN по мотивам [Contrastive Learning of Musical Representations](https://arxiv.org/abs/2103.09410) (Spijkervet et al., 2021).

**Что такое `is_organic`?**
`1` — юзер нашёл трек сам (поиск, каталог); `0` — трек пришёл из персонального фида или плейлиста.

**Что считается прослушиванием (Listen+)?**
Трек прослушан > 50% его длины.

**Что значит `played_ratio_pct > 100`?**
Юзер перемотал и переслушал фрагмент, суммарное время прослушивания превысило длину трека. Ожидаемое поведение.
