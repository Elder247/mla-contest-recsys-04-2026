# Data Dictionary

Подробные схемы всех файлов датасета Yambda.
Общее описание: [dataset-description.md](dataset-description.md)

---

## listens.parquet

События прослушивания с деталями воспроизведения.

| Поле | Тип | Описание |
|------|-----|----------|
| `uid` | uint32 | ID пользователя |
| `item_id` | uint32 | ID трека |
| `timestamp` | uint32 | Временная метка (5-секундные юниты) |
| `is_organic` | uint8 | 1 = органическое, 0 = рекомендательное |
| `played_ratio_pct` | uint16 | % прослушанного трека; >100 означает перемотку |
| `track_length_seconds` | uint32 | Полная длина трека в секундах |

**Положительное прослушивание:** `played_ratio_pct > 50`

---

## likes.parquet / dislikes.parquet / unlikes.parquet / undislikes.parquet

Явные оценки пользователей. Все четыре файла имеют одинаковую схему.

| Поле | Тип | Описание |
|------|-----|----------|
| `uid` | uint32 | ID пользователя |
| `item_id` | uint32 | ID трека |
| `timestamp` | uint32 | Временная метка (5-секундные юниты) |
| `is_organic` | uint8 | 1 = органическое, 0 = рекомендательное |

- `unlikes` — отмена лайка
- `undislikes` — отмена дизлайка

---

## multi_event.parquet

Все события объединены в один файл. Удобен для последовательного моделирования.

| Поле | Тип | Описание |
|------|-----|----------|
| `uid` | uint32 | ID пользователя |
| `item_id` | uint32 | ID трека |
| `timestamp` | uint32 | Временная метка (5-секундные юниты) |
| `is_organic` | uint8 | 1 = органическое, 0 = рекомендательное |
| `event_type` | enum | `listen`, `like`, `dislike`, `unlike`, `undislike` |
| `played_ratio_pct` | Optional[uint16] | Только для `event_type = listen` |
| `track_length_seconds` | Optional[uint32] | Только для `event_type = listen` |

---

## embeddings.parquet

Аудио-эмбеддинги треков. Общий файл для всех размеров датасета (не лежит в `50m/`).

| Поле | Тип | Описание |
|------|-----|----------|
| `item_id` | uint32 | ID трека |
| `embed` | List[float] | Сырой аудио-эмбеддинг (CNN) |
| `normalized_embed` | List[float] | L2-нормализованный эмбеддинг |

Покрытие: **7.72M** треков из 9.39M в Yambda-5B.
Метод: CNN по мотивам Contrastive Learning of Musical Representations (Spijkervet et al., 2021).

---

## album_item_mapping.parquet

Соответствие альбом → треки.

| Поле | Тип | Описание |
|------|-----|----------|
| `album_id` | uint32 | ID альбома |
| `item_id` | uint32 | ID трека |

---

## artist_item_mapping.parquet

Соответствие исполнитель → треки.

| Поле | Тип | Описание |
|------|-----|----------|
| `artist_id` | uint32 | ID исполнителя |
| `item_id` | uint32 | ID трека |

---

## Sequential формат (альтернатива flat)

Доступен как `sequential/{size}/listens.parquet` и аналогично для других событий.
Данные агрегированы по пользователям — каждая строка это вся история юзера.

| Поле | Тип | Описание |
|------|-----|----------|
| `uid` | uint32 | ID пользователя |
| `item_ids` | List[uint32] | Хронологический список треков |
| `timestamps` | List[uint32] | Соответствующие временные метки |
| `is_organic` | List[uint8] | Флаги органичности |
| `played_ratio_pct` | List[Optional[uint16]] | Только в listens и multi_event |
| `track_length_seconds` | List[Optional[uint32]] | Только в listens и multi_event |

Все списки одинаковой длины и в хронологическом порядке.
Удобен для SASRec/GRU4Rec — не нужно делать group_by по uid.

---

## Общие замечания

- Все файлы отсортированы по `(uid, timestamp)` по возрастанию
- Временные метки — глобальный порядок, бинированы в 5-секундные юниты
- 1 день = 86400 сек / 5 = **17 280 единиц**
- uid и item_id — uint32, для Polars/implicit нужно кастить в Int64 при необходимости
