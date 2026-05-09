# Final Push — план для свежего агента (контекст будет сброшен)

## Где что выполняется

- **Агент работает ЛОКАЛЬНО** на macOS (Mac). Здесь: пишет код, запускает `pytest tests/`, smoke-тесты на 50m данных, делает `git commit`/`push`.
- **Пользователь запускает основные вычисления НА СЕРВЕРЕ** (Linux, 1× A100, 120 GB RAM). Сервер делает: `git pull`, фичегенерация на 500m, optuna (~80 trials), train+submit топ-5. Агент **не имеет доступа к серверу** напрямую.
- Команды в Section 3 — это то, **что агент даёт пользователю** для запуска на сервере (не пытается выполнить сам).
- Артефакты сервера (CG pickles, optuna DB, feature parquets) лежат в `~/dc-remote/artifacts/` на сервере; локально доступна только optuna SQLite-база и feature_importance CSV (через git).

## Context

Best public **319.01** (sub_v2_top5). Lead **~330**. Зазор **+11**. Бюджет: ~2 дня, 1× A100 server. Лучший локальный optuna-best был v2_top28 (combined val+test=405.18, public=318.03), top5 trial val хуже но public лучше → val→public gap нестабилен.

Последний прогон **joint_v3** (n_max=800) дал val=397.97 / public=314.64 — хуже v2 (319.01). **Причина**: в `cg_names_list` не было eSASRec (баг в команде). esasrec в feature_importance v2_top1 даёт 3.82+0.83 = 4.6% — заметный сигнал.

Текущее состояние кода (последние коммиты):
- `compute_cg_aggregates` → `cg_count`, `cg_mean_score_norm` (rank-based aggregates **удалены** из-за sample-selection leak с GPU 1023-cap; см. docstring в [merge_candidates.py](src/inference/merge_candidates.py#L196-L213)).
- multi-window embed cosines: `embed_cos_user_last_5/20/50/100` вместо одной `_last_k`.
- chunked `features_phase` (`feature_chunk_size: 5000`) — больше нет OOM на 500m.
- `apply_n_cand_keep` post-merge filter ([merge_candidates.py:103](src/inference/merge_candidates.py#L103)).
- `apply_optuna_top_k.py` — генерит N конфигов из optuna study.
- `train_ranker_multiseed.py` + `submit_ranker_multiseed.py` (готовы, не запускались).

Что ломалось ранее и почему важно знать:
- subprocess phase-скрипты теряли `--config-name` → загружали базовый `ranker.yaml` без `n_cand_keep`. Фикс: parent передаёт `--config-name` через `_run_phase`. Уже в коде.
- `submit_ranker.py` ругается на `submission_dir`/`submission_name` если конфиг авто-сгенерён без них (они только в `submit_ranker.yaml`). Workaround: `+submission_dir=submissions +submission_name=v4_top$i` на CLI.

Цель этого плана: внедрить **LightGBM cascade + новые фичи**, прогнать новую optuna и top-5 → public блоки, получить **322+ public**.

---

## Section 1 — Replace `CLAUDE.md` with this content

Скопировать целиком, заменив весь файл. **Ровно 198 строк.**

```markdown
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
│  score → top-n_ranker    │  optuna: n_ranker ∈ [400, 1500] step 50
└────────┬─────────────────┘
         ▼ <= n_ranker × n_users
┌─ CatBoost 2-stage ───────┐  YetiRank GPU, 1023-cap по RRF
│  score → top-100         │  optuna: iter/depth/lr/l2_leaf_reg
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
- **LightGBM** (`src/models/lightgbm_ranker.py`) — lambdarank loss, CPU. Фиксированные гиперпараметры в коде. Обучается один раз, скоры кэшируются. Первая стадия фильтрации до `n_ranker` кандидатов на юзера. Обе колонки **`lgbm_score` (Float32) + `lgbm_rank` (Int32)** идут как фичи в CatBoost — rank инвариантен к scale.
- **CatBoost** (`src/models/catboost_ranker.py`) — YetiRank GPU, 1023-cap внутри `fit()` через RRF. Финальный top-100.

### Optuna (joint_v4)
Поверхность поиска (~14 dim):
- 9 × `n_cand_keep_X` ∈ [0, 800] step 50.
- `n_ranker` ∈ [400, 1500] step 50.
- CatBoost: `iterations` ∈ [1500, 4000] step 250, `depth` ∈ [4, 8], `learning_rate` ∈ [0.02, 0.15] log, `l2_leaf_reg` ∈ [1, 20] log.
- Objective: `0.5 × (recall_val + recall_test)`, multivariate TPE, `n_startup_trials=20`.
- LightGBM hyperparams **зафиксированы** в коде.

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
```bash
# 0) Tests (быстро, локально)
pytest tests/ -v

# 1) Прогрев features + LGBM (ОДИН раз для данного run_id; даёт _train/_eval/_lgbm parquets)
python -u scripts/train_ranker.py data=500m run_id=v4_features \
  feature_chunk_size=5000 enable_embed_features=true n_ranker=1500 \
  2>&1 | tee /tmp/v4_features.log

# 2) Optuna joint (cascade на cached features из шага 1)
python -u scripts/tune.py phase=joint_v2 data=500m run_id=v4_features \
  n_max_per_cg=800 n_cand_min=0 \
  cg_names_list='[decaypop,als,repeat,itemknn,artist_pop,album_pop,recent_likes,audio_knn,esasrec]' \
  study_name=joint_v4 n_trials=80 2>&1 | tee /tmp/joint_v4.log

# 3) Top-K config gen
python scripts/apply_optuna_top_k.py --study-name joint_v4 \
  --storage sqlite:///${HOME}/dc-remote/artifacts/optuna/joint_v4.db \
  --base configs/ranker.yaml --out-prefix configs/ranker_v4_top --top-k 5

# 4) Train + submit одного конфига (повторить для топ-5)
python -u scripts/train_ranker.py --config-name=ranker_v4_top1 \
  data=500m run_id=v4_top1 feature_chunk_size=5000 2>&1 | tee /tmp/v4_top1_train.log
python -u scripts/submit_ranker.py --config-name=ranker_v4_top1 \
  data=500m run_id=v4_top1 2>&1 | tee /tmp/v4_top1_submit.log

# 5) Validate + inspect
python scripts/validate_submission.py submissions/sub_v4_top1_v4_top1.csv
python scripts/inspect_run.py v4_top1
```

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

## Внешние референсы
https://github.com/antklen/recsys_challenge_2025 · NVIDIA 4-stage recommender blueprint (Merlin)
```

(Конец нового CLAUDE.md.)

---

## Section 2 — Implementation tasks (порядок строгий)

### A. Новые фичи (~1.5 ч, риск low)

**Файл**: [src/data/features.py](src/data/features.py).

В `add_features` orchestrator после существующих join'ов **добавить** 5 cross-ratio + 1 embed агрегат:

```python
# После .join(cross_feats, ...) в enriched_cands:
.with_columns([
    # Pair vs user/item normalization — насколько pair "редкий" для юзера/айтема.
    (pl.col("pair_n_listens").cast(pl.Float32)
     / (pl.col("user_n_listens").cast(pl.Float32) + 1.0))
    .cast(pl.Float32).alias("pair_share_user_listens"),
    (pl.col("pair_n_listens").cast(pl.Float32)
     / (pl.col("item_pop").cast(pl.Float32) + 1.0))
    .cast(pl.Float32).alias("pair_share_item_pop"),
    # Item velocity — растущий vs плато vs падающий.
    (pl.col("item_pop_7d").cast(pl.Float32)
     / (pl.col("item_pop_30d").cast(pl.Float32) + 1.0))
    .cast(pl.Float32).alias("item_pop_acceleration"),
    # Recency-driven ratio: насколько пара "свежая" в последних 30d.
    (pl.col("pair_n_listens_30d").cast(pl.Float32)
     / (pl.col("pair_n_listens").cast(pl.Float32) + 1.0))
    .cast(pl.Float32).alias("pair_recency_share_30d"),
    # User-artist consistency (если pair-айтем = любимый артист юзера).
    (pl.col("user_artist_listens").cast(pl.Float32)
     / (pl.col("user_n_listens").cast(pl.Float32) + 1.0))
    .cast(pl.Float32).alias("user_artist_focus"),
    # Embed agg: max cos across windows (most aligned window).
    pl.max_horizontal(
        pl.col("embed_cos_user_last_5"),
        pl.col("embed_cos_user_last_20"),
        pl.col("embed_cos_user_last_50"),
        pl.col("embed_cos_user_last_100"),
    ).cast(pl.Float32).alias("embed_cos_user_last_max"),
])
```

Новые колонки **ниже** уже существующего `pair_recency_ratio` блока (чтобы не нарушить порядок).

**Тесты** в [tests/test_features.py](tests/test_features.py): расширить `test_add_features_emits_pair_recency_ratio_and_embed_cols` — проверить наличие 6 новых колонок.

**Smoke на 50m**: `pytest tests/ -q && python -u scripts/train_ranker.py data=50m run_id=feat_smoke feature_chunk_size=2500 2>&1 | tee /tmp/feat_smoke.log`. Ожидание: features parquet содержит 91 col (85 + 6). Recall val ≥ baseline.

### B. LightGBMRanker class (~2 ч, риск low)

**Зависимость**: добавить `lightgbm==4.5.0` в [requirements.txt](requirements.txt) (после `catboost==1.2.10`). На сервере: `pip install -r requirements.txt`.

**Новый файл**: `src/models/lightgbm_ranker.py`. Интерфейс зеркально к `RankerModel`:

```python
"""Stage-1 cascade ranker: LightGBM lambdarank.

Used to prune the merged candidate pool to ``n_ranker`` per uid before
the heavier CatBoost YetiRank stage. Fixed hyperparams — tune n_ranker
in optuna instead of LGBM internals (3× cheaper trial cost).
"""
from __future__ import annotations
import logging
from typing import Iterable
import lightgbm as lgb
import numpy as np
import polars as pl

log = logging.getLogger(__name__)

_DEFAULT_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    num_leaves=63,
    learning_rate=0.05,
    n_estimators=1500,
    min_child_samples=50,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)
_SKIP_COLS = {"uid", "item_id", "label"}
DEFAULT_SCORE_CHUNK = 500_000


class LightGBMRanker:
    def __init__(self, **overrides):
        self.params = {**_DEFAULT_PARAMS, **overrides}
        self._model: lgb.LGBMRanker | None = None
        self._feature_cols: list[str] = []

    def fit(self, df_train: pl.DataFrame, df_val: pl.DataFrame | None = None) -> None:
        self._feature_cols = [c for c in df_train.columns if c not in _SKIP_COLS]
        df_train = df_train.sort("uid")
        groups_train = df_train.group_by("uid", maintain_order=True).agg(pl.len())["len"].to_list()
        X_train = df_train[self._feature_cols].to_pandas()
        y_train = df_train["label"].to_pandas()
        eval_kwargs = {}
        if df_val is not None:
            df_val = df_val.sort("uid")
            groups_val = df_val.group_by("uid", maintain_order=True).agg(pl.len())["len"].to_list()
            X_val = df_val[self._feature_cols].to_pandas()
            y_val = df_val["label"].to_pandas()
            eval_kwargs = dict(
                eval_set=[(X_val, y_val)],
                eval_group=[groups_val],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
            )
        self._model = lgb.LGBMRanker(**self.params)
        log.info("LightGBMRanker.fit: %d feats, train=%d rows", len(self._feature_cols), len(df_train))
        self._model.fit(X_train, y_train, group=groups_train, **eval_kwargs)
        log.info("LightGBMRanker fitted, best_iter=%s", self._model.best_iteration_ or self.params["n_estimators"])

    def score(self, df: pl.DataFrame, chunk_size: int = DEFAULT_SCORE_CHUNK) -> pl.DataFrame:
        if self._model is None:
            raise RuntimeError("LightGBMRanker.score: not fitted")
        n = len(df)
        if n == 0:
            return df.select(["uid", "item_id"]).with_columns(pl.lit(0.0, dtype=pl.Float32).alias("lgbm_score"))
        feats = self._feature_cols
        if n <= chunk_size:
            scores = self._model.predict(df[feats].to_pandas())
        else:
            parts = []
            for s in range(0, n, chunk_size):
                parts.append(self._model.predict(df.slice(s, chunk_size)[feats].to_pandas()))
            scores = np.concatenate(parts)
        return df.select(["uid", "item_id"]).with_columns(pl.Series("lgbm_score", scores))

    @staticmethod
    def top_k_per_user(df: pl.DataFrame, k: int, score_col: str = "lgbm_score") -> pl.DataFrame:
        return (
            df.sort(["uid", score_col], descending=[False, True])
            .group_by("uid").head(k)
        )

    def predict(self, df: pl.DataFrame, n: int = 100, chunk_size: int = DEFAULT_SCORE_CHUNK) -> pl.DataFrame:
        return self.top_k_per_user(self.score(df, chunk_size), k=n)
```

**Тесты**: `tests/test_lightgbm_ranker.py` — fixture `tiny_labeled` из `test_tune.py`, fit/score smoke + top_k_per_user + ranker_score column присутствует.

### C. Cascade в train_ranker.py (~1.5 ч, риск medium)

**Файл**: [scripts/train_ranker.py](scripts/train_ranker.py).

После Phase 4 (compute_features), перед Phase 5 (CatBoost fit) **вставить**:

```python
# ── 5a. LightGBM stage-1 ─────────────────────────────────────────────────
from src.models.lightgbm_ranker import LightGBMRanker

log.info("loading labeled features for LGBM ← %s", feats_train_path)
labeled_full = pl.read_parquet(feats_train_path)
df_train_lgbm, df_val_lgbm = _split_for_ranker(labeled_full, seed=cfg.seed)

lgbm = LightGBMRanker()  # fixed defaults
lgbm.fit(df_train_lgbm, df_val_lgbm)

lgbm_path = ranker_dir / f"lgbm_{run_id}.pkl"
with open(lgbm_path, "wb") as f:
    pickle.dump(lgbm, f)
log.info("LGBM saved to %s", lgbm_path)

# Score full labeled + eval pools, cache for tune.py reuse.
labeled_lgbm = lgbm.score(labeled_full)
labeled_lgbm_path = features_dir / f"{run_id}_train_lgbm.parquet"
labeled_lgbm.write_parquet(labeled_lgbm_path, compression="zstd")

eval_full = pl.read_parquet(feats_eval_path)
eval_lgbm = lgbm.score(eval_full)
eval_lgbm_path = features_dir / f"{run_id}_eval_lgbm.parquet"
eval_lgbm.write_parquet(eval_lgbm_path, compression="zstd")
log.info("LGBM scores cached: %s, %s", labeled_lgbm_path, eval_lgbm_path)

# ── 5b. Cascade cutoff: keep top-n_ranker per user, add lgbm_rank ────────
n_ranker = int(cfg.get("n_ranker", 1500))  # default no-cascade
log.info("cascade: keeping top-%d per user (LGBM stage-1)", n_ranker)

def _cascade_cut(df_feat, df_lgbm, n):
    """Join LGBM scores, take top-n by lgbm_score per uid, add lgbm_rank.

    Both ``lgbm_score`` AND ``lgbm_rank`` are kept as feature columns for
    the downstream CatBoost — rank is more invariant to score scale across
    optuna trials.
    """
    joined = (
        df_feat.join(df_lgbm, on=["uid", "item_id"], how="left")
        .sort(["uid", "lgbm_score"], descending=[False, True])
        .group_by("uid", maintain_order=True).head(n)
        .with_columns(
            (pl.int_range(1, pl.len() + 1).over("uid").cast(pl.Int32))
            .alias("lgbm_rank")
        )
    )
    return joined

labeled_full = _cascade_cut(labeled_full, labeled_lgbm, n_ranker)
eval_full_cut = _cascade_cut(eval_full, eval_lgbm, n_ranker)
del eval_full
```

Затем существующий код продолжается: `df_train, df_val = _split_for_ranker(labeled_full, ...)` и `ranker = RankerModel(**cfg.ranker); ranker.fit(df_train, df_val)`.

При scoring eval (для val/test recall): использовать `eval_full_cut` вместо `feats_full`. Получаем **`lgbm_score` + `lgbm_rank`** как дополнительные колонки фичей для CatBoost.

**Важно**: добавить `n_ranker: 1500` в `configs/ranker.yaml` (default). В optuna трейлах его override'ит.

### D. Cascade в submit_ranker.py (~1 ч, риск medium)

**Файл**: [scripts/submit_ranker.py](scripts/submit_ranker.py).

После Phase 3 (compute_features → submit_features parquet), перед load ranker.pkl:

```python
# Load both rankers.
lgbm_path = Path(cfg.ranker_dir) / f"lgbm_{run_id}.pkl"
ranker_path = Path(cfg.ranker_dir) / f"ranker_{run_id}.pkl"
with open(lgbm_path, "rb") as f:
    lgbm = pickle.load(f)
with open(ranker_path, "rb") as f:
    ranker = pickle.load(f)

# Score with LGBM, cascade-cut to n_ranker, add lgbm_rank.
log.info("LGBM stage-1 scoring on %d submission rows", len(feats))
lgbm_scores = lgbm.score(feats)
n_ranker = int(cfg.get("n_ranker", 1500))
feats_cut = (
    feats.join(lgbm_scores, on=["uid", "item_id"], how="left")
    .sort(["uid", "lgbm_score"], descending=[False, True])
    .group_by("uid", maintain_order=True).head(n_ranker)
    .with_columns(
        pl.int_range(1, pl.len() + 1).over("uid").cast(pl.Int32).alias("lgbm_rank")
    )
)
log.info("cascade: %d → %d rows after top-%d cut", len(feats), len(feats_cut), n_ranker)

# CatBoost stage-2 + top-K.
preds = ranker.predict(feats_cut, n=cfg.top_k)
```

### E. Cascade в optuna (~1 ч, риск medium)

**Файл**: [src/training/tune.py](src/training/tune.py) — расширить `tune_ranker_and_n_cand_v2` (добавить `lgbm_scores_train_path`/`lgbm_scores_eval_path` параметры), либо новую функцию `tune_cascade`.

Per-trial:
1. Sample `n_cands` (как сейчас) + `n_ranker = trial.suggest_int("n_ranker", 400, 1500, step=50)`.
2. `keep_expr` filter (как сейчас) → `labeled_f`, `eval_f`.
3. Join `lgbm_scores_train`/`_eval` (закешированные) с filtered DFs.
4. Top-`n_ranker` per uid by `lgbm_score` → добавить `lgbm_rank` колонку (`pl.int_range(1, pl.len()+1).over("uid")` после sort) → `labeled_cut`, `eval_cut`. Обе колонки (`lgbm_score`, `lgbm_rank`) идут как фичи в CatBoost.
5. `_split_for_ranker(labeled_cut)` → ranker.fit → score eval_cut → recall.

LGBM **не переобучается per trial** — кэшированные скоры покрывают все trials.

В `scripts/tune.py:_phase_joint_v2` подгружать LGBM-парquet'ы и передавать в tune-функцию.

### F. apply_optuna_top_k.py — inject submission keys (~10 мин, low risk)

**Файл**: [scripts/apply_optuna_top_k.py](scripts/apply_optuna_top_k.py).

В `_apply_trial_to_config` после установки ranker блока:
```python
# submit_ranker.yaml fields — inject so submit_ranker doesn't need
# +submission_dir/+submission_name CLI overrides.
if "submission_dir" not in cfg:
    cfg.submission_dir = "submissions"
if "submission_name" not in cfg:
    cfg.submission_name = "ranker"
# Also seed n_ranker default (optuna picked it).
if "n_ranker" in trial.params and "n_ranker" not in cfg:
    cfg.n_ranker = int(trial.params["n_ranker"])
```

Pre-condition: использовать `OmegaConf.set_struct(cfg, False)` или `with open_dict(cfg)` для добавления новых ключей в struct mode.

После этого CLI `+submission_dir=...` больше не нужен.

### G. Smoke test на 50m (~30 мин)

```bash
# 1) Feature/LGBM smoke
python -u scripts/train_ranker.py data=50m run_id=cascade_smoke \
  feature_chunk_size=2500 n_ranker=600 \
  2>&1 | tee /tmp/cascade_smoke.log

# 2) Submit smoke
python -u scripts/submit_ranker.py data=50m run_id=cascade_smoke \
  +submission_name=smoke 2>&1 | tee /tmp/cascade_smoke_submit.log

# 3) Validate
python scripts/validate_submission.py submissions/sub_cascade_smoke_smoke.csv
pytest tests/ -q
```

Ожидание:
- Logs показывают `LightGBMRanker.fit`, `LGBM saved to ...`, `cascade: keeping top-600 per user`.
- val/test recall в норме (>250 на 50m).
- submission OK (10000 uids, ≤100 items).

---

## Section 3 — Финальные команды (production, после всех имплементаций)

**На сервере**, после `git pull` и `pip install -r requirements.txt`:

```bash
# 0) Verify env
python -c "import lightgbm; print(lightgbm.__version__)"
pytest tests/ -q

# 1) Откатить configs/ranker.yaml до 9 CGs n_cand=800 + добавить n_ranker
#    Если уже так — пропустить. Проверка:
grep -E "n_cand: |n_ranker" configs/ranker.yaml | head -15

# 2) Features + LGBM-scores cache + warmup CatBoost (run_id=v4_features)
set -o pipefail && \
python -u scripts/train_ranker.py data=500m run_id=v4_features \
  feature_chunk_size=5000 enable_embed_features=true n_ranker=1500 \
  2>&1 | tee /tmp/v4_features.log

# 3) Optuna joint (cascade) — 9 CGs incl esasrec, n_ranker tuned
python -u scripts/tune.py phase=joint_v2 data=500m \
  run_id=v4_features \
  n_max_per_cg=800 n_cand_min=0 \
  cg_names_list='[decaypop,als,repeat,itemknn,artist_pop,album_pop,recent_likes,audio_knn,esasrec]' \
  study_name=joint_v4 n_trials=80 \
  2>&1 | tee /tmp/joint_v4.log

# 4) Generate top-5 configs (apply_optuna_top_k теперь injects submission_dir/_name + n_ranker)
python scripts/apply_optuna_top_k.py \
  --study-name joint_v4 \
  --storage sqlite:///${HOME}/dc-remote/artifacts/optuna/joint_v4.db \
  --base configs/ranker.yaml \
  --out-prefix configs/ranker_v4_top --top-k 5

# 5) Train + submit top-5 (LGBM + CatBoost cascade, использует --config-name override)
for i in 1 2 3 4 5; do \
  set -o pipefail && \
  python -u scripts/train_ranker.py \
    --config-name=ranker_v4_top$i \
    data=500m run_id=v4_top$i feature_chunk_size=5000 \
    2>&1 | tee /tmp/v4_top${i}_train.log && \
  python -u scripts/submit_ranker.py \
    --config-name=ranker_v4_top$i \
    data=500m run_id=v4_top$i \
    2>&1 | tee /tmp/v4_top${i}_submit.log; \
done

# 6) RRF blend top-3 best subs (после public LB feedback по top-5)
python scripts/blend_submissions.py \
  --inputs submissions/sub_v4_top<A>_v4_top<A>.csv:0.5 \
           submissions/sub_v4_top<B>_v4_top<B>.csv:0.3 \
           submissions/sub_v2_top5_v2_top5.csv:0.2 \
  --output submissions/sub_v4_blend.csv

# 7) Final
python scripts/validate_submission.py submissions/sub_v4_blend.csv
```

---

## Section 4 — Risk checklist для агента (только локально-проверяемое)

Перед тем как сообщить пользователю "готово к запуску на сервере", агент проверяет ЛОКАЛЬНО:

1. **`requirements.txt` содержит `lightgbm==4.5.0`** (или 4.x). Локально: `python -c "import lightgbm; print(lightgbm.__version__)"` после `pip install -r requirements.txt`.
2. **`configs/ranker.yaml`** (через `git diff` / `cat`):
   - Все 9 CG активны (audio_knn и esasrec **НЕ закомментированы**).
   - `n_cand: 800` для каждого.
   - **Нет полей `n_cand_keep`** в base ranker.yaml (они только в `ranker_v4_topN.yaml` после optuna).
   - `feature_chunk_size: 0` или явно `5000`.
   - `n_ranker: 1500` (default no-cascade до optuna; Section C добавляет этот ключ в base).
3. **`subprocess phase` config_name fix** уже в `_run_phase` (см. [train_ranker.py](scripts/train_ranker.py) и [submit_ranker.py](scripts/submit_ranker.py)). Verify: `grep "config_name" scripts/train_ranker.py | wc -l` ≥ 4.
4. **`apply_optuna_top_k.py` инжектит submission_dir/submission_name + n_ranker** в выходные конфиги (Section F). Без этого `submit_ranker.py` упадёт на `ConfigAttributeError: Key 'submission_dir' is not in struct`. Verify через локальный smoke: `python scripts/apply_optuna_top_k.py --study-name joint_v3 --storage sqlite:///artifacts/optuna/joint_v3.db --base configs/ranker.yaml --out-prefix /tmp/test_top --top-k 1 && grep -E "submission_dir|submission_name|n_ranker" /tmp/test_top1.yaml`.
5. **All tests green**: `pytest tests/ -q` показывает 60+ passed (включая новые тесты на features + LightGBMRanker).
6. **Smoke на 50m прошёл** (Section G): submission CSV валиден, recall ≥ baseline.

Что **не проверяется локально** (ответственность пользователя на сервере):
- наличие CG pickles в `~/dc-remote/artifacts/cg/` (esasrec_500m{,_full}.pkl etc.) — пользователь даёт `force_refit_cg=true` если что.
- `pip install -r requirements.txt` выполнен на сервере.
- `${HOME}/dc-remote/artifacts/optuna/joint_v4.db` пишется и читается.
- Memory: на 25M-row pool LGBM использует ~10-15 GB RAM. На 120 GB-сервере ок. Если OOM — поднять `chunk_size=200_000` в score().

## Section 5 — Если время на исходе (priority cuts)

Если осталось < 10 ч compute:
- **SKIP** новые фичи (Section A) — может лиfta не дать.
- **SKIP** LightGBM (Section B-E) — pure ranker (CatBoost only) с правильным n_cand_keep тоже даёт ~319.
- **DO**: только пересобрать v2 топ-5 с включённым esasrec (повторить старый pipeline). Основная страховка.
- **DO**: H7 RRF blends среди всех subs ≥318.

## Verification после каждой фазы

- `pytest tests/ -q` (61+ тестов зелёные).
- `python scripts/inspect_run.py {run_id}` — best_iter, top features, cg_recall@∞.
- `python scripts/validate_submission.py submissions/sub_*.csv` — формат.
- Submit на public, строка в `docs/experiment-log.md`: `| {run_id_ts} | joint_v4_topN | 500m | val | test | public | notes |`.

## Известные подводные камни

- `embed_cos_user_last_K` (новый, multi-window) — ранкер v2 был обучен на старой колонке `embed_cos_user_last_k`. После H3 (commit d82e3c7) старая колонка УДАЛЕНА. Существующие `ranker_v2_top*.pkl` НЕ работают на новых features. Требуется retrain.
- `cg_count` и `cg_mean_score_norm` уже в фичах (commit b1c3d4c). `cg_min_rank`/`cg_rrf_score`/etc. удалены — **не возвращать** (leak).
- `lgbm_score` — НОВАЯ фича для CatBoost. Чтобы CatBoost мог её использовать, она должна быть в labeled и eval feature parquets ПО ИМЕНИ. Cascade cut применяется ПОСЛЕ join LGBM скоров.
- `--config-name=ranker_v4_topN` — Hydra flag, передаётся parent'у И всем subprocess'ам через `_run_phase` (фикс уже в коде).
- `+submission_dir=submissions +submission_name=v4_top$i` — нужно если apply_optuna_top_k не инжектит ключи. После Section F не нужно. Если Section F скипнут — оставить `+`-overrides.
