# Experiment Log

Все прогоны моделей с метриками на val/test. Метрика: `Recall@100 × 1000`.

`public_lb` — балл с публичного лидерборда (когда сабмит реально отправлен). Полезен для отслеживания gap'а между локальной val и публичной метрикой.

| run_id | model | data | val | test | public_lb | notes |
|--------|-------|------|-----|------|-----------|-------|
| — | DecayPop (baseline notebook) | 50m | ~50 | — | — | эталонный скор из условия задачи |
| 20260428_143916 | DecayPop | 50m | 58.03 | 57.19 | 49.9 | train-split модель; sub_001_pop.csv обучена на full data |
| 20260428 | ALSModel (standalone) | 50m | 22.97 | 22.64 | — | factors=128, iters=20, alpha=40, n_cand=500; 793/10k cold users без предсказаний |
| 20260428 | ALS + CatBoostRanker | 50m | 24.55 | 24.14 | — | 4 фичи: als_score, als_rank, item_pop, user_n_listens; best_iter=83; 793 cold users без предсказаний; sub_002_als_ranker.csv |
| 20260428_215956 | ALS + CatBoostRanker (filter_already_liked_items=False) | 50m | 132.12 | 130.84 | 106.5 | фикс: предсказания для seen-айтемов больше не отбрасываются; те же 4 фичи |
| 20260429_173718 | **A1 multi-CG ranker** [decaypop+als(weighted)+repeat] | 50m | **309.27** | **308.38** | **238** | 8 фич (3 score + 3 rank + item_pop + user_n_listens); ALS weighted (high=3 mid=1 thr=80); cg_recall@∞ = 503.1; best_iter=213; sub_003_ranker.csv. **gap val→public ≈71** |
