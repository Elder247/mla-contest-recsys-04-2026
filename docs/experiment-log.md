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
| 20260430_112535 | **A1+ multi-CG + dislike filter** | 50m | 306.23 | 305.91 | **238.3** | те же 3 CG; добавлен фильтр dislikes (107k пар train-period, dropped 10k/3.86M кандидатов = 0.26%); cg_recall@∞ = 500.09; best_iter=141; ranker_003b.pkl, sub_003b_ranker_dislike_filter.csv. **Top features:** repeat_score 29%, user_n_listens 21%, als_score 20%, repeat_rank 11%, item_pop 10%. Локально просело на 3 пункта (no-leak eval), public_lb +0.3 от A1 |
| 20260430_114649 | **A2.3 3-tier ALS + effective_dislikes** | 50m | **308.50** | **307.81** | — | те же 3 CG; ALS теперь видит ВСЕ listens с low=0.3 для ≤50% played, mid=1.0, high=3.0; effective_dislikes (99199 active / 107066 raw / 20941 undislikes); cg_recall@∞ = 501.66; best_iter=196; ranker_004.pkl. **+2.3 vs A1+** — 3-tier ALS даёт лифт. Submit отложен до конца A2 |
