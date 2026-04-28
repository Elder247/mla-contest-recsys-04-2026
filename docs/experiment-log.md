# Experiment Log

Все прогоны моделей с метриками на val/test. Метрика: `Recall@100 × 1000`.

| run_id | model | data | val | test | notes |
|--------|-------|------|-----|------|-------|
| — | DecayPop (baseline notebook) | 50m | ~50 | — | эталонный скор из условия задачи |
| 20260428_143916 | DecayPop | 50m | 58.03 | 57.19 | train-split модель; sub_001_pop.csv обучена на full data |
| 20260428 | ALSModel (standalone) | 50m | 22.97 | 22.64 | factors=128, iters=20, alpha=40, n_cand=500; 793/10k cold users без предсказаний |
| 20260428 | ALS + CatBoostRanker | 50m | 24.55 | 24.14 | 4 фичи: als_score, als_rank, item_pop, user_n_listens; best_iter=83; 793 cold users без предсказаний; sub_002_als_ranker.csv |
