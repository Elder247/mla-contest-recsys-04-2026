# Experiment Log

Все прогоны моделей с метриками на val/test. Метрика: `Recall@100 × 1000`.

| run_id | model | data | val | test | notes |
|--------|-------|------|-----|------|-------|
| — | DecayPop (baseline notebook) | 50m | ~50 | — | эталонный скор из условия задачи |
| 20260428_143916 | DecayPop | 50m | 58.03 | 57.19 | train-split модель; sub_001_pop.csv обучена на full data |
