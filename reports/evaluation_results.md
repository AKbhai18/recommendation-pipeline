# Model Evaluation Results

Metrics computed on the held-out time-based test split (K = 10).

| model | precision@10 | recall@10 | ndcg@10 | coverage | diversity | eval_users | rmse | mae |
|---|---|---|---|---|---|---|---|---|
| popularity_baseline | 0.0224 | 0.033 | 0.0298 | 0.044 | 0.8687 | 998 | 0.7074 | 0.5026 |
| item_based_cf | 0.0574 | 0.0806 | 0.0746 | 0.942 | 0.8734 | 998 | 0.7257 | 0.4398 |
| matrix_factorization_svd | 0.0425 | 0.0615 | 0.0623 | 0.62 | 0.8705 | 998 | 0.6691 | 0.4426 |

**Reading the table**: higher Precision@K / Recall@K / NDCG@K / Coverage / Diversity is better; lower RMSE / MAE is better. The popularity model is the non-personalised baseline; the collaborative-filtering and matrix-factorisation models are the personalised recommenders.
