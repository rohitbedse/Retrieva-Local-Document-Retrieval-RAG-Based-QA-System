# Evaluation summary

| metric | B |
|---|---|
| hit1 | 0.815 |
| hit3 | 0.889 |
| hit5 | 0.889 |
| mrr | 0.852 |
| ndcg5 | 0.725 |
| precision5 | 0.326 |
| precision1 | 0.815 |
| page_hit1 | 1.000 |
| page_hit3 | 1.000 |
| page_hit5 | 1.000 |
| page_mrr | 1.000 |
| page_ndcg5 | 0.664 |
| page_precision5 | 0.548 |
| page_precision1 | 1.000 |
| recall5 | 0.792 |
| latency_mean_ms | 623.654 |
| latency_median_ms | 601.204 |
| latency_p95_ms | 797.427 |

Best strategy by `mrr`: **B**

## Failing questions (B)

| Q | reason |
|---|---|
| 5 | no relevant chunk in top 5 (gold pages [5, 6], retrieved pages [1, 4, 5, 7]) |
| 16 | no relevant chunk in top 5 (gold pages [30, 31], retrieved pages [30, 31, 40, 55]) |
| 21 | no relevant chunk in top 5 (gold pages [42], retrieved pages [42, 43, 44, 45]) |
