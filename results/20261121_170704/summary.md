# Evaluation summary

| metric | A | B | C | D |
|---|---|---|---|---|
| hit1 | 0.630 | 0.815 | 0.815 | 0.815 |
| hit3 | 0.778 | 0.889 | 0.889 | 0.889 |
| hit5 | 0.889 | 0.889 | 0.889 | 0.889 |
| mrr | 0.722 | 0.852 | 0.852 | 0.852 |
| ndcg5 | 0.639 | 0.725 | 0.727 | 0.725 |
| precision5 | 0.289 | 0.326 | 0.326 | 0.326 |
| precision1 | 0.630 | 0.815 | 0.815 | 0.815 |
| page_hit1 | 0.852 | 1.000 | 1.000 | 1.000 |
| page_hit3 | 0.926 | 1.000 | 1.000 | 1.000 |
| page_hit5 | 1.000 | 1.000 | 1.000 | 1.000 |
| page_mrr | 0.901 | 1.000 | 1.000 | 1.000 |
| page_ndcg5 | 0.624 | 0.664 | 0.661 | 0.664 |
| page_precision5 | 0.526 | 0.548 | 0.548 | 0.548 |
| page_precision1 | 0.852 | 1.000 | 1.000 | 1.000 |
| recall5 | 0.734 | 0.792 | 0.755 | 0.792 |
| latency_mean_ms | 106.611 | 961.924 | 824.337 | 996.326 |
| latency_median_ms | 102.669 | 958.625 | 828.571 | 973.966 |
| latency_p95_ms | 144.723 | 1343.701 | 1116.346 | 1325.900 |

Best strategy by `mrr`: **B**

## Failing questions (B)

| Q | reason |
|---|---|
| 5 | no relevant chunk in top 5 (gold pages [5, 6], retrieved pages [1, 4, 5, 7]) |
| 16 | no relevant chunk in top 5 (gold pages [30, 31], retrieved pages [30, 31, 40, 55]) |
| 21 | no relevant chunk in top 5 (gold pages [42], retrieved pages [42, 43, 44, 45]) |
