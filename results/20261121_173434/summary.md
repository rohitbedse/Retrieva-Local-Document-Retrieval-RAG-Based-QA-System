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
| answer_correctness | 0.636 |
| score_correctness | 0.981 |
| score_faithfulness | 0.926 |
| score_relevancy | 0.926 |
| score_completeness | 0.815 |
| score_hallucination | 0.185 |
| latency_mean_ms | 1169.453 |
| latency_median_ms | 1133.279 |
| latency_p95_ms | 1503.781 |
| abstain_rate | 1.000 |
| false_refusal_rate | 0.000 |
| e2e_s | 21.331 |
| prompt_tokens | 837.667 |
| completion_tokens | 62.500 |

Best strategy by `answer_correctness`: **B**

## Failing questions (B)

| Q | reason |
|---|---|
| 5 | no relevant chunk in top 5 (gold pages [5, 6], retrieved pages [1, 4, 5, 7]) |
| 9 | relevant chunk retrieved but answer has 33% of key facts |
| 16 | no relevant chunk in top 5 (gold pages [30, 31], retrieved pages [30, 31, 40, 55]) |
| 21 | no relevant chunk in top 5 (gold pages [42], retrieved pages [42, 43, 44, 45]) |
