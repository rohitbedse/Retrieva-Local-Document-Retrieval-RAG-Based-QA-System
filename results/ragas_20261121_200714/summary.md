# RAGAS evaluation (phase 2 pipeline)

- questions scored: 1
- noise filter: -3.0
- judge: gemma3:4b (small judge, scores are noisy; NaN = metric call failed)

| metric | mean | NaN count |
|---|---|---|
| answer_correctness | 0.966 | 0 |
| faithfulness | 1.000 | 0 |
| answer_relevancy | 0.927 | 0 |
| completeness | 1.000 | 0 |
| context_precision | 0.887 | 0 |
| context_recall | 1.000 | 0 |

- mean chunks sent to LLM: 5.00
- mean chunks removed as noise: 0.00
