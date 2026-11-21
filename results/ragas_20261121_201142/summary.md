# RAGAS evaluation (phase 2 pipeline)

- questions scored: 10
- noise filter: -3.0
- judge: gemma3:4b (small judge, scores are noisy; NaN = metric call failed)

| metric | mean | NaN count |
|---|---|---|
| answer_correctness | 0.774 | 0 |
| faithfulness | 1.000 | 0 |
| answer_relevancy | 0.632 | 0 |
| completeness | 0.950 | 0 |
| context_precision | 0.931 | 0 |
| context_recall | 1.000 | 0 |

- mean chunks sent to LLM: 4.20
- mean chunks removed as noise: 0.80
