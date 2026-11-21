# Intelligent PDF RAG System

A fully local PDF question-answering system (no paid APIs) with a two-phase evaluation:
**retrieval-level** metrics (no LLM) and **application-level** metrics (LLM answers, an LLM judge, and RAGAS).

**Stack:** Python, PyMuPDF, ChromaDB, Ollama (`nomic-embed-text`, `llama3.2:3b`), sentence-transformers cross-encoder, rank_bm25, Streamlit, RAGAS.

## How it works

```
PDF -> PyMuPDF text (OCR fallback for sparse pages) -> chunks (750 / 100 overlap)
    -> Ollama embeddings -> ChromaDB (cosine, persistent, cached)

Question -> cosine top-20 -> cross-encoder rerank -> top-5 -> noise filter -> Llama 3.2 3B -> answer
```

- **Noise filter:** drops reranked chunks whose rerank score is below `NOISE_MIN_SCORE` (-3.0, not tuned); the top chunk is always kept.
- **Abstention:** the prompt tells the model to reply exactly `I could not find this in the uploaded documents.` when the context has no answer.
- **Four retrieval strategies** are compared (set in `config.py`):

| ID | Strategy |
|---|---|
| A | cosine similarity, k=5 |
| B | cosine k=20 -> cross-encoder rerank -> top 5 (**used by the app**) |
| C | BM25 k=20 -> rerank -> top 5 |
| D | hybrid (cosine + BM25, reciprocal rank fusion) k=20 -> rerank -> top 5 |

## Files

| File | Purpose |
|---|---|
| `config.py` | every setting: models, chunk size, k values, paths, prompts, thresholds |
| `ingest.py` | PDF -> pages (OCR if a page has almost no text) -> chunks -> embeddings -> Chroma (cached by file hash + settings) |
| `retrieve.py` | `similarity()`, `bm25()`, `hybrid()`, `rerank()`, `remove_noise()`, `answer()`, `pipeline()` |
| `evaluate.py` | retrieval metrics + application metrics + LLM judge; logs to `results/<timestamp>/` |
| `ragas_eval.py` | RAGAS scoring (six metrics) of the full pipeline |
| `app.py` | Streamlit UI: ask a question, see answer, kept and removed chunks, timings, optional RAGAS scores |
| `data/eval/` | `Rag_Eval.pdf` (corpus) and `rag_eval_30_v2.csv` (30 questions, Q28-30 unanswerable) |
| `old/` | archived first version (not used) |

## Setup

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2:3b
ollama pull gemma3:4b          # judge model (must differ from the generator)
```

Optional: install the Tesseract program if you want OCR for scanned PDFs (see Limitations).

## Run

```powershell
venv\Scripts\python ingest.py                                   # build the index (cached afterwards)
venv\Scripts\streamlit run app.py                               # UI

venv\Scripts\python evaluate.py --smoke --no-llm                # quick retrieval check
venv\Scripts\python evaluate.py --strategies A,B,C,D --no-llm   # retrieval evaluation
venv\Scripts\python evaluate.py --strategies B                  # application evaluation (slow)
venv\Scripts\python ragas_eval.py --limit 10                    # RAGAS on the first 10 questions
venv\Scripts\python ragas_eval.py --questions 4,6,7 --strictness 3
```

Each run writes a new folder, nothing is overwritten:
`results/<timestamp>/{metrics.jsonl, summary.csv, summary.md, config_snapshot.json}` and `results/ragas_<timestamp>/{ragas_scores.csv, summary.md}`.

## Evaluation setup

- **Dataset:** 30 questions on a 58-page PostgreSQL manual. Q1-27 are answerable (gold answer, key facts, gold pages); Q28-30 are unanswerable and must be refused.
- **Table-of-contents pages 8-9 and 32-39 are skipped** at ingest (`SKIP_PAGES`).
- **Relevant chunk** = its page is a gold page **and** it contains at least one key fact (a fact matches when at least 60% of its words appear). A page-only version is also reported.
- **Judge model:** `gemma3:4b` (the spec asked for 12b; only 4b was installed). A small judge is noisy.

## Phase 1: retrieval evaluation (no LLM, Q1-27)

Run `20261121_170704`, strategies A-D.

| Metric | A cosine k=5 | B cosine + rerank | C BM25 + rerank | D hybrid + rerank |
|---|---|---|---|---|
| Hit@1 | 0.630 | **0.815** | 0.815 | 0.815 |
| Hit@3 | 0.778 | 0.889 | 0.889 | 0.889 |
| Hit@5 | 0.889 | 0.889 | 0.889 | 0.889 |
| MRR | 0.722 | **0.852** | 0.852 | 0.852 |
| nDCG@5 | 0.639 | 0.725 | 0.727 | 0.725 |
| Precision@1 | 0.630 | 0.815 | 0.815 | 0.815 |
| Precision@5 | 0.289 | 0.326 | 0.326 | 0.326 |
| Recall@5 (key facts) | 0.734 | 0.792 | 0.755 | 0.792 |
| Page-only Hit@1 | 0.852 | 1.000 | 1.000 | 1.000 |
| Page-only Hit@5 | 1.000 | 1.000 | 1.000 | 1.000 |
| Page-only MRR | 0.901 | 1.000 | 1.000 | 1.000 |
| Latency mean (ms) | 107 | 962 | 824 | 996 |
| Latency median (ms) | 103 | 959 | 829 | 974 |
| Latency p95 (ms) | 145 | 1344 | 1116 | 1326 |

**Takeaways**
- Reranking is the main gain: Hit@1 0.63 -> 0.81 and MRR 0.72 -> 0.85. Hit@5 does not change (0.89).
- B, C and D are nearly identical. With a 162-chunk document and 20 candidates, the candidate pools overlap heavily and the cross-encoder picks the same top 5. BM25 and similarity candidate lists do differ (a sanity check in `evaluate.py` enforces this).
- Reranking costs about 0.8-1.0 s per query on CPU versus about 0.1 s for cosine only. Latency for B varied between 0.6 s and 1.2 s across runs.
- Precision@5 is low by design: most questions have one or two relevant chunks among five.

## Phase 2: application-level evaluation

### 2a. End-to-end run with LLM judge (strategy B, 30 questions)

Run `20261121_173434`. Generator `llama3.2:3b`, judge `gemma3:4b`.

| Metric | Value |
|---|---|
| Key-fact answer correctness (mean, Q1-27) | 0.636 |
| Answers with all key facts / at least half | 8 / 27 and 23 / 27 |
| Judge correctness (correct = 1, partial = 0.5) | 0.981 (26 correct, 1 partial) |
| Judge faithfulness (yes rate) | 0.926 |
| Judge relevancy (yes rate) | 0.926 |
| Judge completeness (yes rate) | 0.815 |
| Judge hallucination (yes rate, lower is better) | 0.185 |
| Abstain rate on Q28-30 | 1.000 (3 of 3 refused) |
| False-refusal rate on Q1-27 | 0.000 |
| End-to-end latency (mean) | 21.3 s |
| Prompt / completion tokens (mean) | 838 / 62.5 |

The judge scores are far more generous than the key-fact score (0.98 vs 0.64). Treat the 4B judge as optimistic.

### 2b. RAGAS (pipeline B + noise filter, first 10 questions)

Run `ragas_20261121_201142`. Judge `gemma3:4b`, embeddings `nomic-embed-text`, answer relevancy with `strictness=1`.

| Metric | Mean (n=10) |
|---|---|
| Faithfulness | 1.000 |
| Context recall | 1.000 |
| Completeness (custom RAGAS metric: complete / partial / incomplete) | 0.950 |
| Context precision | 0.931 |
| Answer correctness | 0.774 |
| Answer relevancy | 0.632 (unreliable, see below) |

Per-question: lowest answer correctness is Q5 (0.45; the answer listed 2 of 7 required items), then Q9 (0.58), Q4 (0.66), Q7 (0.68), Q3 (0.69). Time was about 415 s per question on CPU.

**Findings**
- Q3, Q4, Q9: correct but shorter than the reference; RAGAS correctness penalizes the missing reference claims.
- Q5 is a real miss: the chunks contained the full list but the 3B generator returned 2 items.
- Answer relevancy scored 0.00 on Q4, Q6 and Q7 although the answers are on-topic. The likely cause is the small judge flagging them as non-committal, made worse by `strictness=1`. This is not yet confirmed; re-run with `--questions 4,6,7 --strictness 3`.

## Limitations and known issues

- **Small samples:** 27 answerable questions for retrieval, 10 for RAGAS. The first 10 questions all come from the Preface and bug-report sections. Differences of a few points are noise.
- **Small judge:** `gemma3:4b` makes judge and RAGAS scores noisy.
- **OCR is untested:** the code path exists, but Tesseract is not installed and the test PDF has text on every page.
- **Noise cutoff (-3.0) is untuned:** no comparison against `--no-filter` has been run.
- **RAGAS is slow on CPU** (about 7 minutes per question).
- **Metric definitions are lenient:** key-fact matching uses 60% word overlap, so a few paraphrased facts count as matches.
- Retrieval and application numbers come from different runs; re-running changes LLM-dependent numbers slightly.
