"""Score the phase 2 pipeline with RAGAS (judge and embeddings run locally through Ollama).

Usage: python ragas_eval.py [--smoke] [--limit N] [--questions 4,6,7] [--strictness 3] [--no-filter]
Metrics: answer_correctness, faithfulness, answer_relevancy, completeness, context_precision, context_recall.
"""
import argparse
import os
import time
from datetime import datetime

import instructor
import pandas as pd
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (AnswerCorrectness, AnswerRelevancy, ContextPrecision,
                                       ContextRecall, Faithfulness)

import config
import retrieve
from ingest import ingest


# Build the RAGAS judge LLM, embeddings and metric objects, all pointing at local Ollama.
def make_scorers(strictness=config.RAGAS_STRICTNESS):
    client = AsyncOpenAI(base_url=config.RAGAS_BASE_URL, api_key="ollama", timeout=600)
    llm = llm_factory(config.JUDGE_MODEL, provider="openai", client=client, mode=instructor.Mode.JSON)
    emb = embedding_factory("openai", model=config.EMBED_MODEL, client=client)
    completeness = DiscreteMetric(name="completeness", prompt=config.COMPLETENESS_PROMPT,
                                  allowed_values=list(config.COMPLETENESS_VALUES))
    return {"llm": llm, "faithfulness": Faithfulness(llm), "answer_relevancy": AnswerRelevancy(llm, emb, strictness=strictness),
            "answer_correctness": AnswerCorrectness(llm, emb), "completeness": completeness,
            "context_precision": ContextPrecision(llm), "context_recall": ContextRecall(llm)}


# Run one metric call; a failure (small judges often break the JSON format) becomes NaN, not a crash.
def safe(name, call):
    try:
        value = call().value
        return config.COMPLETENESS_VALUES.get(value, float("nan")) if isinstance(value, str) else float(value)
    except Exception as e:
        print(f"  [ragas] {name} failed: {type(e).__name__}: {str(e)[:120]}")
        return float("nan")


# Score one answer. Without a reference only the three reference-free metrics are computed.
def score_one(scorers, question, answer, contexts, reference=None):
    s = scorers
    scores = {
        "faithfulness": safe("faithfulness", lambda: s["faithfulness"].score(
            user_input=question, response=answer, retrieved_contexts=contexts)),
        "answer_relevancy": safe("answer_relevancy", lambda: s["answer_relevancy"].score(
            user_input=question, response=answer)),
        "completeness": safe("completeness", lambda: s["completeness"].score(
            llm=s["llm"], question=question, answer=answer, reference=reference or "not provided")),
    }
    if reference:
        scores["answer_correctness"] = safe("answer_correctness", lambda: s["answer_correctness"].score(
            user_input=question, response=answer, reference=reference))
        scores["context_precision"] = safe("context_precision", lambda: s["context_precision"].score(
            user_input=question, reference=reference, retrieved_contexts=contexts))
        scores["context_recall"] = safe("context_recall", lambda: s["context_recall"].score(
            user_input=question, retrieved_contexts=contexts, reference=reference))
    return scores


# Write the markdown summary: mean per metric, number of failed (NaN) scores, noise settings.
def write_summary(path, df, args):
    lines = ["# RAGAS evaluation (phase 2 pipeline)", "",
             f"- questions scored: {len(df)}", f"- noise filter: {'off' if args.no_filter else config.NOISE_MIN_SCORE}",
             f"- judge: {config.JUDGE_MODEL} (small judge, scores are noisy; NaN = metric call failed)", "",
             "| metric | mean | NaN count |", "|---|---|---|"]
    for m in config.RAGAS_METRICS:
        lines.append(f"| {m} | {df[m].mean():.3f} | {int(df[m].isna().sum())} |")
    lines += ["", f"- mean chunks sent to LLM: {df['chunks_sent'].mean():.2f}",
              f"- mean chunks removed as noise: {df['chunks_removed'].mean():.2f}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="2 answerable questions only")
    parser.add_argument("--limit", type=int, help="score only the first N answerable questions")
    parser.add_argument("--questions", help="comma-separated question numbers, e.g. 4,6,7")
    parser.add_argument("--strictness", type=int, default=config.RAGAS_STRICTNESS,
                        help="answer_relevancy samples per answer (RAGAS default 3)")
    parser.add_argument("--no-filter", action="store_true", help="skip noise removal (to compare)")
    args = parser.parse_args()

    table = pd.read_csv(config.EVAL_CSV)
    questions = [(n, r) for n, (_, r) in enumerate(table.iterrows(), 1) if n < config.FIRST_UNANSWERABLE]
    if args.smoke:
        questions = [q for q in questions if q[0] in config.SMOKE_QUESTIONS]
    if args.questions:
        wanted = {int(x) for x in args.questions.split(",")}
        questions = [q for q in questions if q[0] in wanted]
    if args.limit:
        questions = questions[:args.limit]
    ingest(config.PDF_PATH)
    scorers = make_scorers(args.strictness)
    out_dir = os.path.join(config.RESULTS_DIR, "ragas_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir)

    rows = []
    for n, r in questions:
        start = time.perf_counter()
        result = retrieve.pipeline(r["question"], remove=not args.no_filter)
        contexts = [c[0] for c in result["kept"]]
        scores = score_one(scorers, r["question"], result["answer"], contexts, r["gold_answer"])
        rows.append({"qid": n, "question": r["question"], "answer": result["answer"],
                     "kept_pages": [c[1] for c in result["kept"]], "chunks_sent": len(result["kept"]),
                     "chunks_removed": len(result["removed"]), **scores})
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ragas_scores.csv"), index=False)  # saved every question
        print(f"Q{n:>2} ({time.perf_counter() - start:.0f}s) " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    write_summary(os.path.join(out_dir, "summary.md"), pd.DataFrame(rows), args)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
