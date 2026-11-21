"""Evaluate retrieval strategies on the question set. Logs to results/<timestamp>/ only.

Usage: python evaluate.py --strategies A,B,C,D [--no-llm] [--smoke]
"""
import argparse
import json
import math
import os
import platform
import re
import sys
import time
from datetime import datetime
from importlib.metadata import version

import ollama
import pandas as pd

import config
import retrieve
from ingest import ingest

METRIC_PREFIXES = ("hit", "mrr", "ndcg", "precision", "page_", "recall", "answer_correctness", "score_")


# Lowercase word list.
def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# True if at least WORD_OVERLAP of the fact's words occur in the text.
def fact_in(fact, text):
    fact_words = set(words(fact))
    return bool(fact_words) and len(fact_words & set(words(text))) / len(fact_words) >= config.WORD_OVERLAP


# Share of key facts found in the text.
def coverage(facts, text):
    return sum(fact_in(f, text) for f in facts) / len(facts)


# Parse "5-6" or "53, 55" into a set of page numbers ("none" -> empty set).
def parse_pages(text):
    pages = set()
    for part in re.split(r"[,;]", str(text)):
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", part)
        if m:
            pages |= set(range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1))
    return pages


# Is a chunk relevant? Gold page, plus (unless page_only) at least one key fact in its text.
def is_relevant(chunk, gold_pages, facts, page_only):
    return chunk[1] in gold_pages and (page_only or any(fact_in(f, chunk[0]) for f in facts))


# Hit@1/3/5, MRR, nDCG@5, Precision@5/@1 from a list of relevance flags; also the rank of the first hit.
def rank_metrics(flags, total_relevant, prefix):
    first = next((i + 1 for i, f in enumerate(flags) if f), None)
    dcg = sum(f / math.log2(i + 2) for i, f in enumerate(flags))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(total_relevant, len(flags))))
    metrics = {"hit1": float(any(flags[:1])), "hit3": float(any(flags[:3])), "hit5": float(any(flags[:5])),
               "mrr": 1 / first if first else 0.0, "ndcg5": dcg / ideal if ideal else 0.0,
               "precision5": sum(flags) / len(flags), "precision1": float(flags[0])}
    return {prefix + k: v for k, v in metrics.items()}, first


# Ask JUDGE_MODEL one yes/no (or correct/partial/wrong) question; returns {"label", "reason"}.
def judge(metric, **fields):
    labels, template = config.JUDGE_PROMPTS[metric]
    prompt = template.format(labels="|".join(labels), **fields)
    try:
        reply = ollama.chat(model=config.JUDGE_MODEL, options=config.JUDGE_OPTIONS, format="json",
                            messages=[{"role": "user", "content": prompt}])
        out = json.loads(reply["message"]["content"])
        label = str(out.get("label", "")).lower().strip()
        return {"label": label if label in labels else "error", "reason": str(out.get("reason", ""))}
    except Exception as e:
        return {"label": "error", "reason": str(e)}


# Evaluate one (question, strategy): retrieval metrics, then optionally the LLM answer and judges.
def evaluate_one(row, strategy, use_llm):
    question, gold, facts = row["question"], row["gold_pages"], row["facts"]
    start = time.perf_counter()
    chunks = retrieve.retrieve(question, strategy)
    rec = {"qid": row["qid"], "strategy": strategy, "question": question, "unans": row["unans"],
           "retrieval_ms": (time.perf_counter() - start) * 1000,
           "top5": [{"page": c[1], "chunk_index": c[2], "score": c[3]} for c in chunks]}
    if not row["unans"]:
        for page_only, prefix in ((False, ""), (True, "page_")):
            flags = [is_relevant(c, gold, facts, page_only) for c in chunks]
            total = sum(is_relevant(c, gold, facts, page_only) for c in retrieve.all_chunks())
            metrics, first = rank_metrics(flags, total, prefix)
            rec.update(metrics)
            rec[prefix + "first_relevant_rank"] = first
        rec["recall5"] = coverage(facts, " ".join(c[0] for c in chunks))
    if use_llm:
        gen = retrieve.generate(question, chunks)
        rec.update(gen, e2e_s=time.perf_counter() - start)
        rec["abstained"] = config.ABSTAIN_TEXT.lower().rstrip(".") in gen["answer"].lower()
        if not row["unans"]:
            rec["answer_correctness"] = coverage(facts, gen["answer"])
            fields = {"question": question, "gold": row["gold_answer"], "answer": gen["answer"],
                      "context": "\n\n".join(c[0] for c in chunks)}
            rec["judge"] = {m: judge(m, **fields) for m in config.JUDGE_PROMPTS}
            for m, j in rec["judge"].items():
                rec["score_" + m] = config.JUDGE_SCORES.get(j["label"])
    return rec


# One summary row per strategy: mean of every metric over Q1-27, latency stats, abstain rates.
def summarize(df):
    rows = []
    for strategy, g in df.groupby("strategy"):
        a = g[~g["unans"]]
        row = {"strategy": strategy}
        for col in df.columns:
            if col.startswith(METRIC_PREFIXES) and not col.endswith("first_relevant_rank"):
                row[col] = a[col].mean()
        ms = g["retrieval_ms"]
        row.update(latency_mean_ms=ms.mean(), latency_median_ms=ms.median(), latency_p95_ms=ms.quantile(0.95))
        if "abstained" in g:
            row.update(abstain_rate=g[g["unans"]]["abstained"].mean(), false_refusal_rate=a["abstained"].mean(),
                       e2e_s=g["e2e_s"].mean(), prompt_tokens=g["prompt_tokens"].mean(),
                       completion_tokens=g["completion_tokens"].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")


# Why did this (question, strategy) fail? Returns a reason string, or None if it passed.
def failure_reason(rec, gold_pages):
    if rec["unans"]:
        return "answered an unanswerable question" if rec.get("abstained") is False else None
    pages = sorted({c["page"] for c in rec["top5"]})
    if rec["hit5"] == 0:
        return f"no relevant chunk in top 5 (gold pages {sorted(gold_pages)}, retrieved pages {pages})"
    if rec.get("abstained"):
        return "false refusal although a relevant chunk was retrieved"
    if rec.get("answer_correctness", 1) < 0.5:
        return f"relevant chunk retrieved but answer has {rec['answer_correctness']:.0%} of key facts"
    return None


# Write summary.md: metrics x strategies table, best strategy, failing questions.
def write_markdown(path, summary, records, golds, use_llm):
    metric = config.BEST_METRIC_LLM if use_llm else config.BEST_METRIC_NO_LLM
    best = summary[metric].idxmax()
    lines = ["# Evaluation summary", "", "| metric | " + " | ".join(summary.index) + " |",
             "|---|" + "---|" * len(summary)]
    for col in summary.columns:
        lines.append(f"| {col} | " + " | ".join(f"{v:.3f}" for v in summary[col]) + " |")
    lines += ["", f"Best strategy by `{metric}`: **{best}**", "", f"## Failing questions ({best})", "",
              "| Q | reason |", "|---|---|"]
    for rec in records:
        reason = failure_reason(rec, golds[rec["qid"]]) if rec["strategy"] == best else None
        if reason:
            lines.append(f"| {rec['qid']} | {reason} |")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


# Save every setting, model name and package version for reproducibility.
def write_snapshot(path, args):
    settings = {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(config).items() if k.isupper()}
    packages = {}
    for name in config.SNAPSHOT_PACKAGES:
        try:
            packages[name] = version(name)
        except Exception:
            packages[name] = "not installed"
    info = {"settings": settings, "packages": packages, "python": sys.version, "platform": platform.platform(),
            "args": vars(args), "time": datetime.now().isoformat()}
    json.dump(info, open(path, "w"), indent=2)


# Stop loudly if BM25 and similarity return the same candidate list (BM25 would be redundant).
def check_bm25_differs(question):
    sim = [(c[1], c[2]) for c in retrieve.similarity(question, config.CANDIDATE_K)]
    bm = [(c[1], c[2]) for c in retrieve.bm25(question, config.CANDIDATE_K)]
    print("[check] similarity candidates:", sim, "\n[check] bm25 candidates:      ", bm)
    if sim == bm:
        sys.exit("[check] FAILED: BM25 and similarity candidate lists are identical")
    print(f"[check] OK, overlap {len(set(sim) & set(bm))}/{config.CANDIDATE_K}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", default=",".join(config.STRATEGIES))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="3 questions only")
    args = parser.parse_args()
    strategies, use_llm = args.strategies.upper().split(","), not args.no_llm

    table = pd.read_csv(config.EVAL_CSV)
    rows = [{"qid": n, "question": r["question"], "gold_answer": r["gold_answer"],
             "unans": n >= config.FIRST_UNANSWERABLE, "gold_pages": parse_pages(r["pdf_page(s)"]),
             "facts": [f.strip() for f in str(r["key_facts"]).split(";") if words(f)]}
            for n, (_, r) in enumerate(table.iterrows(), 1)]
    if args.smoke:
        rows = [r for r in rows if r["qid"] in config.SMOKE_QUESTIONS]

    ingest(config.PDF_PATH)
    check_bm25_differs(rows[0]["question"])
    for s in strategies:  # warm-up so model loading is not counted as latency
        retrieve.retrieve("warm up", s)

    out_dir = os.path.join(config.RESULTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir)  # never reuses an old run folder
    write_snapshot(os.path.join(out_dir, "config_snapshot.json"), args)
    records = []
    with open(os.path.join(out_dir, "metrics.jsonl"), "w", encoding="utf-8") as log:
        for row in rows:
            for s in strategies:
                rec = evaluate_one(row, s, use_llm)
                records.append(rec)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"Q{row['qid']:>2} {s} retrieval {rec['retrieval_ms']:.0f} ms", flush=True)

    df = pd.DataFrame(records)
    for col in df.columns:
        if col.startswith("score_"):
            df[col] = pd.to_numeric(df[col])
    summary = summarize(df)
    summary.to_csv(os.path.join(out_dir, "summary.csv"))
    write_markdown(os.path.join(out_dir, "summary.md"), summary, records,
                   {r["qid"]: r["gold_pages"] for r in rows}, use_llm)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
