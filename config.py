"""All settings for the RAG system. Nothing is hard-coded elsewhere."""

# --- models (Ollama unless noted) ---
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2:3b"
JUDGE_MODEL = "gemma3:4b"  # the spec asked for gemma3:12b; only the 4b is installed
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # sentence-transformers
assert JUDGE_MODEL != GEN_MODEL, "judge model must differ from generator"
GEN_OPTIONS = {"temperature": 0, "num_ctx": 8192}
JUDGE_OPTIONS = {"temperature": 0, "num_ctx": 8192}

# --- chunking and retrieval ---
CHUNK_SIZE = 750
CHUNK_OVERLAP = 100
CANDIDATE_K = 20  # candidates fetched before reranking
FINAL_K = 5       # chunks returned / sent to the LLM
RRF_K = 60        # reciprocal rank fusion constant
EMBED_BATCH = 32

# --- ingest ---
MIN_TEXT_CHARS = 50          # a page with fewer characters is OCRed
OCR_DPI = 200
SKIP_PAGES = {8, 9, 32, 33, 34, 35, 36, 37, 38, 39}  # table-of-contents pages; use set() to keep all

# --- paths ---
PDF_PATH = "data/eval/Rag_Eval.pdf"
EVAL_CSV = "data/eval/rag_eval_30_v2.csv"
CHROMA_DIR = "chroma_db"
COLLECTION = "docs"
RESULTS_DIR = "results"

# --- strategies: name -> (candidate retriever, rerank?) ---
STRATEGIES = {
    "A": ("similarity", False),
    "B": ("similarity", True),
    "C": ("bm25", True),
    "D": ("hybrid", True),
}

# --- phase 2 pipeline: cosine k=20 -> rerank -> top 5 -> remove noise -> LLM ---
PIPELINE_STRATEGY = "B"
NOISE_MIN_SCORE = -3.0  # drop chunks whose rerank score is below this (top chunk is always kept); tune it

# --- RAGAS (judge = JUDGE_MODEL, embeddings = EMBED_MODEL, both served by Ollama's OpenAI-compatible API) ---
RAGAS_BASE_URL = "http://localhost:11434/v1"
COMPLETENESS_VALUES = {"complete": 1.0, "partial": 0.5, "incomplete": 0.0}
COMPLETENESS_PROMPT = (
    "Question: {question}\nReference answer (may be 'not provided'): {reference}\nAnswer: {answer}\n"
    "Does the answer cover every part of the question, and every item of the reference answer if one is given? "
    "Reply complete, partial or incomplete.")
RAGAS_STRICTNESS = 1  # answer_relevancy generates this many questions per answer (RAGAS default 3); 1 is about 3x cheaper
RAGAS_METRICS = ["answer_correctness", "faithfulness", "answer_relevancy", "completeness",
                 "context_precision", "context_recall"]

# --- evaluation ---
FIRST_UNANSWERABLE = 28   # questions 28+ (1-indexed row number) must be refused
SMOKE_QUESTIONS = [1, 11, 28]
WORD_OVERLAP = 0.6        # share of a key fact's words that must appear to count as a match
BEST_METRIC_NO_LLM = "mrr"
BEST_METRIC_LLM = "answer_correctness"

# --- prompts ---
ABSTAIN_TEXT = "I could not find this in the uploaded documents."
SYSTEM_PROMPT = (
    "Answer ONLY from the context. If the answer is not in the context, reply exactly: "
    + ABSTAIN_TEXT + " For list questions, include every item."
)
ANSWER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"

# judge metric -> (allowed labels, prompt). Each prompt must return JSON.
_JSON = ' Reply with JSON only: {{"label": "{labels}", "reason": "<one short line>"}}'
JUDGE_PROMPTS = {
    "correctness": (["correct", "partial", "wrong"],
        "Question: {question}\nReference answer: {gold}\nModel answer: {answer}\n"
        "Is the model answer correct compared with the reference answer?" + _JSON),
    "faithfulness": (["yes", "no"],
        "Context:\n{context}\n\nAnswer: {answer}\nAre ALL claims in the answer supported by the context?" + _JSON),
    "relevancy": (["yes", "no"],
        "Question: {question}\nAnswer: {answer}\nDoes the answer address the question?" + _JSON),
    "completeness": (["yes", "no"],
        "Question: {question}\nReference answer: {gold}\nModel answer: {answer}\n"
        "Does the model answer cover all items of the reference answer?" + _JSON),
    "hallucination": (["yes", "no"],
        "Context:\n{context}\n\nAnswer: {answer}\nDoes the answer contain any claim NOT supported by the context?" + _JSON),
}
JUDGE_SCORES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0, "yes": 1.0, "no": 0.0}
SNAPSHOT_PACKAGES = ["chromadb", "pymupdf", "ollama", "sentence-transformers", "rank-bm25",
                     "langchain-text-splitters", "numpy", "pandas", "streamlit"]
