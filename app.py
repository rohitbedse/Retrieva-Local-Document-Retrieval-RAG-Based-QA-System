"""Streamlit UI: ask a question -> cosine k=20 -> rerank -> top 5 -> remove noise -> answer -> optional RAGAS."""
import os
import tempfile

import pandas as pd
import streamlit as st

import config
import ragas_eval
import retrieve
from ingest import ingest


# Build the RAGAS scorers once per session.
@st.cache_resource
def get_scorers():
    return ragas_eval.make_scorers()


# Index an uploaded PDF (cached by content) and reset the in-memory chunk caches.
def index_upload(upload):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(upload.getvalue())
    ingest(f.name)
    os.unlink(f.name)
    retrieve.all_chunks.cache_clear()
    retrieve.bm25_index.cache_clear()


# Show a list of chunks as expanders with page, chunk index and rerank score.
def show_chunks(chunks):
    for text, page, index, score in chunks:
        with st.expander("page %d, chunk %d, rerank score %.2f" % (page, index, score)):
            st.text(text)


st.title("Document Q&A (cosine k=20 + rerank + noise removal)")
upload = st.file_uploader("PDF (optional: leave empty to use the already indexed document)", type="pdf")
question = st.text_input("Question")

if st.button("Ask") and question:
    if upload:
        with st.spinner("Indexing (cached if unchanged)..."):
            index_upload(upload)
    try:
        with st.spinner("Searching and answering..."):
            st.session_state["result"] = retrieve.pipeline(question)
            st.session_state["question"] = question
    except Exception as e:
        st.error("No index found or Ollama is not running: %s" % e)

result = st.session_state.get("result")
if result:
    st.subheader("Answer")
    st.write(result["answer"])
    st.caption("retrieval %.0f ms, total %.1f s, %d chunks sent, %d removed as noise" % (
        result["retrieval_ms"], result["total_s"], len(result["kept"]), len(result["removed"])))
    st.subheader("Chunks sent to the LLM")
    show_chunks(result["kept"])
    if result["removed"]:
        st.subheader("Removed as noise")
        show_chunks(result["removed"])

    st.subheader("Evaluate with RAGAS")
    reference = st.text_area("Reference (gold) answer, optional. Needed for correctness, context precision, context recall.")
    if st.button("Run RAGAS"):
        with st.spinner("Scoring with %s (slow on CPU)..." % config.JUDGE_MODEL):
            scores = ragas_eval.score_one(get_scorers(), st.session_state["question"], result["answer"],
                                          [c[0] for c in result["kept"]], reference or None)
        st.table(pd.DataFrame({"score": scores}))
        st.caption("Small judge model: scores are noisy. NaN means the metric call failed.")
