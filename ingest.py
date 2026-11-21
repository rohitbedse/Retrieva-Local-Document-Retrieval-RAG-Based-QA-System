"""Read a PDF, OCR sparse pages, chunk, embed, and store in a persistent Chroma collection."""
import hashlib
import io
import os

import chromadb
import fitz  # PyMuPDF
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


# OCR one page image with pytesseract; return "" if Tesseract is not installed.
def ocr_page(page):
    try:
        import pytesseract
        from PIL import Image
        png = page.get_pixmap(dpi=config.OCR_DPI).tobytes("png")
        return pytesseract.image_to_string(Image.open(io.BytesIO(png)))
    except Exception as e:  # missing Tesseract binary, bad image, ...
        print(f"[ingest] OCR unavailable for page {page.number + 1}: {type(e).__name__}")
        return ""


# Return [(page_number_1_indexed, text, method)] for every non-skipped page.
def read_pages(pdf_path):
    pages = []
    for page in fitz.open(pdf_path):
        number = page.number + 1
        if number in config.SKIP_PAGES:
            continue
        text, method = page.get_text().strip(), "text"
        if len(text) < config.MIN_TEXT_CHARS:
            ocr_text = ocr_page(page).strip()
            if len(ocr_text) > len(text):
                text, method = ocr_text, "ocr"
        if text:
            pages.append((number, text, method))
    return pages


# Split pages into chunk dicts with source, page, chunk_index and method.
def split_pages(pages, source):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = []
    for number, text, method in pages:
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append({"text": piece, "source": source, "page": number,
                           "chunk_index": i, "method": method})
    return chunks


# Embed a list of texts with Ollama, in batches.
def embed(texts):
    vectors = []
    for start in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[start:start + config.EMBED_BATCH]
        vectors += ollama.embed(model=config.EMBED_MODEL, input=batch)["embeddings"]
    return vectors


# Hash of the PDF bytes plus every setting that changes the index (the cache key).
def fingerprint(pdf_path):
    settings = (config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.EMBED_MODEL,
                config.MIN_TEXT_CHARS, sorted(config.SKIP_PAGES))
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read() + repr(settings).encode()).hexdigest()


# Return the Chroma collection for this PDF, rebuilding it only if the PDF or settings changed.
def ingest(pdf_path=config.PDF_PATH):
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    key = fingerprint(pdf_path)
    try:
        collection = client.get_collection(config.COLLECTION)
        if collection.metadata.get("fingerprint") == key and collection.count() > 0:
            print(f"[ingest] cache hit: {collection.count()} chunks")
            return collection
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass  # collection does not exist yet
    chunks = split_pages(read_pages(pdf_path), os.path.basename(pdf_path))
    print(f"[ingest] embedding {len(chunks)} chunks ...")
    collection = client.create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine", "fingerprint": key})
    collection.add(
        ids=[f"{c['page']}-{c['chunk_index']}" for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[{k: c[k] for k in ("source", "page", "chunk_index", "method")} for c in chunks])
    return collection


if __name__ == "__main__":
    ingest()
