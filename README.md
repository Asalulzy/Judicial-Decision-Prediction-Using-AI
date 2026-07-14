# Judicial Decision Prediction Using AI — with RAG Legal Assistant

This repository combines a judicial decision prediction project with a Retrieval-Augmented Generation (RAG) assistant for Indonesian court decisions. It includes code for:

- Predictive modeling on historical court decisions (XGBoost + IndoBERT embeddings).
- A RAG pipeline: document ingestion, chunking, embeddings, FAISS vector store, retriever, and LLM-backed answer + structured extraction.

## Overview

The predictive component estimates prison sentence duration from Indonesian criminal court texts using NLP and supervised learning. The RAG assistant complements the predictor by enabling semantic search over case documents and generating explainable legal answers and structured JSON extractions (case type, laws, articles, demands, decisions, sentence duration).

## Quick Start

1. Create a Python 3.10+ virtual env and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

2. Place your dataset ZIP or folder and set the path in `.env` (see `.env.example`) or enter it in the UI.

3. Run the Streamlit app:

```bash
streamlit run src/app.py
```

## Notes

- Default embedding model: `intfloat/multilingual-e5-base` (configurable via `.env`). For fast tests use `all-MiniLM-L6-v2`.
- LLM backend: HuggingFace models or Ollama (configure via `.env`).
- Vector store is saved under `vectorstore/faiss_index/` to avoid re-embedding.
- Do not commit raw datasets or `.env` to Git — see `.gitignore`.

## Project Layout

- `dataset/` — place or extract `dataset.zip` here
- `vectorstore/faiss_index/` — FAISS index files and metadata
- `src/` — code modules (data loader, preprocessing, embedding, vector DB, retriever, LLM wrapper, extractor, RAG pipeline, Streamlit app)

## Next Steps & Recommendations

- To push this code into an existing repo, we merged and resolved README conflicts. If you prefer keeping project histories separate, consider reviewing commit history before pushing.
- Build the full vectorstore from your real dataset (may take time and disk space).
- Improve the extractor prompts and LLM backend for more accurate structured outputs.
- Add LICENSE and CI if you want continuous checks.

See `PROJECT_SUMMARY.txt` for a compact summary of implemented features and remaining tasks.

---

