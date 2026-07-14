import os
import streamlit as st
import logging
from dotenv import load_dotenv
from pathlib import Path
import sys
import os

load_dotenv()

# Ensure src directory is importable when running `streamlit run src/app.py`
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rag_pipeline import RAGPipeline
from extractor import extract_structured

logging.basicConfig(level=logging.INFO)

ST = st

def load_config():
    return {
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
        "VECTORSTORE_DIR": os.getenv("VECTORSTORE_DIR", "vectorstore/faiss_index"),
        "CHUNK_SIZE": os.getenv("CHUNK_SIZE", "1000"),
        "CHUNK_OVERLAP": os.getenv("CHUNK_OVERLAP", "200"),
        "TOP_K": os.getenv("TOP_K", "5"),
        "OLLAMA_URL": os.getenv("OLLAMA_URL"),
        "LLM_BACKEND": os.getenv("LLM_BACKEND", "hf"),
        "HF_MODEL": os.getenv("HF_MODEL", "google/flan-t5-small"),
    }


def main():
    st.title("Legal AI Assistant Indonesia — RAG")

    cfg = load_config()
    dataset_path = st.text_input("Dataset path (zip or folder)", value=os.getenv("DATASET_PATH", "dataset.zip"))

    pipeline = RAGPipeline(cfg)

    if st.button("Build vectorstore (may take long)"):
        with st.spinner("Building vectorstore — embedding and indexing..."):
            pipeline.build_vectorstore(dataset_path)
        st.success("Vectorstore built. You can now ask questions.")

    q = st.text_input("Pertanyaan hukum", value="Kasus pencurian dengan pemberatan biasanya dikenakan pasal apa?")
    if st.button("Search") and q.strip():
        with st.spinner("Running retrieval and generation..."):
            # attempt to load store if not built in this session
            try:
                if pipeline.store is None:
                    # try load with common dim (will be overwritten if different at runtime)
                    pipeline.load_vectorstore(dim=768)
                out = pipeline.query(q)
                st.subheader("Jawaban AI")
                st.write(out.get("answer"))
                st.subheader("Referensi")
                for r in out.get("references", [])[:20]:
                    st.write(r)
                st.subheader("Struktur Ekstraksi (contoh)")
                structured = extract_structured(out.get("answer", ""), backend=cfg.get("LLM_BACKEND"), ollama_url=cfg.get("OLLAMA_URL"), hf_model=cfg.get("HF_MODEL"))
                st.json(structured)
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
