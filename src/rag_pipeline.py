import os
import logging
from pathlib import Path
from data_loader import ensure_extracted, load_txt_documents
from preprocessing import clean_text, chunk_text
from embedding import Embedder
from vector_database import FaissStore
from llm import generate

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = config.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        self.chunk_size = int(config.get("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(config.get("CHUNK_OVERLAP", 200))
        self.top_k = int(config.get("TOP_K", 5))
        self.vector_dir = config.get("VECTORSTORE_DIR", "vectorstore/faiss_index")
        self.ollama_url = config.get("OLLAMA_URL")
        self.llm_backend = config.get("LLM_BACKEND", "hf")
        self.hf_model = config.get("HF_MODEL", "google/flan-t5-small")

        self.embedder = None
        self.store = None

    def build_vectorstore(self, dataset_path: str):
        extracted = ensure_extracted(dataset_path, extract_to="dataset")
        docs = list(load_txt_documents(extracted))
        texts = []
        metadatas = []
        # chunk docs
        for text, meta in docs:
            text = clean_text(text)
            for i, chunk in enumerate(chunk_text(text, self.chunk_size, self.chunk_overlap)):
                texts.append(chunk)
                m = meta.copy()
                m.update({"chunk_id": i, "text": chunk})
                metadatas.append(m)

        # create embeddings
        self.embedder = Embedder(self.embedding_model)
        vectors = self.embedder.embed_texts(texts)

        # build FAISS
        dim = vectors.shape[1]
        self.store = FaissStore(dim, store_dir=self.vector_dir)
        self.store.add(vectors, metadatas)
        self.store.save()
        logger.info("Built vectorstore with %d chunks", len(metadatas))

    def load_vectorstore(self, dim: int = 768):
        # initialize embedder and store
        self.embedder = Embedder(self.embedding_model)
        self.store = FaissStore(dim, store_dir=self.vector_dir)
        self.store.load()

    def query(self, question: str):
        if self.embedder is None or self.store is None:
            raise RuntimeError("Vectorstore not loaded. Call build_vectorstore or load_vectorstore first.")
        qv = self.embedder.embed_texts([question])[0]
        results = self.store.search(qv.reshape(1, -1), top_k=self.top_k)
        # collect context
        contexts = []
        refs = []
        for r in results:
            md = r["metadata"]
            contexts.append(md.get("text", ""))
            refs.append(md.get("filename"))

        prompt = "Berikan jawaban hukum singkat berdasarkan konteks berikut:\n\n" + "\n---\n".join(contexts) + f"\n\nPertanyaan: {question}\n"
        answer = generate(prompt, backend=self.llm_backend, ollama_url=self.ollama_url, hf_model=self.hf_model)
        return {"answer": answer, "references": list(dict.fromkeys(refs))}
