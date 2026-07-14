import os
import faiss
import pickle
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FaissStore:
    def __init__(self, dim: int, store_dir: str = "vectorstore/faiss_index"):
        self.dim = dim
        self.store_dir = Path(store_dir)
        self.index_path = self.store_dir / "index.faiss"
        self.meta_path = self.store_dir / "metadatas.pkl"
        self.index = None
        self.metadatas = []
        os.makedirs(self.store_dir, exist_ok=True)

    def create_index(self):
        # Using inner product (cosine after normalization)
        self.index = faiss.IndexFlatIP(self.dim)

    def add(self, vectors: np.ndarray, metadatas: list[dict]):
        if self.index is None:
            self.create_index()
        self.index.add(vectors.astype('float32'))
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        if self.index is None or self.index.ntotal == 0:
            return []
        D, I = self.index.search(query_vector.astype('float32'), top_k)
        results = []
        for scores, idxs in zip(D, I):
            for score, idx in zip(scores, idxs):
                if idx < 0:
                    continue
                meta = self.metadatas[int(idx)]
                results.append({"score": float(score), "metadata": meta, "index": int(idx)})
        return results

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)
        logger.info("Saved FAISS index and metadata to %s", self.store_dir)

    def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.meta_path.exists():
            with open(self.meta_path, "rb") as f:
                self.metadatas = pickle.load(f)
