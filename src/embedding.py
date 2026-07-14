from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # returns 2D numpy array (n, d)
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # normalize to unit vectors for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
