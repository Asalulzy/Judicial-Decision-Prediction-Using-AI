from typing import List
import numpy as np


def retrieve_documents(store, query_vector: np.ndarray, top_k: int = 5):
    """Return top_k results from FaissStore with aggregated context text."""
    hits = store.search(query_vector.reshape(1, -1), top_k=top_k)
    # build contexts list
    contexts = []
    for h in hits:
        meta = h["metadata"]
        contexts.append({"score": h["score"], "text": meta.get("text", ""), "metadata": meta})
    return contexts
