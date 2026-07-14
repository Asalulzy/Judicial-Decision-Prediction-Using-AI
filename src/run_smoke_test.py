import numpy as np
from preprocessing import clean_text, chunk_text
from vector_database import FaissStore


def test_preprocessing():
    s = "Ini    teks\n\n\n dengan   spasi\tberlebih."
    cleaned = clean_text(s)
    chunks = list(chunk_text(cleaned, chunk_size=20, overlap=5))
    print("CLEANED:", cleaned)
    print("CHUNKS:", chunks)


def test_faiss():
    dim = 8
    store = FaissStore(dim, store_dir="vectorstore/test_index")
    store.create_index()
    # create 10 random vectors and metadatas
    vecs = np.random.rand(10, dim).astype('float32')
    # normalize for IP index
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    metas = [{"filename": f"doc_{i}.txt", "document_id": f"doc_{i}", "text": f"content {i}"} for i in range(10)]
    store.add(vecs, metas)
    store.save()
    # reload
    s2 = FaissStore(dim, store_dir="vectorstore/test_index")
    s2.load()
    q = vecs[0].reshape(1, -1)
    res = s2.search(q, top_k=3)
    print("SEARCH RESULTS:", res)


if __name__ == '__main__':
    print("Running smoke tests...")
    test_preprocessing()
    test_faiss()
    print("Smoke tests done.")
