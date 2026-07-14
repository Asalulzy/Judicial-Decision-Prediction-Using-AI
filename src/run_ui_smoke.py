import os
from rag_pipeline import RAGPipeline


def prepare_sample_dataset(base_dir="dataset"):
    os.makedirs(os.path.join(base_dir, "file_putusan", "file_putusan"), exist_ok=True)
    docs = {
        "doc_1.txt": "Kronologi: terdakwa melakukan pencurian. Dakwaan: Pasal 363 KUHP. Tuntutan jaksa 3 tahun.",
        "doc_2.txt": "Kronologi: kasus narkotika. UU: UU Narkotika, Pasal 114. Vonis 5 tahun.",
        "doc_3.txt": "Kronologi: penggelapan, Pasal 372 KUHP. Vonis denda dan 1 tahun penjara."
    }
    folder = os.path.join(base_dir, "file_putusan", "file_putusan")
    for name, txt in docs.items():
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write(txt)
    return base_dir


def main():
    sample_dir = prepare_sample_dataset()
    cfg = {
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "CHUNK_SIZE": 500,
        "CHUNK_OVERLAP": 50,
        "TOP_K": 3,
        "VECTORSTORE_DIR": "vectorstore/faiss_index",
        "LLM_BACKEND": "hf",
        "HF_MODEL": "google/flan-t5-small",
    }

    p = RAGPipeline(cfg)
    print("Building vectorstore from sample dataset... (using small embedding model)")
    p.build_vectorstore(sample_dir)
    print("Vectorstore built. Running a sample retrieval for 'pencurian'...")
    # use embedder + store directly to avoid heavy LLM calls
    qv = p.embedder.embed_texts(["pencurian"])[0]
    res = p.store.search(qv.reshape(1, -1), top_k=3)
    for r in res:
        print("SCORE:", r["score"], "FILE:", r["metadata"]["filename"], "TEXT:", r["metadata"].get("text")[:200])


if __name__ == '__main__':
    main()
