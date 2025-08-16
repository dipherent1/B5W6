from pathlib import Path
import pandas as pd
import numpy as np
import faiss, pickle
from sentence_transformers import SentenceTransformer
from .config import FILTERED_CSV, CHUNKS_CSV, FAISS_INDEX_PATH, META_PATH, EMBEDDING_MODEL
from .config import VECTOR_DIR

def chunk_text(text, chunk_size=400, chunk_overlap=60):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks

def run_chunk_embed_index(chunk_size=400, chunk_overlap=60):
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FILTERED_CSV)
    rows = []
    for _, r in df.iterrows():
        chunks = chunk_text(r['narrative_clean'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, c in enumerate(chunks):
            rows.append({
                "complaint_id": r.get("complaint_id", ""),
                "product": r["product"],
                "issue": r.get("issue", ""),
                "company": r.get("company", ""),
                "date_submitted": r.get("date_submitted", ""),
                "chunk_id": f"{r.get('complaint_id','row')}_{i}",
                "text": c
            })
    cdf = pd.DataFrame(rows)
    cdf.to_csv(CHUNKS_CSV, index=False)

    # Embeddings
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(cdf["text"].tolist(), convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    # Build FAISS index (Inner Product since normalized -> cosine)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    meta = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "rows": cdf.to_dict(orient="records")
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    return len(cdf), dim

if __name__ == "__main__":
    run_chunk_embed_index()
