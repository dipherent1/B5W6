# Simple CLI to run the full pipeline end-to-end on local data
from src.preprocess import run_preprocess
from src.chunk_embed_index import run_chunk_embed_index

if __name__ == "__main__":
    print("1) Preprocessing raw dataset...")
    df = run_preprocess()
    print(f"Saved filtered dataset with {len(df)} rows.")
    print("2) Chunking, embedding, and indexing...")
    n, dim = run_chunk_embed_index()
    print(f"Built FAISS index with {n} chunks, dim={dim}. Done.")
