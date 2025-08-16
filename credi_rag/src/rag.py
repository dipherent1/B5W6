import os, pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .config import FAISS_INDEX_PATH, META_PATH, TOP_K, EMBEDDING_MODEL, LLM_PROVIDER, TRANSFORMERS_MODEL
from .config import MAX_INPUT_TOKENS

PROMPT_TEMPLATE = (
    "You are a financial analyst assistant for CrediTrust. "
    "Your task is to answer questions about customer complaints. "
    "Use ONLY the following retrieved complaint excerpts to formulate your answer. "
    "If the context doesn't contain the answer, say you don't have enough information.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

def _load_index_and_meta():
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def _load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

def _embed_query(embedder, q: str):
    v = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    return v.astype("float32")

def retrieve(question: str, product_filters: List[str] = None, k: int = TOP_K):
    index, meta = _load_index_and_meta()
    rows = meta["rows"]
    df = pd.DataFrame(rows)

    if product_filters:
        df_mask = df["product"].isin(product_filters)
        if not df_mask.any():
            return [], []
        df = df[df_mask].reset_index(drop=True)

        # If filtered, we need to search across only those entries.
        # Simplest approach: we will map filtered indices to original index positions.
        # For inner-product search, we must gather the corresponding vectors. To keep it simple for this project,
        # we will do a brute-force re-embed over filtered texts (small-scale acceptable). For large-scale, slice the original index.
        embedder = _load_embedder()
        vecs = embedder.encode(df["text"].tolist(), convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        qvec = _embed_query(embedder, question)
        sims = (vecs @ qvec.T).reshape(-1)
        top_idx = np.argsort(-sims)[:k]
        selected = df.iloc[top_idx]
        return selected.to_dict(orient="records"), sims[top_idx].tolist()

    # No product filter: use prebuilt FAISS over all texts
    embedder = _load_embedder()
    qvec = _embed_query(embedder, question)
    D, I = index.search(qvec, k)
    I = I[0]
    D = D[0]
    df_all = pd.DataFrame(rows)
    hits = [df_all.iloc[i].to_dict() for i in I]
    scores = [float(s) for s in D]
    return hits, scores

# --- LLM backends ---
def _generate_transformers(prompt: str) -> str:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    pipe = pipeline("text2text-generation", model=TRANSFORMERS_MODEL)
    # basic length control
    out = pipe(prompt, max_new_tokens=256, do_sample=False)
    return out[0]["generated_text"]

def _generate_openai(prompt: str) -> str:
    # Requires OPENAI_API_KEY and openai package
    from openai import OpenAI
    client = OpenAI()
    chat = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role":"system","content":"You are a concise, evidence-backed financial analyst assistant."},
            {"role":"user","content":prompt}
        ],
        temperature=0.2
    )
    return chat.choices[0].message.content

def answer_question(question: str, product_filters: List[str] = None, k: int = TOP_K) -> Dict:
    hits, scores = retrieve(question, product_filters=product_filters, k=k)
    context_parts = []
    for i, h in enumerate(hits):
        context_parts.append(f"[{i+1}] product={h.get('product','')}, issue={h.get('issue','')}, date={h.get('date_submitted','')}: {h.get('text','')}".strip())
    context = "\n".join(context_parts)[:MAX_INPUT_TOKENS*4]  # rough char limit
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    if LLM_PROVIDER == "openai" and os.getenv("OPENAI_API_KEY"):
        gen = _generate_openai(prompt)
    else:
        gen = _generate_transformers(prompt)

    # Package sources (top 2 for display convenience)
    sources = []
    for i, h in enumerate(hits[:2]):
        sources.append({
            "rank": i+1,
            "product": h.get("product",""),
            "issue": h.get("issue",""),
            "date_submitted": h.get("date_submitted",""),
            "excerpt": h.get("text","")[:500]
        })

    return {
        "answer": gen.strip(),
        "sources": sources,
        "k": k
    }

if __name__ == "__main__":
    print(answer_question("Why are people unhappy with BNPL?", product_filters=["Buy Now, Pay Later"]))
