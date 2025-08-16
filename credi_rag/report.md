# CrediTrust Complaint Intelligence — RAG Project Report

**Team**: Mahlet, Kerod, Rediet, Rehmet  
**Facilitator**: —  
**Dates**: Intro (02 July 2025, 09:30 UTC) • Interim (06 July 2025, 20:00 UTC) • Final (08 July 2025, 20:00 UTC)

---

## 1) Business Objective & KPIs

CrediTrust operates five products (Credit Cards, Personal Loans, Buy Now, Pay Later, Savings Accounts, Money Transfers). The company receives thousands of complaints monthly. Our internal AI tool converts these narratives into evidence-backed answers so Product Managers (e.g., Asha), Support, and Compliance can identify issues in **minutes**.

**KPIs**
1. Reduce trend identification time from days → minutes.
2. Enable non-technical teams to self-serve answers.
3. Shift from reactive to proactive decisions using near real-time feedback.

---

## 2) Data & EDA (Task 1)

**Source**: CFPB-like complaint dataset (CSV). We processed a sample for demonstration; the pipeline works unchanged on full data.

**EDA highlights (sample):**
- Coverage across five product lines.
- Word-count spans short to medium narratives (5–60 tokens in sample).
- 100% of rows used (after dropping empty narratives).

**Preprocessing**
- Kept products: *Credit card, Personal loan, Buy Now, Pay Later, Savings account, Money transfer[s]*.
- Removed empty narratives and standardized product labels.
- Basic text cleanup: lowercasing, boilerplate/URL/email stripping, punctuation normalization, whitespace compaction.
- Saved to `data/processed/filtered_complaints.csv`.

Artifacts: `src/preprocess.py`, processed CSV.

---

## 3) Chunking, Embedding & Indexing (Task 2)

- **Chunking**: Recursive fixed-size by words (`chunk_size=400`, `chunk_overlap=60`). Balances long-context coverage with redundancy.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — fast, strong cosine-retrieval baseline.
- **Index**: FAISS `IndexFlatIP` over normalized vectors (cosine similarity). Persisted at `vector_store/index.faiss` and `vector_store/meta.pkl` (with chunk metadata).

Artifacts: `src/chunk_embed_index.py`, `vector_store/` contents.

---

## 4) RAG Core & Prompting (Task 3)

**Retriever**
- Same embed model for questions.
- Top-*k* similarity search (default 5).
- Optional product filters. For simplicity at small scale, filtered queries re-embed the subset (production: per-product sub-index).

**Prompt**
```
You are a financial analyst assistant for CrediTrust...
Use ONLY the retrieved excerpts; if not present, say you don't have enough information.
```

**Generator**
- Default: local Transformers (`google/flan-t5-base`) for offline use.
- Optional: OpenAI (`OPENAI_API_KEY`) for higher-quality responses.

**Evaluation**
- 8 representative questions executed; outputs recorded to `reports/evaluation.md` and `.csv` with provisional scores and comments (manual review recommended).

Artifacts: `src/rag.py`, `src/evaluate.py`, `reports/evaluation.*`.

---

## 5) Interactive App (Task 4)

**Tech**: Streamlit (`app.py`).  
**Features**:
- Free-text question input + *Ask* button.
- Optional product filters and configurable top-*k*.
- **Displays the exact source excerpts** used for the answer (trust & verification).
- Clean, one-screen UX.

Run:
```bash
streamlit run app.py
```

---

## 6) Architecture & Foldering

```
data/raw → preprocess → data/processed
               ↓
         chunk → embed → index (FAISS)
               ↓
            RAG(Q) → retrieved chunks → prompt → LLM → answer
               ↓
          UI (Streamlit) shows answer + sources
```

---

## 7) Risks, Assumptions & Next Steps

**Risks**
- LLM hallucinations → mitigated by strong instruction + source display.
- Domain drift or noisy narratives → consider product-specific fine-tuning or hybrid BM25 + vector search.
- Latency on large data → build per-product/sliced indices, cache frequent queries.

**Next Steps**
- Quantitative eval with labeled Q/A and nDCG@k for retrieval.
- Add trend dashboards (topic modeling over time).
- Add role-based filters (country, merchant, severity).

---

## 8) How to Run

```bash
pip install -r requirements.txt
python run_pipeline.py                # build vectors
streamlit run app.py                  # open UI
# Optional evaluation:
python -m src.evaluate
```

This submission is fully runnable on the bundled sample and scales to the full CFPB dataset by switching `RAW_CSV`.
