# CrediTrust Complaint Intelligence (RAG)

An internal RAG tool to turn raw complaint narratives into evidence-backed answers for Product, Support, and Compliance.

## Project Structure
```
.
├── app.py                     # Streamlit interface
├── run_pipeline.py            # CLI to run preprocess + index
├── data/
│   ├── raw/complaints_sample.csv   # tiny sample data (replace with full CFPB CSV)
│   └── processed/                  # filtered + chunks
├── reports/                   # evaluation outputs
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── preprocess.py
│   ├── chunk_embed_index.py
│   ├── rag.py
│   └── evaluate.py
└── vector_store/             # FAISS index + metadata
```

## Quickstart

1. **Create environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Use your own dataset (recommended)**  
   Place the CFPB-style CSV at `data/raw/complaints.csv` and run with:
```bash
export RAW_CSV=data/raw/complaints.csv
python run_pipeline.py
```

If you don't have the full dataset yet, the project ships with a small sample at `data/raw/complaints_sample.csv`.  
To run on the sample just do:
```bash
python run_pipeline.py
```

3. **Start the UI**
```bash
streamlit run app.py
```

## Configuration

Environment variables (optional):

- `RAW_CSV` – path to your raw CSV.
- `EMBEDDING_MODEL` – default `sentence-transformers/all-MiniLM-L6-v2`.
- `TOP_K` – top retrieved chunks (default 5).
- `LLM_PROVIDER` – `openai` or `transformers` (default `transformers`).
- `TRANSFORMERS_MODEL` – default `google/flan-t5-base`.
- `OPENAI_API_KEY` – required if using `LLM_PROVIDER=openai`.
- `OPENAI_MODEL` – default `gpt-4o-mini`.

## Expected Columns
Minimum columns expected in the raw CSV (headers may vary – script tries to adapt):
- `complaint_id`
- `product` (one of: Credit card, Personal loan, Buy Now, Pay Later, Savings account, Money transfer[s])
- `issue`
- `company`
- `date_submitted`
- `narrative` (or `Consumer complaint narrative` etc.)

## Notes
- This repo uses FAISS (inner-product over normalized vectors) for fast semantic search.
- The evaluation step produces `reports/evaluation.md` and `.csv`. Review manually and adjust scoring.
- For large datasets, consider swapping brute-force re-embed during filtered search with per-product sub-indexes.
