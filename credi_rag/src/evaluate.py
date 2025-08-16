from pathlib import Path
import pandas as pd
from .rag import answer_question
from .config import REPORTS_DIR

QUESTIONS = [
    "Why are people unhappy with BNPL?",
    "What are the most common issues with credit cards?",
    "Are there complaints about locked savings accounts?",
    "Do users face delayed money transfers?",
    "Any problems with personal loan interest rates?",
    "What refund issues appear in BNPL?",
    "Are there repeated disputes with merchants for BNPL?",
    "Which products seem to have access problems?"
]

def run_eval():
    rows = []
    for q in QUESTIONS:
        # naive product inference for demo
        prod_filters = None
        if "bnpl" in q.lower():
            prod_filters = ["Buy Now, Pay Later"]
        elif "credit card" in q.lower():
            prod_filters = ["Credit card"]
        elif "savings" in q.lower():
            prod_filters = ["Savings account"]
        elif "transfer" in q.lower():
            prod_filters = ["Money transfer"]
        elif "personal loan" in q.lower():
            prod_filters = ["Personal loan"]

        out = answer_question(q, product_filters=prod_filters, k=5)
        # placeholder scoring (manual review recommended)
        rows.append({
            "Question": q,
            "Generated Answer": out["answer"],
            "Retrieved Sources (top-2)": "\n".join([f"{s['product']} | {s['issue']} | {s['excerpt'][:120]}..." for s in out["sources"]]),
            "Quality Score (1-5)": 3,
            "Comments/Analysis": "Auto-generated score; replace with human review."
        })
    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md = df.to_markdown(index=False)
    (REPORTS_DIR / "evaluation.md").write_text(md, encoding="utf-8")
    df.to_csv(REPORTS_DIR / "evaluation.csv", index=False, encoding="utf-8")
    return md

if __name__ == "__main__":
    print(run_eval())
