import streamlit as st
from src.config import APP_TITLE
from src.rag import answer_question

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Ask natural-language questions about complaint narratives across products.")

with st.sidebar:
    st.header("Filters")
    product_filters = st.multiselect(
        "Limit to products",
        ["Credit card","Personal loan","Buy Now, Pay Later","Savings account","Money transfer"],
        default=[]
    )
    k = st.slider("Top-k retrieved chunks", 3, 10, 5, 1)

query = st.text_input("Your question", placeholder="e.g., Why are people unhappy with BNPL?")
ask = st.button("Ask")

if ask and query.strip():
    with st.spinner("Thinking..."):
        out = answer_question(query.strip(), product_filters=product_filters or None, k=k)
    st.subheader("Answer")
    st.write(out["answer"])
    st.subheader("Sources")
    for s in out["sources"]:
        with st.expander(f"Source [{s['rank']}] • {s['product']} • {s['issue']} • {s['date_submitted']}"):
            st.write(s["excerpt"])
