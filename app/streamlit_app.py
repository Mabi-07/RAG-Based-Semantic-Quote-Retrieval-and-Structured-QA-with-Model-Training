import sys
import os

# 🔑 Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from rag.rag_pipeline import rag_pipeline

st.set_page_config(page_title="RAG Quote Assistant", layout="wide")

st.title("📜 RAG-Based Semantic Quote Assistant")

query = st.text_input("Ask a question about quotes")

if query:
    result = rag_pipeline(query)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Quotes")
    for q in result["quotes"]:
        st.markdown(f"- *{q}*")

    st.subheader("Authors")
    st.write(result["authors"])

    st.subheader("Tags")
    st.write(result["tags"])
