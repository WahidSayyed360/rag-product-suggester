import streamlit as st
from engine_rag import rag_suggest, retrieve_products

st.set_page_config(page_title="RAG Product Suggester", layout="centered")

st.title("🧾 RAG Product Name Suggester")
query = st.text_input("Enter product keyword (e.g., Paracetamol, Vitamin, Syrup):")
k = st.slider("Retrieve top-K products", 3, 10, 5)
n = st.slider("Number of AI suggestions", 1, 5, 3)

if query:
    st.subheader("📂 Retrieved Products")
    for p in retrieve_products(query, k):
        st.write(f"- {p}")

    st.subheader("✨ AI-Generated Suggestions")
    suggestions = rag_suggest(query, k, n)
    st.write(suggestions)
