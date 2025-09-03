import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("products.csv")
products = df["ProductName"].dropna().tolist()

# -------------------------
# Embedding Model
# -------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(products).tolist()

# -------------------------
# Setup Qdrant
# -------------------------
client = QdrantClient(":memory:")
collection_name = "products"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
)
client.upsert(
    collection_name=collection_name,
    points=[
        {"id": i, "vector": embeddings[i], "payload": {"name": products[i]}}
        for i in range(len(products))
    ]
)

# -------------------------
# Local LLM
# -------------------------
print("Loading local LLM... (first run may take time)")
llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype="auto",
    max_new_tokens=256
)

# -------------------------
# Functions
# -------------------------
def retrieve_products(query: str, k: int = 5):
    query_vector = embedder.encode([query]).tolist()[0]
    results = client.search(collection_name=collection_name, query_vector=query_vector, limit=k)
    return [res.payload["name"] for res in results]

def rag_suggest(query: str, k: int = 5, n: int = 3):
    retrieved = retrieve_products(query, k)
    prompt = f"""
User searched: "{query}"

Related products:
- {chr(10).join(retrieved)}

Suggest {n} realistic product names. 
Include product form and strength if possible.
Output as a numbered list.
"""
    response = llm(prompt)[0]["generated_text"]
    return response
