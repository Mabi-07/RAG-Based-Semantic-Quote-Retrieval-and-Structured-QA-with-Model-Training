import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("model/embedding_model")

# Load FAISS index
index = faiss.read_index("vectorstore/index.faiss")

# Load metadata
with open("vectorstore/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


def retrieve(query, top_k=5):
    """
    Retrieve top_k most similar quotes for a given query
    """
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    return [metadata[i] for i in indices[0]]
