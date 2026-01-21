import faiss
import numpy as np
from data.quotes_dataset import load_quotes
from sentence_transformers import SentenceTransformer
import pickle
import os

df = load_quotes()
model = SentenceTransformer("model/embedding_model")

embeddings = model.encode(df["quote"].tolist()).astype("float32")
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "vectorstore/index.faiss")

with open("vectorstore/metadata.pkl", "wb") as f:
    pickle.dump(df.to_dict(orient="records"), f)
