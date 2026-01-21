from rag.retriever import retrieve
from rag.generator import generate_answer


def rag_pipeline(query, top_k=5):
    docs = retrieve(query, top_k)
    answer = generate_answer(query, docs)

    return {
        "answer": answer,
        "quotes": [d["quote"] for d in docs],
        "authors": list(set(d["author"] for d in docs)),
        "tags": list(set(tag for d in docs for tag in d["tags"]))
    }
