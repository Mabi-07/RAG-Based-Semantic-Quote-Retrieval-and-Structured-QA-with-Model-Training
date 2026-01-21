from rag.rag_pipeline import rag_pipeline
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_groq import ChatGroq

llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

data = [{
    "question": "Give quotes about life",
    "answer": rag_pipeline("quotes about life")["answer"],
    "contexts": rag_pipeline("quotes about life")["quotes"],
    "ground_truth": "Life is about growth, meaning, and resilience."
}]

result = evaluate(
    data,
    metrics=[
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm)
    ]
)

print(result)
