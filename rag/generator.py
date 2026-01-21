from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

def generate_answer(query, contexts):
    context_text = "\n".join(
        [f"- \"{c['quote']}\" — {c['author']}" for c in contexts]
    )

    prompt = f"""
You are a quote assistant.

Question: {query}

Context:
{context_text}

Return a structured answer:
- Summary
- Quotes
- Authors
- Themes
"""

    return llm.invoke(prompt).content
