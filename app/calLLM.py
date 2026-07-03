import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

_SYSTEM = (
    "You are a document Q&A assistant. Answer only from the provided context. "
    "Cite sources like [Source: name]."
)


def callLLM(context: str, question: str) -> str:
    """Single-shot Q&A over a pre-formatted context, via LangChain ChatOpenAI (DeepSeek)."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    chain = prompt | llm
    return chain.invoke({"context": context, "question": question}).content


if __name__ == "__main__":
    print(callLLM("The repo rate is 6.5%.", "What is the repo rate?"))
