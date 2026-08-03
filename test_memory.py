"""
Quick test for conversation memory chain without needing Redis.
Swaps RedisChatMessageHistory for in-memory storage.
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

_store = {}  # session_id -> ChatMessageHistory (in-memory)

def get_session_history(session_id: str):
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document Q&A assistant. Answer only from the provided context."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

chain = prompt | llm | StrOutputParser()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

CONTEXT = "The repo rate is 6.5%. It was last changed in April 2024. The inflation target is 4%."
SESSION = "test-session-1"

def ask(question: str) -> str:
    return chain_with_history.invoke(
        {"context": CONTEXT, "question": question},
        config={"configurable": {"session_id": SESSION}},
    )

if __name__ == "__main__":
    print("Q1: What is the repo rate?")
    print("A1:", ask("What is the repo rate?"))
    print()

    print("Q2: When was it last changed?")
    print("A2:", ask("When was it last changed?"))  # 'it' refers to repo rate from history
    print()

    print("Q3: What is the inflation target?")
    print("A3:", ask("What is the inflation target?"))
    print()

    print("--- Stored history ---")
    for msg in _store[SESSION].messages:
        print(f"[{msg.type}]: {msg.content[:80]}")
