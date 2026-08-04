import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from redis_checkpointer import UpstashRedisSaver

load_dotenv()

WINDOW_SIZE = 6  # last 6 messages = 3 turns

_SYSTEM = (
    "You are a document Q&A assistant. Answer from the provided document context "
    "and the conversation history. For follow-up questions, use the conversation "
    "history to understand what the user is referring to. Do not invent facts not "
    "present in the document context or the conversation. "
    "Cite sources like [Source: name] when referencing the document."
)

_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)


def _call_model(state: MessagesState):
    messages = [SystemMessage(content=_SYSTEM)] + state["messages"][-WINDOW_SIZE:]
    response = _llm.invoke(messages)
    return {"messages": [response]}


_checkpointer = UpstashRedisSaver.from_url(
    os.environ.get("REDIS_URL", "redis://localhost:6379"),
    ttl=86400,  # 24 hours in seconds, refreshed on every read/write
)

_builder = StateGraph(MessagesState)
_builder.add_node("model", _call_model)
_builder.add_edge(START, "model")
_builder.add_edge("model", END)
_graph = _builder.compile(checkpointer=_checkpointer)


def callLLM(context: str, question: str, session_id: str) -> str:
    result = _graph.invoke(
        {"messages": [HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")]},
        config={"configurable": {"thread_id": session_id}},
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    print(callLLM("The repo rate is 6.5%.", "What is the repo rate?", session_id="test-session"))
