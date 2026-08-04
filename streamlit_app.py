import os
import uuid
import streamlit as st
import requests
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Q&A", layout="centered")
st.title("Document Q&A")

# Persist session_id in the URL so it survives page refreshes.
# On first visit a new UUID is written to ?session_id=...; on refresh
# the same URL is loaded and the existing ID is reused.
if "session_id" not in st.query_params:
    st.query_params["session_id"] = str(uuid.uuid4())
st.session_state["session_id"] = st.query_params["session_id"]

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Upload section ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if st.session_state.get("uploaded_name") != Path(uploaded_file.name).stem:
        with st.spinner("Processing document..."):
            resp = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
            )
        if resp.ok:
            st.session_state["uploaded_name"] = Path(uploaded_file.name).stem
            st.session_state["messages"] = []
            st.success(f"Ready: **{st.session_state['uploaded_name']}**")
        else:
            st.error(f"Upload failed: {resp.text}")

# --- Chat section ---
if st.session_state.get("uploaded_name"):
    st.divider()

    # render full chat history from session state
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**[{s['index']}]** `{s['doc']}` — {s['text']}...")

    question = st.chat_input("Ask a question about the document...")

    if question:
        # append and show user message
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = requests.post(
                    f"{API_URL}/question",
                    json={
                        "question": question,
                        "doc_name": st.session_state["uploaded_name"],
                        "session_id": st.session_state["session_id"],
                    },
                )

            if resp.ok:
                data = resp.json()
                answer = data.get("answer") or data.get("response") or str(data)
                sources = data.get("sources", [])
                warnings = data.get("warnings", [])
            else:
                answer = f"Error: {resp.text}"
                sources = []
                warnings = []

            st.write(answer)

            if warnings:
                for w in warnings:
                    st.warning(w)

            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"**[{s['index']}]** `{s['doc']}` — {s['text']}...")

        # save assistant message to session state
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

else:
    st.info("Upload a PDF above to start chatting.")
