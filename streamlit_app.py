import os
import streamlit as st
import requests
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Q&A", layout="centered")
st.title("Document Q&A")

# --- Upload section ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if st.session_state.get("uploaded_name") != uploaded_file.name:
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

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Ask a question about the document...")

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        print(st.session_state["uploaded_name"])

        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{API_URL}/question",
                json={"question": question, "doc_name": st.session_state["uploaded_name"]},
            )

        if resp.ok:
            data = resp.json()
            answer = data.get("answer") or data.get("response") or str(data)
        else:
            answer = f"Error: {resp.text}"

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
else:
    st.info("Upload a PDF above to start chatting.")
