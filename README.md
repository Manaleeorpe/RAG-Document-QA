A full-stack Retrieval-Augmented Generation (RAG) system for uploading PDF documents and asking questions over them. Built with a FastAPI backend, Streamlit chat UI, hybrid vector + keyword retrieval, query routing, safety guardrails, persistent conversation memory, and LLM-based answer evaluation.

## Live Demo

**Deployed on Railway:** https://rag-document-qa-production-19ff.up.railway.app

<video src="docs/demo.mp4" controls width="100%"></video>

---

## Features

- Upload PDF documents and query them through a conversational chat interface.
- **Hybrid retrieval** — Pinecone dense search + BM25 keyword search, fused with Reciprocal Rank Fusion (RRF) and re-ranked by a cross-encoder.
- **Query routing** — classifies each question as factual, analytical, or ambiguous. Analytical questions are decomposed into sub-questions and retrieved separately before merging results.
- **Input guardrails** — prompt-injection detection (hard block) and PII redaction applied before any retrieval happens.
- **Output guardrails** — toxicity filter on generated answers before they reach the user.
- **Persistent conversation memory** — per-session history stored in Upstash Redis via a custom LangGraph checkpointer, surviving page refreshes and server restarts. Sessions auto-expire after 24 hours of inactivity.
- **LLM evaluator** — scores every answer on relevance, completeness, and groundedness (0–1 scale).
- **Structured responses** — every answer includes source citations, evaluation scores, and safety warnings where applicable.
- **Health check** — `/health` endpoint reports Pinecone and LLM connectivity status.

---

## Architecture Diagram

<img width="679" height="710" alt="Architecture diagram" src="https://github.com/user-attachments/assets/2cb3ae3e-be69-4e1b-aa54-9781a2b58da2" />

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| LLM | DeepSeek (`deepseek-chat`) via OpenAI-compatible API |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Vector store | Pinecone (serverless, AWS us-east-1) |
| Keyword search | BM25 (`rank_bm25`) |
| Re-ranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2` via sentence-transformers) |
| Memory | LangGraph + custom `UpstashRedisSaver` |
| Session persistence | Upstash Redis (basic commands — `SET`/`GET`/`RPUSH`) |
| Containerisation | Docker |
| Deployment | Railway |

---

## Project Structure

```text
raf-document-qa/
├── app/
│   ├── main.py               # FastAPI app, routes, request-logging middleware
│   ├── pipeline.py           # End-to-end RAG orchestrator
│   ├── calLLM.py             # LangGraph conversation graph + Redis checkpointer wiring
│   ├── redis_checkpointer.py # Custom UpstashRedisSaver (Upstash-compatible LangGraph checkpointer)
│   ├── vector_store.py       # Pinecone wrapper via LangChain
│   ├── retrievers.py         # Hybrid BM25 + Pinecone retriever with cross-encoder reranker
│   ├── embeddings.py         # OpenAI embedding model wrapper
│   ├── chunker.py            # PDF chunking and metadata enrichment
│   ├── document_processor.py # PDF loading and text extraction
│   ├── query_router.py       # Topic guardrail, query classification, sub-question decomposition
│   ├── guardrails.py         # Input (injection/PII) + output (toxicity) guardrails
│   ├── evaluator.py          # LLM evaluator (relevance, completeness, groundedness)
│   └── logger.py             # Structured JSON request logging
├── streamlit_app.py          # Streamlit chat UI
├── Dockerfile
├── start.sh                  # Starts FastAPI (port 8000) then Streamlit (PORT env var)
└── requirements.txt
```

---

## Design Decisions

Decisions made while building and deploying this app on Railway — covering trade-offs between local simplicity and production reliability.

### 1. OpenAI Embeddings over a Local Model

**Before:** `all-MiniLM-L6-v2` loaded locally via sentence-transformers.

**After:** OpenAI `text-embedding-3-small` via API.

The local model loaded into RAM on startup, which increased memory pressure on Railway's constrained containers. Scaling up pods would mean each one loading the model independently. Switching to the OpenAI API offloaded that entirely — no RAM cost, no cold-start penalty, and higher-quality 1536-dimensional embeddings at $0.02 per million tokens.

---

### 2. Pinecone over ChromaDB

**Before:** ChromaDB persisted to a local `chroma_db/` folder.

**After:** Pinecone serverless index (free tier, AWS us-east-1).

Railway's filesystem is ephemeral — every redeploy wipes the container's disk. Any ChromaDB data would be lost on every deploy, meaning users would have to re-upload their documents after every release. Pinecone is fully managed and external to the pod, so indexed documents survive deploys indefinitely.

---

### 3. Known Edge Case: Stale Vectors on Re-upload

When a document is re-uploaded, chunks are indexed using IDs like `myfile_0`, `myfile_1`, ..., `myfile_N`. If the new version of the file is shorter than the original (fewer chunks), the old extra chunk IDs — `myfile_50`, `myfile_51`, etc. — remain as stale vectors in Pinecone because there is no delete step before re-indexing.

This is a known limitation. The app currently has no delete endpoint, so the stale vectors will persist until the Pinecone index is manually cleared. For most use cases (uploading once, or replacing with a same-length document) this is not an issue, but it is worth being aware of.

---

### 4. Persistent Conversation Context via Redis

**Before:** `InMemorySaver` from LangGraph stored all conversation history in process RAM.

**After:** Custom `UpstashRedisSaver` persists LangGraph checkpoints in Upstash Redis.

`InMemorySaver` silently failed in two ways — it never raised an error, it just lost context:

1. **Server restarts wiped everything.** Railway redeploys, crashes, and scale-to-zero events all kill the process, taking every conversation's history with it.
2. **Page refreshes generated a new session ID.** The `session_id` was stored in `st.session_state`, which Streamlit resets on refresh — so every refresh looked like a brand new user to the backend.

LangGraph treats "no checkpoint found for this thread" as a valid fresh start, so there was never an error — the LLM simply answered every question as if it were the first one.

**Fix — two parts:**

- `session_id` is now written into the browser URL via `st.query_params` (`?session_id=abc-123`). The URL persists across refreshes, so the same ID reaches the backend every time. A new tab or new browser gets a fresh UUID and therefore a fresh session.
- Checkpoints are stored in Upstash Redis (external to the pod) using a custom `UpstashRedisSaver`. Sessions expire automatically after 24 hours of inactivity, with the TTL refreshed on every question.

---

### 5. Upstash Redis vs Standard Redis

| | Upstash Redis | Standard Redis (Redis Stack) |
|---|---|---|
| Hosting | Serverless, fully managed | Self-hosted or managed (e.g. Redis Cloud) |
| Basic commands (`GET`, `SET`, `EXPIRE`) | Yes | Yes |
| Redis Search (`FT.*` commands) | **No** | Yes |
| Redis JSON module | **No** | Yes |
| Cost model | Pay per request | Pay per instance |

`langgraph-checkpoint-redis` (the official LangGraph Redis checkpointer) uses `redisvl` internally, which requires Redis Search (`FT.CREATE`, `FT._LIST`, `FT.SEARCH`). Upstash does not support these commands, so that library fails immediately on startup.

The custom `UpstashRedisSaver` in this project uses only basic Redis commands (`SET`, `GET`, `RPUSH`, `LRANGE`, `EXPIRE`, `PIPELINE`), making it fully compatible with Upstash while implementing the complete LangGraph `BaseCheckpointSaver` interface.

---

### 6. Session ID Persistence in the URL

**The problem:** The app had no persistent conversation context. Every page refresh or server restart started a completely fresh conversation with no memory of previous questions — silently, with no errors.

**Root cause:** Two independent failures stacking on top of each other:

1. **`session_id` died on refresh.** It was stored only in `st.session_state` (Streamlit's server-side in-memory store), which is wiped on every page refresh. Every refresh generated a new UUID, so the backend never saw the same session twice.
2. **Checkpoints died on restart.** `InMemorySaver` kept all conversation history in process RAM. Any server restart — Railway redeploy, crash, scale-to-zero — instantly wiped every session. Neither failure produced an error; LangGraph simply treated the missing checkpoint as a new conversation.

**Resolution:**

1. `session_id` is written to `st.query_params`, embedding it in the browser URL (`?session_id=abc-123`). The URL survives refreshes, so the same ID is always sent to the backend.
2. Checkpoints are stored in Upstash Redis via `UpstashRedisSaver`, persisting conversation state outside the pod and across restarts.

---

## Demo

### Upload a document

![Upload screen](docs/screenshot-upload.png)

### Ask a question

![Chat screen](docs/screenshot-chat.png)

---

## API Reference

### POST `/upload`

Upload a PDF for processing. The document is chunked and indexed in Pinecone.

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@sample.pdf"
```

**Response (200)**
```json
{
  "filename": "sample.pdf",
  "result": {
    "document": "sample.pdf",
    "chunks": 42,
    "status": "processed"
  }
}
```

---

### POST `/question`

Ask a question about an uploaded document.

```bash
curl -X POST "http://localhost:8000/question" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "doc_name": "sample", "session_id": "abc123"}'
```

**Response (200)**
```json
{
  "question": "What are the key findings?",
  "answer": "The key findings are ...",
  "category": "analytical",
  "sources": [
    {"index": 1, "doc": "sample", "chunk_id": "sample_4", "text": "...excerpt..."}
  ],
  "sub_questions": ["What findings are mentioned in section 1?", "..."],
  "evaluation": {
    "relevance": 0.95,
    "completeness": 0.88,
    "groundedness": 0.91,
    "overall": 0.913
  },
  "warnings": []
}
```

**Blocked response (injection / off-topic)**
```json
{
  "question": "...",
  "answer": "Your request was blocked by the input safety check.",
  "blocked": true,
  "reasons": ["Possible prompt injection detected."],
  "sources": []
}
```

---

### GET `/health`

Returns Pinecone and LLM connectivity status.

```bash
curl "http://localhost:8000/health"
```

**Response (200 healthy / 503 degraded)**
```json
{
  "status": "healthy",
  "checks": {
    "db": {"status": "connected", "total_chunks": 312},
    "llm": {"status": "reachable"}
  }
}
```

---

## Run Locally

### Prerequisites

- Python 3.11+
- A [Pinecone](https://www.pinecone.io/) account and API key
- An [OpenAI](https://platform.openai.com/) API key (for embeddings)
- A [DeepSeek](https://platform.deepseek.com/) API key (for the LLM)
- An [Upstash Redis](https://upstash.com/) database and its connection URL

### Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=sk-...
   DEEPSEEK_API_KEY=sk-...
   PINECONE_API_KEY=pcsk_...
   REDIS_URL=rediss://default:...@...upstash.io:6379
   ```

3. Start both services:
   ```bash
   # Together (production-style)
   chmod +x start.sh && ./start.sh

   # Or separately (development)
   PYTHONPATH=app uvicorn app.main:app --reload --port 8000
   streamlit run streamlit_app.py
   ```

4. Open:
   - Streamlit UI: `http://localhost:8501`
   - API docs: `http://localhost:8000/docs`

### Docker

```bash
docker build -t rag-document-qa .
docker run -p 8000:8000 -p 8501:8501 \
  -e OPENAI_API_KEY=sk-... \
  -e DEEPSEEK_API_KEY=sk-... \
  -e PINECONE_API_KEY=pcsk_... \
  -e REDIS_URL=rediss://... \
  rag-document-qa
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for `text-embedding-3-small` |
| `DEEPSEEK_API_KEY` | Yes | DeepSeek API key for `deepseek-chat` LLM |
| `PINECONE_API_KEY` | Yes | Pinecone API key for vector storage |
| `REDIS_URL` | Yes | Upstash Redis connection URL for session persistence |
| `PORT` | No | Port for Streamlit (Railway sets this automatically) |
| `FASTAPI_PORT` | No | Port for FastAPI backend (default: `8000`) |
| `API_URL` | No | URL Streamlit uses to reach the backend (default: `http://localhost:8000`) |
