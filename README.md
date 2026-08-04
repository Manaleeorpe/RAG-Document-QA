A full-stack Retrieval-Augmented Generation (RAG) system for uploading PDF documents and asking questions over them. Combines a FastAPI backend with a Streamlit chat UI, hybrid vector + keyword retrieval, query routing, safety guardrails, context memory, and LLM-based answer evaluation.

## Live Demo

**Deployed on Railway:** https://rag-document-qa-production-19ff.up.railway.app

## Features

- Upload PDF documents and query them via a chat interface.
- **Hybrid retrieval** — Pinecone dense search + BM25 keyword search fused with Reciprocal Rank Fusion (RRF), re-ranked by a cross-encoder.
- **Query routing** — classifies each question as factual, analytical, or ambiguous. Analytical questions are decomposed into sub-questions for multi-search.
- **Input guardrails** — prompt-injection detection (hard block) and PII redaction before retrieval.
- **Output guardrails** — toxicity filter on generated answers.
- **Context memory** — per-session conversation history via LangGraph (last 3 turns retained).
- **LLM evaluator** — scores every answer on relevance, completeness, and groundedness (0–1).
- **Structured responses** — answers include source citations, evaluation scores, and warnings.
- **Health check** — reports Pinecone and LLM connectivity status.

## Architecture Diagram

<img width="679" height="710" alt="image" src="https://github.com/user-attachments/assets/2cb3ae3e-be69-4e1b-aa54-9781a2b58da2" />

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| LLM | DeepSeek (`deepseek-chat`) via OpenAI-compatible API |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Vector store | Pinecone (serverless, AWS us-east-1) |
| Keyword search | BM25 (rank_bm25) |
| Re-ranking | Cross-encoder via sentence-transformers |
| Memory | LangGraph + InMemorySaver |
| Containerisation | Docker |
| Deployment | Railway |

## Project Structure

```text
RAG-Document-QA/
├── app/
│   ├── main.py               # FastAPI app, routes, request-logging middleware
│   ├── pipeline.py           # End-to-end RAG orchestrator
│   ├── calLLM.py             # LangGraph conversation graph (context memory)
│   ├── vector_store.py       # Pinecone wrapper via LangChain
│   ├── retrievers.py         # Hybrid BM25 + Pinecone retriever, cross-encoder reranker
│   ├── embeddings.py         # OpenAI embedding model wrapper
│   ├── chunker.py            # PDF chunking logic
│   ├── document_processor.py # PDF loading and text extraction
│   ├── query_router.py       # Topic guardrail, query classification, decomposition
│   ├── guardrails.py         # Input (injection/PII) + output (toxicity) guardrails
│   ├── evaluator.py          # LLM evaluator (relevance, completeness, groundedness)
│   └── logger.py             # Structured request logging
├── streamlit_app.py          # Streamlit chat UI
├── Dockerfile
├── start.sh                  # Starts FastAPI (port 8000) + Streamlit (PORT env)
└── requirements.txt
```

## API Reference

### POST `/upload`

Upload a PDF for processing. Chunks the document and indexes it in Pinecone.

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

## Demo

### 1. Upload a document
<img width="1406" height="565" alt="image" src="https://github.com/user-attachments/assets/39cea932-2ad9-4be5-b146-4cbbce2f297b" />

### 2. Ask a question
<img width="1412" height="615" alt="image" src="https://github.com/user-attachments/assets/d4b81463-7cf8-48fe-9ce9-dd797b02596d" />

## Run Locally

### Prerequisites

- Python 3.11+
- A [Pinecone](https://www.pinecone.io/) account and API key
- An [OpenAI](https://platform.openai.com/) API key (for embeddings)
- A [DeepSeek](https://platform.deepseek.com/) API key (for the LLM)

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
   ```

3. Start the backend and UI together:
   ```bash
   chmod +x start.sh && ./start.sh
   ```

   Or start them separately:
   ```bash
   # Backend (FastAPI)
   PYTHONPATH=app uvicorn app.main:app --reload --port 8000

   # UI (Streamlit)
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
  rag-document-qa
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for `text-embedding-3-small` |
| `DEEPSEEK_API_KEY` | Yes | DeepSeek API key for `deepseek-chat` LLM |
| `PINECONE_API_KEY` | Yes | Pinecone API key for vector storage |
| `PORT` | No | Port for Streamlit (Railway sets this automatically) |
| `FASTAPI_PORT` | No | Port for FastAPI backend (default: `8000`) |
| `API_URL` | No | URL Streamlit uses to reach the backend (default: `http://localhost:8000`) |
