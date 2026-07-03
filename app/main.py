import os
import uuid
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import shutil
from pathlib import Path
from openai import OpenAI, APIConnectionError
from dotenv import load_dotenv
from pydantic import BaseModel

from pipeline import DocumentPipeline
from document_processor import test_extraction
from logger import get_logger, request_id_var

load_dotenv()

app = FastAPI(title="RAG Document Q&A")

logger = get_logger("app.request")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        thread_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(thread_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "request completed",
                extra={
                    "latency_ms": latency_ms,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                },
            )
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = thread_id
        return response


app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    doc_name: str


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

pipeline = DocumentPipeline()


# Both handlers do blocking work (PDF parsing, model inference, sequential LLM
# calls). Declaring them as sync `def` makes FastAPI run them in a threadpool so
# a slow request never blocks the event loop / other requests.
@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    test_extraction(file_path)
    result = pipeline.process_document(Path(file_path))
    return {"filename": file.filename, "result": result}


@app.post("/question")
def answer_question(request: QuestionRequest):
    return pipeline.process_question(request.question, request.doc_name)


@app.get("/health")
async def health_check():
    checks: dict = {}
    overall = "healthy"

    # --- ChromaDB ---
    try:
        stats = pipeline.vector_store.get_stats()
        checks["db"] = {"status": "connected", "total_chunks": stats["total_chunks"]}
    except Exception as exc:
        checks["db"] = {"status": "disconnected", "error": str(exc)}
        overall = "degraded"

    # --- LLM (DeepSeek) ---
    try:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        client.models.list()
        checks["llm"] = {"status": "reachable"}
    except APIConnectionError as exc:
        checks["llm"] = {"status": "unreachable", "error": str(exc)}
        overall = "degraded"
    except Exception:
        # Any non-connection error (e.g. auth) means the endpoint is up
        checks["llm"] = {"status": "reachable"}

    status_code = 200 if overall == "healthy" else 503
    return JSONResponse({"status": overall, "checks": checks}, status_code=status_code)
