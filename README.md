<img width="1417" height="569" alt="image" src="https://github.com/user-attachments/assets/201f9a1a-996b-452c-b60c-4804186412df" />
A lightweight FastAPI service for uploading PDF files and extracting their text content as a first step toward a Retrieval-Augmented Generation (RAG) workflow.

## Features

- Upload PDF documents via REST API.
- Automatic server-side file storage in an `uploads/` directory.
- PDF text extraction using `PyPDF2`.
- Simple health-check endpoint for monitoring.
- CORS enabled for cross-origin frontend integration.

## Tech Stack

- **Language:** Python 3
- **Framework:** FastAPI
- **ASGI Server:** Uvicorn (recommended for local run)
- **PDF Parsing:** PyPDF2
- **File Handling:** Python standard library (`pathlib`, `shutil`)

## Architecture Diagram 
<img width="679" height="710" alt="image" src="https://github.com/user-attachments/assets/2cb3ae3e-be69-4e1b-aa54-9781a2b58da2" />


## Project Structure

```text
RAG-Document-QA/
├── app/
│   ├── main.py                # FastAPI app and API routes
│   ├── document_processor.py  # PDF text extraction logic
│   └── chunker.py             # Placeholder for chunking logic
└── README.md
```

## API Paths

### 1) Upload Document
- **Method:** `POST`
- **Path:** `/upload`
- **Description:** Uploads a PDF file, stores it in `uploads/`, extracts text, and returns extracted content.
- **Content-Type:** `multipart/form-data`
- **Request Field:**
  - `file` (required): PDF file

## Demo

### 1. Upload a document
<img width="1406" height="565" alt="image" src="https://github.com/user-attachments/assets/39cea932-2ad9-4be5-b146-4cbbce2f297b" />


### 2. Ask a question
<img width="1412" height="615" alt="image" src="https://github.com/user-attachments/assets/d4b81463-7cf8-48fe-9ce9-dd797b02596d" />


#### Example cURL
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf"
```

#### Success Response (200)
```json
{
  "filename": "sample.pdf",
  "file_content": "...extracted text..."
}
```

#### Error Response (400)
```json
{
  "detail": "Only PDF files allowed"
}
```

---

### 2) Health Check
- **Method:** `GET`
- **Path:** `/health`
- **Description:** Returns service health status.

#### Example cURL
```bash
curl "http://127.0.0.1:8000/health"
```

#### Success Response (200)
```json
{
  "status": "healthy"
}
```

## Run Locally

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn pypdf2 python-multipart
   ```

2. Start the API:
   ```bash
   uvicorn app.main:app --reload
   ```

3. Open API docs:
   - Swagger UI: `http://127.0.0.1:8000/docs`
   - ReDoc: `http://127.0.0.1:8000/redoc`
