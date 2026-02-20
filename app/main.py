from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path

from document_processor import test_extraction


app = FastAPI(title= "RAG Document Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = Path("uploads")  #path object where the uploaded files will be saved
UPLOAD_DIR.mkdir(exist_ok=True) # if folder doesnt exist then it creates it


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):  #file parameter is of type UploadFile, File(...) -> file is expected /mandatory
    #upload and process PDF document

    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    
    file_path = UPLOAD_DIR/file.filename #file path

    #save file 
    with file_path.open("wb") as buffer: #opens file in write binary mode
        shutil.copyfileobj(file.file, buffer)  # (source, des) copies data to buffer 

    text = test_extraction(file_path)

    return {"filename":file.filename, "file_content":text}

@app.get("/health")
def health_check():
    return {"status":"healthy"}

