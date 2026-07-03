from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from langchain_core.documents import Document
from datetime import datetime


def _splitter(chunk_size: int = 500, chunk_overlap: int = 100) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],  # Priority | Separator
    )


def chunk_documents(
    pages: List[Document],
    doc_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Split loaded PDF pages (LangChain Documents) into chunk Documents and enrich
    their metadata for retrieval, citation, and per-document filtering.

    `document` is the filter key used by both the Chroma and BM25 retrievers.
    """
    chunks = _splitter(chunk_size, chunk_overlap).split_documents(pages)
    total = len(chunks)
    now = datetime.now().isoformat()
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "doc_name": doc_name,
                "document": doc_name,  # filter key for Chroma / BM25
                "chunk_id": i,
                "char_count": len(chunk.page_content),
                "total_chunks": total,
                "uploaded_at": now,
            }
        )
    return chunks


def chunk_text(text:str,filename:str, doc_name:str, chunk_size:int=500, chunk_overlap:int=100) -> List[Dict]:
   splitter = RecursiveCharacterTextSplitter( #langchain library
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      separators=["\n\n", "\n", ".", " ",""]  #Priority | Separator 
   )
   chunks = splitter.split_text(text)

   """
   chunks example-
   [
    "Artificial intelligence is transforming the world. It is being used in healthcare, finance, and education.",

    "In healthcare, AI helps doctors diagnose diseases faster. It can analyze medical images with high accuracy.",

    "In finance, AI detects fraud and automates trading. Banks use it to assess loan risks."
    ]"""

   chunk_data = []
   #returing document object
   for i ,chunk in enumerate(chunks):
      doc = Document(
         page_content=chunk,
         metadata={
            "source": filename,
            "doc_name": doc_name,
            "chunk_id": i,
            "char_count": len(chunk),
            "total_chunks": len(chunks),
            "uploaded_at": datetime.now().isoformat()
         }
      )
      chunk_data.append(doc)
      
   return chunk_data
#only run this code if this file is run directly
if __name__=="__main__":
    sample = "Your long document text here..." * 100
    chunks = chunk_text(sample)
    print(f"Created {len(chunks)} chunks")
    print(f"First chunk: {chunks[0]['text'][:100]}...") #first 100 characters of the string value


   

