import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import initialize_pipeline, process_documents, ask_question

app = FastAPI(title="Crop Advisory Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Starting API and initializing pipeline...")
qa_chain, vector_store = initialize_pipeline()


class QueryRequest(BaseModel):
    query: str


@app.post("/upload")
async def upload_documents_endpoint(files: List[UploadFile] = File(...)):
    """User upload - temporary storage and processing."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                if not file.filename.endswith('.pdf'):
                    raise HTTPException(status_code=400, detail="Only PDF files are allowed")
                
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
            
            num_chunks = process_documents(temp_dir, vector_store)
        
        return {
            "message": f"Successfully processed {len(files)} files",
            "chunks_added": num_chunks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/admin")
async def upload_documents_admin_endpoint(files: List[UploadFile] = File(...)):
    try:
        "For permanent storage"
        raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            file_path = os.path.join(raw_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        
        num_chunks = process_documents(raw_dir, vector_store)
        
        return {
            "message": f"Successfully uploaded and processed {len(files)} files to permanent storage",
            "chunks_added": num_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_documents_endpoint(request: QueryRequest):
    """Query the document collection."""
    try:
        response = ask_question(qa_chain, request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)