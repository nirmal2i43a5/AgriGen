import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List
from backend.src.data_loaders import load_all_documents
from langchain_community.document_loaders import PyPDFLoader

router = APIRouter()

@router.get("/documents")
async def list_documents_endpoint(request: Request):
    """List all documents with their chunk information."""
    try:
        if not hasattr(request.app.state, 'rag_pipeline'):
            return {
                "status": "error",
                "message": "RAG pipeline not initialized",
                "documents": []
            }
        
        rag_pipeline = request.app.state.rag_pipeline
        vector_db = rag_pipeline.vector_db
        
        documents = []
        for doc_id in vector_db.get_document_ids():
            doc_info = vector_db.get_document_info(doc_id)
            if doc_info:
                documents.append(doc_info)
        
        return {
            "status": "success",
            "documents": documents,
            "total_documents": len(documents)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing documents: {str(e)}",
            "documents": []
        }

@router.get("/documents/{document_id}/chunks")
async def get_document_chunks_endpoint(request: Request, document_id: str):
    """Get all chunks for a specific document."""
    try:
        if not hasattr(request.app.state, 'rag_pipeline'):
            raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
        
        rag_pipeline = request.app.state.rag_pipeline
        vector_db = rag_pipeline.vector_db
        
        chunks = vector_db.get_document_chunks(document_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "status": "success",
            "document_id": document_id,
            "chunks": chunks,
            "total_chunks": len(chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chunks/{chunk_id}")
async def get_chunk_endpoint(request: Request, chunk_id: str):
    """Get a specific chunk by its ID."""
    try:
        if not hasattr(request.app.state, 'rag_pipeline'):
            raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
        
        rag_pipeline = request.app.state.rag_pipeline
        vector_db = rag_pipeline.vector_db
        
        chunk = vector_db.get_chunk_by_id(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        return {
            "status": "success",
            "chunk": chunk
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def check_status_endpoint(request: Request):
    """Check the status of the documents API and RAG pipeline."""
    try:
        # Check if RAG pipeline is available
        if not hasattr(request.app.state, 'rag_pipeline'):
            return {
                "status": "error",
                "message": "RAG pipeline not initialized",
                "rag_pipeline_available": False
            }
        
        rag_pipeline = request.app.state.rag_pipeline
        
        # Check vector database status
        vector_db_size = rag_pipeline.vector_db.size if rag_pipeline.vector_db else 0
        
        # Check data directory
        raw_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw')
        raw_dir_exists = os.path.exists(raw_dir)
        
        # Count existing PDFs
        existing_pdfs = []
        if raw_dir_exists:
            for file in os.listdir(raw_dir):
                if file.endswith('.pdf'):
                    existing_pdfs.append(file)
        
        return {
            "status": "success",
            "message": "Documents API is working",
            "rag_pipeline_available": True,
            "vector_database_size": vector_db_size,
            "raw_directory": raw_dir,
            "raw_directory_exists": raw_dir_exists,
            "existing_pdfs": existing_pdfs,
            "total_existing_pdfs": len(existing_pdfs)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking status: {str(e)}",
            "rag_pipeline_available": False
        }

@router.post("/upload")
async def upload_documents_endpoint(
    request: Request,
    files: List[UploadFile] = File(...)
):
    """User upload - temporary storage and processing."""
    try:
        # Access RAG pipeline from app state
        rag_pipeline = request.app.state.rag_pipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                if not file.filename.endswith('.pdf'):
                    raise HTTPException(
                        status_code=400, 
                        detail="Only PDF files are allowed"
                    )
                
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
            
            # Load documents using the new document loader
            documents = load_all_documents(temp_dir)
            
            # Index documents using the new RAG pipeline
            rag_pipeline.index_documents(documents)
        
        return {
            "message": f"Successfully processed {len(files)} files",
            "chunks_added": len(documents)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/admin")
async def upload_documents_admin_endpoint(
    request: Request,
    files: List[UploadFile] = File(...)
):
    """For permanent storage with detailed error handling"""
    try:
        print(f"Starting admin upload process...")
        print(f"Received {len(files)} files")
        
        # Check if RAG pipeline is available
        if not hasattr(request.app.state, 'rag_pipeline'):
            raise HTTPException(
                status_code=500, 
                detail="RAG pipeline not initialized"
            )
        
        rag_pipeline = request.app.state.rag_pipeline
        print(f"RAG pipeline accessed successfully")
        
        # Create raw directory
        raw_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw')
        print(f"Raw directory: {raw_dir}")
        
        try:
            os.makedirs(raw_dir, exist_ok=True)
            print(f"Directory created/verified: {raw_dir}")
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create directory {raw_dir}: {str(e)}"
            )
        
        # Check for existing files to avoid duplicates
        existing_files = set()
        if os.path.exists(raw_dir):
            for file in os.listdir(raw_dir):
                if file.endswith('.pdf'):
                    existing_files.add(file)
        
        print(f"Found {len(existing_files)} existing PDFs in directory")
        
        # Get existing sources from vector store to avoid re-processing
        existing_sources = set()
        try:
            existing_sources = rag_pipeline.vector_db.get_existing_sources()
            print(f"Found {len(existing_sources)} existing sources in vector store")
        except Exception as e:
            print(f"Could not check existing sources: {e}")
        
        # Process each file
        processed_files = []
        failed_files = []
        skipped_files = []
        
        for i, file in enumerate(files):
            print(f"Processing file {i+1}/{len(files)}: {file.filename}")
            
            try:
                # Validate file type
                if not file.filename or not file.filename.endswith('.pdf'):
                    error_msg = f"File {file.filename} is not a PDF"
                    print(f" {error_msg}")
                    failed_files.append({"filename": file.filename, "error": error_msg})
                    continue
                
                # Read file content first
                file_content = await file.read()
                file_size = len(file_content)
                print(f"File size: {file_size} bytes")
                
                if file_size == 0:
                    error_msg = f"File {file.filename} is empty"
                    print(f"{error_msg}")
                    failed_files.append({"filename": file.filename, "error": error_msg})
                    continue
                
                if file_size > 50 * 1024 * 1024:  # 50MB limit
                    error_msg = f"File {file.filename} is too large ({file_size} bytes)"
                    print(f"{error_msg}")
                    failed_files.append({"filename": file.filename, "error": error_msg})
                    continue
                
                # Check for duplicates using source path
                file_path = os.path.join(raw_dir, file.filename)
                if file_path in existing_sources:
                    print(f"Skipping duplicate: {file.filename} (already processed)")
                    skipped_files.append(file.filename)
                    continue
                
                # Also check filename duplicates but allow re-processing if content changed
                if file.filename in existing_files:
                    print(f"File {file.filename} exists but will check if content changed")
                    # Don't skip - continue with processing to check content hash
                
                # Save file
                file_path = os.path.join(raw_dir, file.filename)
                print(f"Saving to: {file_path}")
                
                with open(file_path, "wb") as buffer:
                    buffer.write(file_content)
                
                print(f"File saved successfully: {file.filename}")
                processed_files.append(file.filename)
                
            except Exception as e:
                error_msg = f"Failed to process {file.filename}: {str(e)}"
                print(f"{error_msg}")
                failed_files.append({"filename": file.filename, "error": error_msg})
        
        if not processed_files:
            raise HTTPException(
                status_code=400,
                detail=f"No files were successfully processed. Failed files: {failed_files}"
            )
        
        print(f"Loading only newly uploaded documents...")
        
        # Load only the newly uploaded documents, not all documents in directory
        try:
            documents = []
            for filename in processed_files:
                file_path = os.path.join(raw_dir, filename)
                print(f" Loading: {filename}")
                
                # Load individual PDF
                loader = PyPDFLoader(file_path)
                file_documents = loader.load()
                documents.extend(file_documents)
                print(f" Loaded {len(file_documents)} pages from {filename}")
            
            print(f" Total loaded: {len(documents)} documents from {len(processed_files)} files")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load documents: {str(e)}"
            )
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No documents were loaded from the files"
            )
        
        print(f" Indexing documents...")
        
        # Index documents with error handling
        try:
            rag_pipeline.index_documents(documents)
            print(f" Documents indexed successfully")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to index documents: {str(e)}"
            )
        
        return {
            "message": f"Successfully processed {len(processed_files)} files",
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "failed_files": failed_files,
            "chunks_added": len(documents),
            "total_files_received": len(files)
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during upload: {str(e)}"
        )

