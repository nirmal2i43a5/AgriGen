import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
sys.path.insert(0, os.path.dirname(__file__))

from api.routes import chat, documents

app = FastAPI(title="AgriGen - Farm Advisor Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Starting API and initializing RAG pipeline...")
    from backend.src.rag_pipeline import initialize_rag_pipeline
    
    rag_pipeline = initialize_rag_pipeline()
    
    # Store the pipeline in app state for API routes(saves time and resources)
    app.state.rag_pipeline = rag_pipeline
    print("RAG pipeline initialized successfully")


app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])

@app.get("/")
async def root():
    return {
        "message": "AgriGen - Farm Advisor Assistant-API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

