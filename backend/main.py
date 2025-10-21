import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(__file__))

from api.routes import chat, documents, health

app = FastAPI(title="AgriAdvisor-Ai-Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Starting API and initializing pipeline...")
    from src.pipeline import initialize_pipeline
    
    qa_chain, vector_store = initialize_pipeline()
    
    app.state.qa_chain = qa_chain
    app.state.vector_store = vector_store
    print("Pipeline initialized successfully")


app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(health.router, prefix="/api", tags=["health"])


@app.get("/")
async def root():
    return {
        "message": "AgriAdvisor-Ai-Assistant-API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

