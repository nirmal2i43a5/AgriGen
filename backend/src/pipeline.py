import os
from dotenv import load_dotenv
from backend.src.data_loader import load_all_documents
from backend.src.embedding import GTELargeEmbeddings, EmbeddingPipeline
from backend.src.vectorstore import VectorStore
from backend.src.llm import get_llm
from backend.src.retriever import setup_qa_chain

load_dotenv()


def initialize_pipeline(persist_directory: str = "faiss_store"):
        
    embeddings = GTELargeEmbeddings()
    vector_store = VectorStore(persist_dir=persist_directory, embeddings=embeddings)
    vector_store.load_or_create()
    
    llm = get_llm()
    retriever = vector_store.as_retriever(k=4)
    qa_chain = setup_qa_chain(llm, retriever)
    
    print("Pipeline initialized successfully")
    return qa_chain, vector_store


def process_documents(directory_path: str, vector_store):
    
    print(f"Processing documents from: {directory_path}")
    docs = load_all_documents(directory_path)
    
    if not docs:
        print("[WARNING] No documents found")
        return 0
    
    pipeline = EmbeddingPipeline()
    chunks = pipeline.chunk_documents(docs)
    vector_store.add_documents(chunks)
    
    print(f"Successfully processed {len(chunks)} chunks")
    return len(chunks)


def ask_question(qa_chain, query: str) -> str:
    print(f"Processing query: {query}")
    response = qa_chain({"query": query})
    return response["result"]

