
from typing import List, Dict, Any
from langchain.schema import Document
import numpy as np
from .text_chunker import TextChunker
from .embeddings import DocumentEmbedder
from .vector_db import VectorDatabase
from .llm.groq_model import get_groq_client
from langchain.schema import Document

class RAGPipeline:

    # Pipeline flow: Chunker -> Embedder -> VectorDB -> LLM
 
    
    def __init__(
        self,
        chunker,     
        embedder,    
        vector_db,   
        llm          
    ):
     
        self.chunker = chunker
        self.embedder = embedder
        self.vector_db = vector_db
        self.llm = llm
        print("RAGPipeline initialized (Chunker → Embedder → VectorDB → LLM)")
    
    def load_and_index_documents(self, data_dir: str):
       
        from .data_loaders import load_all_documents
        
        print(f"Loading documents from: {data_dir}")
        documents = load_all_documents(data_dir)
        self.index_documents(documents)
        return len(documents)
    
    def index_documents(self, documents: List[Document]):
      
        if not documents:
            print("No documents to index")
            return
        
        print(f"Indexing {len(documents)} documents...")
        
        chunks = self.chunker.chunk(documents)
        print(f" Chunked into {len(chunks)} pieces")
        
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        embeddings_array = np.array(embeddings).astype('float32')
        print(f"Created {embeddings_array.shape[0]} embeddings")
        

        # Enhanced metadata with document and chunk IDs
        metadata = []
        for chunk in chunks:
            metadata.append({
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "document_id": chunk.metadata.get("document_id", "unknown"),
                "chunk_id": chunk.metadata.get("chunk_id", "unknown"),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "total_chunks": chunk.metadata.get("total_chunks", 1)
            })
        
        self.vector_db.add(embeddings_array, metadata)
        self.vector_db.save()
        
        print(f" Successfully indexed {len(chunks)} chunks")
    
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for: '{query}'")
        
        query_embedding = np.array([self.embedder.embed_text(query)]).astype('float32')
        
        results = self.vector_db.search(query_embedding, top_k=top_k)
        
        print(f"Retrieved {len(results)} documents")
        return results
    
    def answer(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        print(f"Answering query: '{query}'--------------------------")
       
        retrieved_docs = self.retrieve(query, top_k=top_k)
        
        # Check if we have relevant results (distance threshold)
        relevant_docs = [doc for doc in retrieved_docs if doc["distance"] < 0.7]
        
        if relevant_docs:
            
            # Check if the relevant docs actually contain useful information for the query
            query_keywords = query.lower().split()
            useful_docs = []
            
            for doc in relevant_docs:
                doc_text = doc["metadata"]["text"].lower()
                
                # Check if document contains relevant keywords
                if any(keyword in doc_text for keyword in query_keywords):
                    useful_docs.append(doc)
            
            
            # useful docs => use enhanced prompt
            if useful_docs:
                context = "\n\n".join([
                    f"[Source: {doc['metadata']['source']}]\n{doc['metadata']['text']}"
                    for doc in useful_docs
                ])
                
                prompt = f"""Based on the following context, answer the question comprehensively. Use the provided context as your primary source, and supplement with your knowledge when helpful.

Context:
{context}

Question: {query}

Answer:"""
                
                answer_text = self.llm(prompt)
                
                unique_sources = {}
                for doc in useful_docs:
                    source_path = doc["metadata"]["source"]
                    if source_path not in unique_sources:
                        unique_sources[source_path] = {
                            "source": source_path,
                            "excerpt": doc["metadata"]["text"][:200] + "...",
                            "distance": doc["distance"],
                            "chunk_count": 1
                        }
                    else:
                        # Update with better excerpt if distance is lower
                        if doc["distance"] < unique_sources[source_path]["distance"]:
                            unique_sources[source_path]["excerpt"] = doc["metadata"]["text"][:200] + "..."
                            unique_sources[source_path]["distance"] = doc["distance"]
                        unique_sources[source_path]["chunk_count"] += 1
                
                return {
                    "answer": answer_text,
                    "sources": list(unique_sources.values()),
                    "used_fallback": False
                }
        
        # Fallback to general knowledge (improved)
        print("No relevant documents found, using general knowledge")
        
        fallback_prompt = f"""You are an experienced Agricultural advisor. A farmer has asked: "{query}"

Provide a comprehensive, helpful answer based on your Agricultural expertise. Be practical and actionable. If the question is not about Agriculture, politely redirect to Agricultural topics.

Answer naturally:"""
        
        answer_text = self.llm(fallback_prompt)
        
        return {
            "answer": answer_text,
            "sources": [],
            "used_fallback": True
        }
    

    
    def get_retriever(self):

        class SimpleRetriever:
            def __init__(self, rag_pipeline):
                self.rag_pipeline = rag_pipeline
            
            def get_relevant_documents(self, query, k=4):
                """Get relevant documents for a query"""
                results = self.rag_pipeline.retrieve(query, top_k=k)
                
                docs = []
                for result in results:
                    doc = Document(
                        page_content=result["metadata"]["text"],
                        metadata={
                            "source": result["metadata"]["source"],
                            "distance": result["distance"]
                        }
                    )
                    docs.append(doc)
                return docs
        
        return SimpleRetriever(self)



def initialize_rag_pipeline(storage_path: str = "faiss_store", data_dir: str = None, model_name: str = "llama-3.3-70b-versatile"):
 
    print("Initializing RAG Pipeline...")
    
    chunker = TextChunker(chunk_size=1500, chunk_overlap=300)
    embedder = DocumentEmbedder()
    vector_db = VectorDatabase(storage_path)
    vector_db.load()  # Load existing data if available
    
    print("Connecting to AI model...")
    llm_client = get_groq_client()
    
    def llm_call(prompt: str) -> str:
        """Wrapper to make Groq client compatible with RAG service"""
        try:
            response = llm_client.chat.completions.create(
                model=model_name,  
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    

    rag = RAGPipeline(
        chunker=chunker,    
        embedder=embedder, 
        vector_db=vector_db,
        llm=llm_call      
    )
    
    if data_dir:
        print(f"Loading documents from: {data_dir}")
        rag.load_and_index_documents(data_dir)
    else:
        print("RAG system ready - no documents loaded. Use API endpoints to upload documents.")
    
    print("RAG system ready!")
    return rag

