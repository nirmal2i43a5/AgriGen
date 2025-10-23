
import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any


class VectorDatabase:
    
    def __init__(self, storage_path: str = "faiss_store"):
        self.storage_path = storage_path
        self.index = None
        self.metadata = []
        os.makedirs(storage_path, exist_ok=True)
        print(f"VectorDatabase initialized at: {storage_path}")
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add vectors and their metadata to the database."""
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match metadata entries")
        
        # Initialize index if needed
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            print(f"Created new FAISS index (dimension={dim})")
        
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        print(f"Added {embeddings.shape[0]} vectors to database")
    
    # Similarity search
    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            print("Database is empty")
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(
            query_embedding, 
            min(top_k, self.index.ntotal)
        )
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append({
                    "distance": float(dist),
                    "metadata": self.metadata[idx]
                })
        return results
    
    def save(self):
        """Persist index and metadata to disk."""
        if self.index is None:
            print("No index to save")
            return
        
        index_path = os.path.join(self.storage_path, "faiss.index")
        meta_path = os.path.join(self.storage_path, "metadata.pkl")
        
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Database saved to {self.storage_path}")
    
    def load(self) -> bool:
        index_path = os.path.join(self.storage_path, "faiss.index")
        meta_path = os.path.join(self.storage_path, "metadata.pkl")
        
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            print("No existing database found")
            return False
        
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded database from {self.storage_path} ({self.index.ntotal} vectors)")
        return True
    
    @property
    def size(self) -> int:
        """Return vector size in database."""
        return self.index.ntotal if self.index else 0
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        chunks = []
        seen_chunk_ids = set()
        
        for i, meta in enumerate(self.metadata):
            if meta.get("document_id") == document_id:
                chunk_id = meta.get("chunk_id")
                

                if chunk_id not in seen_chunk_ids:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "chunk_index": meta.get("chunk_index"),
                        "text": meta.get("text"),
                        "source": meta.get("source"),
                        "vector_index": i
                    })
                    seen_chunk_ids.add(chunk_id)
        
        return sorted(chunks, key=lambda x: x["chunk_index"])
    
    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        for i, meta in enumerate(self.metadata):
            if meta.get("chunk_id") == chunk_id:
                return {
                    "chunk_id": meta.get("chunk_id"),
                    "document_id": meta.get("document_id"),
                    "text": meta.get("text"),
                    "source": meta.get("source"),
                    "chunk_index": meta.get("chunk_index"),
                    "total_chunks": meta.get("total_chunks"),
                    "vector_index": i
                }
        return None
    
    def get_document_ids(self) -> List[str]:
        doc_ids = set()
        for meta in self.metadata:
            if "document_id" in meta:
                doc_ids.add(meta["document_id"])
        return list(doc_ids)
    
    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        chunks = self.get_document_chunks(document_id)
        if not chunks:
            return None
        
        # Get unique chunk IDs to avoid duplicates
        unique_chunk_ids = list(set(chunk["chunk_id"] for chunk in chunks))
        
        return {
            "document_id": document_id,
            "source": chunks[0].get("source", "unknown"),
            "total_chunks": len(unique_chunk_ids),
            "chunk_ids": unique_chunk_ids
        }
    
    def get_existing_sources(self) -> set:
        #Get  existing source files from metadata.
        sources = set()
        for meta in self.metadata:
            if "source" in meta:
                sources.add(meta["source"])
        return sources
