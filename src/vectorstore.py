import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from langchain.schema import Document
from src.embedding import GTELargeEmbeddings, EmbeddingPipeline
from src.retriever import VectorStoreRetriever


class VectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embeddings: GTELargeEmbeddings = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []        
        self.embeddings = embeddings
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"VectorStore initialized with persist_dir: {persist_dir}")

    def build_from_documents(self, documents: List[Any]):
        """Build vector store from raw documents."""
        print(f"Building vector store from {len(documents)} raw documents...")
        
        # Chunk documents
        emb_pipe = EmbeddingPipeline(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        
        
        # Embed chunks
        texts = [chunk.page_content for chunk in chunks]
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        # Create metadata
        metadatas = [{"text": chunk.page_content, "source": chunk.metadata.get("source", "")} for chunk in chunks]
        
        # Add to index
        self.add_embeddings(embeddings_array, metadatas)
        self.save()
        print(f"Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        """Add embeddings to the FAISS index."""
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"Added {embeddings.shape[0]} vectors to FAISS index.")


    def add_documents(self, documents: List[Document]):
        """Add documents to existing store (chunk and embed them)."""
        if not documents:
            print("No documents to add")
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Chunk if needed (if already chunked, skip)
        if all(len(doc.page_content) < self.chunk_size * 2 for doc in documents):
            chunks = documents
        else:
            emb_pipe = EmbeddingPipeline(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            chunks = emb_pipe.chunk_documents(documents)
        
        # Embed
        texts = [chunk.page_content for chunk in chunks]
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        # Metadata
        metadatas = [{"text": chunk.page_content, "source": chunk.metadata.get("source", "")} for chunk in chunks]
        

        self.add_embeddings(embeddings_array, metadatas)
        self.save()


    def save(self):
        
        if self.index is None:  
            print("No index to save")
            return
        
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved FAISS index and metadata to {self.persist_dir}")
        
        

    def load(self):
        """Load FAISS index and metadata from disk."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"Loaded FAISS index and metadata from {self.persist_dir}")
            return True
        return False

    def load_or_create(self):
        if not self.load():
            print("No existing index found, starting with empty index")
        return self

    def search(self, query_embedding: np.ndarray, top_k: int = 4):
        """Search for similar vectors."""
        if self.index is None or self.index.ntotal == 0:
            print("Index is empty")
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        D, I = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 4):
        print(f"Querying vector store for: '{query_text}'")
        query_emb = np.array([self.embeddings.embed_query(query_text)]).astype('float32')
        return self.search(query_emb, top_k=top_k)

    def as_retriever(self, k: int = 4):
        return VectorStoreRetriever(self, k=k)