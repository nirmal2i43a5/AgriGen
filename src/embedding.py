from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()


class GTELargeEmbeddings(Embeddings):
    model: SentenceTransformer
    model_name: str = "thenlper/gte-large"
    batch_size: int = 32
    
    def __init__(self, model_name: str = "thenlper/gte-large", batch_size: int = 32):
        print(f"Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        print(f"Initialized GTE-Large embeddings (batch_size={batch_size})")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"Embedding {len(texts)} documents...")
        texts = [t.strip() for t in texts if t and t.strip()]
        
        if not texts:
            raise ValueError("No valid texts to embed")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=False
        )
        
        embeddings_list = [emb.tolist() for emb in embeddings]
        print(f"Successfully embedded {len(embeddings_list)} documents")
        return embeddings_list
    
    def embed_query(self, text: str) -> List[float]:
        text = text.strip()
        if not text:
            raise ValueError("Query text cannot be empty")
        
        embedding = self.model.encode(text, convert_to_numpy=False)
        return embedding.tolist()


class EmbeddingPipeline:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"Initialized EmbeddingPipeline (chunk_size={chunk_size}, overlap={chunk_overlap})")

    def chunk_documents(self, documents: List[Any]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks


def get_embeddings(batch_size: int = 32) -> Embeddings:
    return GTELargeEmbeddings(batch_size=batch_size)