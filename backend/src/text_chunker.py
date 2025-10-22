
import hashlib
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class TextChunker:
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        print(f"TextChunker initialized (size={chunk_size}, overlap={chunk_overlap})")
    
    def generate_document_id(self, source_path: str) -> str:
        doc_hash = hashlib.md5(source_path.encode()).hexdigest()[:12]
        return f"doc_{doc_hash}"
    
    def generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        return f"{doc_id}_chunk_{chunk_index:04d}"
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        
        for doc in documents:
            # Generate document ID
            doc_id = self.generate_document_id(doc.metadata.get("source", "unknown"))
            
            doc_chunks = self.splitter.split_documents([doc])
            

            for i, chunk in enumerate(doc_chunks):
                chunk_id = self.generate_chunk_id(doc_id, i)
                
                chunk.metadata.update({
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(doc_chunks),
                    "source": doc.metadata.get("source", "unknown")
                })
                
                all_chunks.append(chunk)
        
        print(f"Split {len(documents)} documents into {len(all_chunks)} chunks with IDs")
        return all_chunks
