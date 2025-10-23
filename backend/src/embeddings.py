
from typing import List
from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    
    def __init__(self, model_name: str = "thenlper/gte-large", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name,
                                         
                                         
                                         )
        print(f"Model loaded successfully (batch_size={batch_size})")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            raise ValueError("No valid texts to embed")
        
        print(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=False
        )
        
        result = [emb.tolist() for emb in embeddings]
        print(f"Successfully embedded {len(result)} texts")
        return result
    
    #for incomeing query 
    def embed_text(self, text: str) -> List[float]:
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        
        embedding = self.model.encode(text, convert_to_numpy=False)
        return embedding.tolist()
