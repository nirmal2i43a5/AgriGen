# Multi-Model Router for Orchestrating Parallel Queries
import sys
import os
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.src.llm import get_llm
from backend.src.retriever import setup_qa_chain


class ModelRouter:
    # Orchestrates queries across multiple models
    
    def __init__(self):
        self.qa_chains_cache = {}
        print("[INFO] ModelRouter initialized")
    
    def get_qa_chain_for_model(self, model_name, retriever):
        # Get or create a QA chain for a specific model (cached)
        if model_name not in self.qa_chains_cache:
            print(f"[INFO] Creating QA chain for {model_name}")
            llm = get_llm(model_name=model_name)
            qa_chain = setup_qa_chain(llm, retriever)
            self.qa_chains_cache[model_name] = qa_chain
        
        return self.qa_chains_cache[model_name]
    
    def ask_multi_models(self, models, query, retriever, top_k=3):
        # Ask the same question to multiple models in parallel
        results = {}
        
        for model in models:
            try:
                print(f"[INFO] Querying {model}...")
                
                qa_chain = self.get_qa_chain_for_model(model, retriever)
                response = qa_chain({"query": query})
                
                answer = response.get("result", "No answer available")
                source_docs = response.get("source_documents", [])
                
                # Extract top sources
                sources = []
                for doc in source_docs[:top_k]:
                    sources.append({
                        "content": doc.page_content[:200],  # First 200 chars
                        "source": doc.metadata.get("source", "Unknown"),
                        "distance": doc.metadata.get("distance", 0)
                    })
                
                results[model] = {
                    "answer": answer,
                    "sources": sources
                }
                
                print(f"[INFO] {model} responded with {len(answer)} characters")
            
            except Exception as e:
                print(f"[ERROR] {model} failed: {e}")
                results[model] = {
                    "answer": f"Error: {str(e)}",
                    "sources": []
                }
        
        return results
    
    def clear_cache(self):
        # Clear cached QA chains
        self.qa_chains_cache.clear()
        print("[INFO] QA chain cache cleared")

