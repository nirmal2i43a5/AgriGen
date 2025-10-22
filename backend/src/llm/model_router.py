# Multi-Model Router for Orchestrating Parallel Queries

import sys
import os
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from .groq_model import get_groq_client


class ModelRouter:
    
    def __init__(self):
        self.client = get_groq_client()
        print("ModelRouter initialized")
    
    def ask_multi_models(self, models, query, retriever, top_k=3):
     
        results = {}
        
        # Get relevant documents once (shared across all models)
        docs = retriever.get_relevant_documents(query, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        has_relevant_docs = len(docs) > 0
        
        for model_name in models:
            try:
                if has_relevant_docs:
                    prompt = f"""Based on the following context, answer the question. IMPORTANT: If the context doesn't contain enough information to fully answer the question, provide comprehensive knowledge from your training to give a complete response. Don't just say "the context doesn't contain information" - use your full knowledge to answer the question.

Context:
{context}

Question: {query}

Answer:"""
                else:
                    # Fallback to general knowledge - comprehensive response
                    prompt = f"""You are an experienced Agricultural advisor specializing in crops, farming, climate, and Agricultural practices. A farmer has asked you: "{query}"

Share your comprehensive knowledge about this topic. Be detailed and practical. Draw from your understanding of Agricultural principles, best practices, and regional considerations. Provide specific, actionable advice.

If the question is NOT related to Agriculture (like medical health, politics, technology, etc.), politely redirect by saying something like: "I'm here to help you with Agricultural topics like farming, crops, climate, and soil management. Could you ask me something about Agriculture instead? I'd be happy to help with farming advice, crop selection, pest management, or any other Agricultural questions you might have."

Respond naturally as if you're having a conversation with a farmer:"""
                
                # Get response from specific model
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=2048
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Prepare sources information
                sources = []
                if has_relevant_docs:
                    sources = [
                        {
                            "source": doc.metadata.get("source", "unknown"),
                            "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        }
                        for doc in docs
                    ]
                
                results[model_name] = {
                    "answer": answer,
                    "sources": sources,
                    "used_fallback": not has_relevant_docs
                }
                
            except Exception as e:
                results[model_name] = {
                    "answer": f"Error generating response: {str(e)}",
                    "sources": [],
                    "used_fallback": True
                }
        
        return results
    
    def ask_single_model(self, model_name, query, retriever, top_k=3):
       
        results = self.ask_multi_models([model_name], query, retriever, top_k)
        return results[model_name]
    
    def get_available_models(self):
        from .groq_model import GROQ_TEXT_MODELS
        return list(GROQ_TEXT_MODELS.keys())