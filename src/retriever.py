from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from typing import List,Any


class VectorStoreRetriever(BaseRetriever):
    
    
    vector_store: Any  
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, vector_store, k: int = 4):
        super().__init__(vector_store=vector_store, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        results = self.vector_store.query(query, top_k=self.k)
        documents = []
        for result in results:
            metadata = result.get("metadata", {})
            doc = Document(
                page_content=metadata.get("text", ""),
                metadata={
                    "source": metadata.get("source", ""),
                    "distance": result.get("distance", 0)
                }
            )
            documents.append(doc)
        return documents



def setup_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )