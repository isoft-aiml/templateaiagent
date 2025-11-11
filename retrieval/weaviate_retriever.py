from __future__ import annotations
from typing import Optional
from langchain_community.vectorstores import Weaviate as LCWeaviate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from settings import Settings
from vector.weaviate_client import get_weaviate_client

class WeaviateSimpleRetriever(BaseRetriever):
    def __init__(self, store: LCWeaviate, k: int = 5):
        self.store = store
        self.k = k

    def _get_relevant_documents(self, query: str):
        return self.store.similarity_search(query, k=self.k)

def get_retriever(settings: Settings, top_k: int = 5) -> BaseRetriever:
    client = get_weaviate_client(settings.WEAVIATE_URL, settings.WEAVIATE_API_KEY)
    embeddings = OpenAIEmbeddings(model=settings.OPENAI_EMBED_MODEL or "text-embedding-3-large", api_key=settings.OPENAI_API_KEY)  # you may swap to Azure/OpenAI or other
    # Assume a default collection/class name; set WEAVIATE_CLASS in .env
    class_name = settings.WEAVIATE_CLASS or "KnowledgeBase"
    store = LCWeaviate(client=client, index_name=class_name, text_key="text", embedding=embeddings)
    return WeaviateSimpleRetriever(store=store, k=top_k)
