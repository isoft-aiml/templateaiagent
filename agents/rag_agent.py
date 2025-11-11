from __future__ import annotations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from settings import Settings
from graphs.rag_graph import build_rag_graph
from clients.openai_client import get_openai_llm
from clients.anthropic_client import get_anthropic_llm
from retrieval.weaviate_retriever import get_retriever

@dataclass
class RAGAgent:
    settings: Settings
    provider: str = "openai"  # "openai" or "anthropic"
    model: str | None = None
    temperature: float = 0.2
    top_k: int = 5

    def run(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        retriever = get_retriever(self.settings, top_k=self.top_k)
        if self.provider == "anthropic":
            llm = get_anthropic_llm(self.settings, model=self.model, temperature=self.temperature)
        else:
            llm = get_openai_llm(self.settings, model=self.model, temperature=self.temperature)
        graph = build_rag_graph(retriever=retriever, llm=llm)
        result = graph.invoke({"question": question})
        return result["answer"], result.get("sources", [])
