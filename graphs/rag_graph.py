from __future__ import annotations
from typing import Dict, Any, List
from langgraph.graph import START, StateGraph
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

class RAGState(BaseModel):
    question: str
    docs: List[Dict[str, Any]] = Field(default_factory=list)
    context: str | None = None
    answer: str | None = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)

def build_rag_graph(retriever, llm: BaseChatModel):
    def retrieve_node(state: RAGState) -> RAGState:
        docs = retriever.get_relevant_documents(state.question)
        # docs are LangChain Documents; convert minimal fields
        packed = []
        for d in docs:
            meta = dict(d.metadata or {})
            packed.append({"text": d.page_content, "uri": meta.get("source") or meta.get("uri") or "", "score": meta.get("_score")})
        state.docs = packed
        state.context = "\n\n".join([d["text"] for d in packed])
        state.sources = packed[:5]
        return state

    def synth_node(state: RAGState) -> RAGState:
        prompt = [
            ("system", "You are a precise assistant. Answer only using the provided context. If the answer is unknown, say 'Insufficient context.'"),
            ("human", f"Question: {state.question}\n\nContext:\n{state.context}")
        ]
        resp = llm.invoke(prompt)
        state.answer = resp.content if hasattr(resp, "content") else str(resp)
        return state

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesize", synth_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "synthesize")
    return g.compile()
