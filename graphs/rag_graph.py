# graphs/rag_graph.py
from __future__ import annotations
from typing import Dict, Any, List, TypedDict
from langgraph.graph import START, StateGraph
from langchain_core.language_models.chat_models import BaseChatModel

class RAGState(TypedDict, total=False):
    question: str
    docs: List[Dict[str, Any]]
    context: str
    answer: str
    sources: List[Dict[str, Any]]

def build_rag_graph(retriever, llm: BaseChatModel):
    def retrieve_node(state: RAGState) -> RAGState:
        docs = retriever.get_relevant_documents(state["question"])
        packed = []
        for d in docs:
            meta = dict(d.metadata or {})
            packed.append({
                "text": d.page_content,
                "uri": meta.get("source") or meta.get("uri") or "",
                "score": meta.get("_score"),
            })
        state["docs"] = packed
        state["context"] = "\n\n".join([d["text"] for d in packed])
        state["sources"] = packed[:5]
        return state

    def synth_node(state: RAGState) -> RAGState:
        prompt = [
            ("system", "Answer only from the provided context. If unknown, say 'Insufficient context.'"),
            ("human", f"Question: {state['question']}\n\nContext:\n{state.get('context','')}"),
        ]
        resp = llm.invoke(prompt)
        state["answer"] = getattr(resp, "content", str(resp))
        return state

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesize", synth_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "synthesize")
    return g.compile()
