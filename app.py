# Streamlit multi-agent starter using LangChain, LangGraph, Weaviate, OpenAI, and Anthropic
# Author: iSoft ANZ | Amitesh Jha
# Run: `pip install -r requirements.txt` then `streamlit run app.py`
from __future__ import annotations
import os
import streamlit as st

from settings import Settings, load_settings
from agents.rag_agent import RAGAgent

# --- App init ---
st.set_page_config(page_title="AI Agents — iSoft", page_icon="🤖", layout="wide")
st.title("AI Agents — Streamlit + LangChain + LangGraph + Weaviate")

# Avatars
USER_AVATAR = "assets/avatar.png"   # user bubble
LLM_AVATAR  = "assets/llm.png"      # assistant bubble

with st.sidebar:
    st.header("Runtime Settings")
    model_provider = st.selectbox("LLM Provider", ["openai", "anthropic"], index=0)
    k = st.slider("Top-K", 1, 20, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    st.caption("Configure .env or environment variables to set API keys and endpoints.")

settings: Settings = load_settings()
if not settings.OPENAI_API_KEY and not settings.ANTHROPIC_API_KEY:
    st.error("Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY")
if not settings.WEAVIATE_URL:
    st.warning("Set WEAVIATE_URL for retrieval to work.")

# --- Simple multi-agent selector ---
tab_rag, = st.tabs(["RAG QA"])

with tab_rag:
    st.subheader("RAG Question Answering")
    question = st.text_input("Ask a question about your KB", placeholder="What is Forecast360?")

    if st.button("Ask", type="primary") and question.strip():
        # User message with custom avatar
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)

        agent = RAGAgent(settings=settings, provider=model_provider, temperature=temperature, top_k=k)
        with st.spinner("Thinking..."):
            answer, sources = agent.run(question)

        # Assistant message with custom avatar
        with st.chat_message("assistant", avatar=LLM_AVATAR):
            st.markdown("### Answer")
            st.write(answer)

            if sources:
                st.markdown("### Sources")
                for i, s in enumerate(sources, 1):
                    uri = s.get("uri", "")
                    score = s.get("score", "")
                    st.markdown(f"{i}. {uri} — score={score}")
