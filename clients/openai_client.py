from __future__ import annotations
from typing import Optional
from langchain_openai import ChatOpenAI
from settings import Settings

def get_openai_llm(settings: Settings, model: Optional[str] = None, temperature: float = 0.2):
    model_name = model or settings.OPENAI_MODEL or "gpt-4o-mini"
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=settings.OPENAI_API_KEY, timeout=60)
