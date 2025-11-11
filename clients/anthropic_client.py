from __future__ import annotations
from typing import Optional
from langchain_anthropic import ChatAnthropic
from settings import Settings

def get_anthropic_llm(settings: Settings, model: Optional[str] = None, temperature: float = 0.2):
    model_name = model or settings.ANTHROPIC_MODEL or "claude-3-5-sonnet-latest"
    return ChatAnthropic(model=model_name, temperature=temperature, api_key=settings.ANTHROPIC_API_KEY, timeout=60)
