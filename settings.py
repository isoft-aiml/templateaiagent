from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass
class Settings:
    # OpenAI
    OPENAI_API_KEY: str | None
    OPENAI_MODEL: str | None
    OPENAI_EMBED_MODEL: str | None

    # Anthropic
    ANTHROPIC_API_KEY: str | None
    ANTHROPIC_MODEL: str | None

    # Weaviate
    WEAVIATE_URL: str | None
    WEAVIATE_API_KEY: str | None
    WEAVIATE_CLASS: str | None

def load_settings() -> Settings:
    return Settings(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
        OPENAI_MODEL=os.getenv("OPENAI_MODEL"),
        OPENAI_EMBED_MODEL=os.getenv("OPENAI_EMBED_MODEL"),
        ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY"),
        ANTHROPIC_MODEL=os.getenv("ANTHROPIC_MODEL"),
        WEAVIATE_URL=os.getenv("WEAVIATE_URL"),
        WEAVIATE_API_KEY=os.getenv("WEAVIATE_API_KEY"),
        WEAVIATE_CLASS=os.getenv("WEAVIATE_CLASS"),
    )
