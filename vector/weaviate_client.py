# vector/weaviate_client.py
from __future__ import annotations
import os
from typing import Optional
import weaviate

def _normalize_url(url: str) -> str:
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        # default to https
        url = "https://" + url
    return url.rstrip("/")

def get_weaviate_client(url: Optional[str], api_key: Optional[str] = None) -> weaviate.WeaviateClient:
    if not url or not isinstance(url, str):
        raise ValueError("WEAVIATE_URL is not set")

    url = _normalize_url(url)

    # Choose connection helper based on host
    host = url.split("://", 1)[-1]
    is_wcs = host.endswith(".semi.network") or host.endswith(".weaviate.network")

    # Auth handling
    auth = weaviate.auth.AuthApiKey(api_key) if api_key else None

    if is_wcs:
        # Weaviate Cloud Service
        client = weaviate.connect_to_wcs(
            cluster_url=url,
            auth_credentials=auth,
            headers={"X-OpenAI-Project": os.getenv("OPENAI_PROJECT", "")} if os.getenv("OPENAI_PROJECT") else None,
        )
    else:
        # Self-hosted or custom
        client = weaviate.connect_to_custom(
            http_host=url,
            auth_credentials=auth,
        )

    return client
