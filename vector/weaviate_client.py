from __future__ import annotations
from typing import Optional
import weaviate
from weaviate.classes.init import Auth

def get_weaviate_client(url: str, api_key: Optional[str] = None, timeout_s: int = 30):
    auth = Auth.api_key(api_key) if api_key else None
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=auth,
        headers={"X-OpenAI-Api-Key": ""},  # no-op header by default
        timeout=timeout_s,
    ) if url.endswith(".semi.network") or url.endswith(".weaviate.network") else weaviate.connect_to_custom(
        http_host=url.replace("https://","").replace("http://",""),
        http_port=443 if url.startswith("https") else 80,
        https=url.startswith("https"),
        auth_credentials=auth,
        timeout=timeout_s,
    )
    return client
