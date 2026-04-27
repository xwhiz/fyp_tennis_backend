from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator

import httpx

from src.config import settings


def _client(timeout: float | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.ollama_base_url.rstrip("/"),
        timeout=timeout or settings.ollama_timeout_seconds,
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    async with _client() as client:
        response = await client.post(
            "/api/embed",
            json={"model": settings.ollama_embedding_model, "input": texts},
        )
        response.raise_for_status()
        payload = response.json()
    embeddings = payload.get("embeddings") or []
    return [list(map(float, embedding)) for embedding in embeddings]


async def generate_text(messages: list[dict]) -> str:
    async with _client() as client:
        response = await client.post(
            "/api/chat",
            json={
                "model": settings.ollama_chat_model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
    return ((payload.get("message") or {}).get("content") or "").strip()


async def stream_chat(
    messages: list[dict],
    image_bytes: bytes | None = None,
) -> AsyncIterator[dict]:
    payload_messages = [dict(message) for message in messages]
    if image_bytes:
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        last_user_index = max(
            (idx for idx, item in enumerate(payload_messages) if item.get("role") == "user"),
            default=-1,
        )
        if last_user_index >= 0:
            payload_messages[last_user_index] = {
                **payload_messages[last_user_index],
                "images": [encoded_image],
            }

    async with _client(timeout=None) as client:
        async with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": settings.ollama_chat_model,
                "messages": payload_messages,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                yield payload
