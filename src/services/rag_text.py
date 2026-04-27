from __future__ import annotations

from src.config import settings


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []

    size = chunk_size or settings.rag_chunk_size
    step_overlap = overlap if overlap is not None else settings.rag_chunk_overlap
    if size <= 0:
        return [normalized]
    if step_overlap >= size:
        step_overlap = max(size // 4, 1)

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - step_overlap, start + 1)
    return chunks


def summarize_title_from_message(message: str) -> str:
    clean = " ".join((message or "").split()).strip()
    if not clean:
        return "New chat"
    if len(clean) <= 60:
        return clean
    return clean[:57].rstrip() + "..."
