from __future__ import annotations

import json
import os
import re
from collections import defaultdict

from pypdf import PdfReader
from sqlmodel import Session, select

from src.config import settings
from src.db.vector import cosine_similarity
from src.models.background_task import BackgroundTask
from src.models.bounces import Bounces
from src.models.chat_message import ChatMessage
from src.models.chat_session import ChatSession
from src.models.document_chunk import DocumentChunk
from src.models.game_stat_embedding import GameStatEmbedding
from src.models.knowledge_document import KnowledgeDocument
from src.models.player_positions import PlayerPositions
from src.models.rally_stats import RallyStats
from src.models.speed import Speed
from src.models.system_prompt import SystemPrompt
from src.models.user import User
from src.models.user_memory_entry import UserMemoryEntry
from src.services.ollama_service import embed_texts, generate_text
from src.services.stats_players import build_player_displays, split_speed_by_hitter

DEFAULT_SYSTEM_PROMPT = (
    "You are AceVision, a tennis analysis assistant. Answer with a coaching mindset, "
    "ground claims in the retrieved tennis documents, user history, and game stats, "
    "and clearly say when the stored context is incomplete."
)

GOVERNING_BODIES = ("itf", "atp", "wta", "usta")


def _json_value(value):
    return json.loads(value) if isinstance(value, str) else value


def _normalize_text_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _page_range_label(page_start: int, page_end: int) -> str:
    return str(page_start) if page_start == page_end else f"{page_start}-{page_end}"


def _knowledge_document_url(document: KnowledgeDocument) -> str:
    return f"/uploads/knowledge_documents/{os.path.basename(document.source_file_path)}"


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def extract_pdf_chunks(path: str) -> list[dict]:
    reader = PdfReader(path)
    chunks: list[dict] = []
    chunk_index = 0
    max_chars = max(settings.rag_chunk_size, 1)

    for page_number, page in enumerate(reader.pages, start=1):
        lines = _normalize_text_lines(page.extract_text() or "")
        if not lines:
            continue

        start_line = 1
        buffer: list[str] = []
        current_chars = 0

        for line_index, line in enumerate(lines, start=1):
            if buffer and current_chars + len(line) + 1 > max_chars:
                content = "\n".join(buffer).strip()
                if content:
                    chunks.append(
                        {
                            "chunk_index": chunk_index,
                            "page_start": page_number,
                            "page_end": page_number,
                            "line_start": start_line,
                            "line_end": line_index - 1,
                            "content": content,
                        },
                    )
                    chunk_index += 1
                buffer = []
                current_chars = 0
                start_line = line_index

            buffer.append(line)
            current_chars += len(line) + 1

        content = "\n".join(buffer).strip()
        if content:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "page_start": page_number,
                    "page_end": page_number,
                    "line_start": start_line,
                    "line_end": len(lines),
                    "content": content,
                },
            )
            chunk_index += 1

    return chunks


def build_document_filters(query: str) -> dict[str, object]:
    lowered = query.lower()
    filters: dict[str, object] = {}
    for body in GOVERNING_BODIES:
        if re.search(rf"\b{re.escape(body)}\b", lowered):
            filters["governing_body"] = body.upper()
            break
    year_match = re.search(r"\b(19|20)\d{2}\b", query)
    if year_match:
        filters["season_year"] = int(year_match.group(0))
    return filters


async def ingest_document(session: Session, document_id: int) -> int:
    document = session.exec(
        select(KnowledgeDocument).where(KnowledgeDocument.id == document_id),
    ).first()
    if document is None:
        raise ValueError(f"Knowledge document {document_id} not found")

    document.ingestion_status = "processing"
    document.ingestion_error = None
    session.add(document)
    session.commit()

    for existing in session.exec(
        select(DocumentChunk).where(DocumentChunk.document_id == document_id),
    ).all():
        session.delete(existing)
    session.commit()

    try:
        chunks = extract_pdf_chunks(document.source_file_path)
        embeddings = await embed_texts([chunk["content"] for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            session.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk["chunk_index"],
                    page_start=chunk["page_start"],
                    page_end=chunk["page_end"],
                    line_start=chunk["line_start"],
                    line_end=chunk["line_end"],
                    content=chunk["content"],
                    metadata_json={
                        "title": document.title,
                        "governingBody": document.governing_body,
                        "competition": document.competition,
                        "seasonYear": document.season_year,
                        "pageStart": chunk["page_start"],
                        "pageEnd": chunk["page_end"],
                        "lineStart": chunk["line_start"],
                        "lineEnd": chunk["line_end"],
                    },
                    embedding=embedding,
                ),
            )
        document.ingestion_status = "completed"
        session.add(document)
        session.commit()
        return len(chunks)
    except Exception as exc:
        session.rollback()
        document = session.exec(
            select(KnowledgeDocument).where(KnowledgeDocument.id == document_id),
        ).first()
        if document is not None:
            document.ingestion_status = "failed"
            document.ingestion_error = str(exc)
            session.add(document)
            session.commit()
        raise


def _summarize_speed_block(block: dict) -> str:
    if not block:
        return "No speed data available."
    numeric_values = [
        float(item.get("speed"))
        for item in block.values()
        if isinstance(item, dict) and item.get("speed") is not None
    ]
    if not numeric_values:
        return f"{len(block)} tracked shots with no numeric speed values."
    avg_speed = sum(numeric_values) / len(numeric_values)
    max_speed = max(numeric_values)
    return f"{len(block)} tracked shots, average speed {avg_speed:.2f}, max speed {max_speed:.2f}."


def _summarize_positions_block(block: dict | None) -> str:
    if not block:
        return "No position data available."
    frames = (block or {}).get("positions") or {}
    return f"{len(frames)} tracked position frames."


def build_game_stat_documents(session: Session, task_id: int) -> list[dict]:
    task = session.exec(select(BackgroundTask).where(BackgroundTask.id == task_id)).first()
    if task is None:
        raise ValueError(f"Background task {task_id} not found")

    p1_display, p2_display = build_player_displays(session, task)
    speeds_row = session.exec(select(Speed).where(Speed.task_id == task_id)).first()
    rallies_row = session.exec(select(RallyStats).where(RallyStats.task_id == task_id)).first()
    positions_row = session.exec(select(PlayerPositions).where(PlayerPositions.task_id == task_id)).first()
    bounces_row = session.exec(select(Bounces).where(Bounces.task_id == task_id)).first()

    speed_payload = _json_value(speeds_row.speeds) if speeds_row else {}
    p1_speeds, p2_speeds, unassigned = split_speed_by_hitter(speed_payload)
    positions_payload = _json_value(positions_row.positions) if positions_row else {}
    p1_positions = {"positions": {k: {"bbox": v.get("top")} for k, v in positions_payload.items()}} if positions_payload else None
    p2_positions = {"positions": {k: {"bbox": v.get("bottom")} for k, v in positions_payload.items()}} if positions_payload else None
    rallies = _json_value(rallies_row.rallies) if rallies_row else []
    bounces = _json_value(bounces_row.bounces) if bounces_row else {}

    total_serves = sum(1 for item in (bounces or {}).values() if isinstance(item, dict) and item.get("serve"))
    shared_summary = (
        f"Task {task_id} named '{task.name}' has status '{task.status}'. "
        f"Total rallies recorded: {len(rallies)}. Total detected serves: {total_serves}. "
        f"Unassigned speed events: {len(unassigned)}."
    )

    return [
        {
            "player_scope": "shared",
            "player_user_id": None,
            "source_type": "match_overview",
            "content": shared_summary,
            "metadata_json": {"taskId": task_id},
        },
        {
            "player_scope": "opponent",
            "player_user_id": task.opponent_id,
            "source_type": "player_summary",
            "content": (
                f"Player scope opponent. Display: {p1_display}. "
                f"Speed summary: {_summarize_speed_block(p1_speeds)} "
                f"Position summary: {_summarize_positions_block(p1_positions)}"
            ),
            "metadata_json": {"taskId": task_id, "role": "opponent"},
        },
        {
            "player_scope": "owner",
            "player_user_id": task.owner_id,
            "source_type": "player_summary",
            "content": (
                f"Player scope owner. Display: {p2_display}. "
                f"Speed summary: {_summarize_speed_block(p2_speeds)} "
                f"Position summary: {_summarize_positions_block(p2_positions)}"
            ),
            "metadata_json": {"taskId": task_id, "role": "owner"},
        },
    ]


async def ingest_game_stats(session: Session, task_id: int) -> int:
    for existing in session.exec(
        select(GameStatEmbedding).where(GameStatEmbedding.task_id == task_id),
    ).all():
        session.delete(existing)
    session.commit()

    documents = build_game_stat_documents(session, task_id)
    embeddings = await embed_texts([item["content"] for item in documents])
    for item, embedding in zip(documents, embeddings, strict=False):
        session.add(
            GameStatEmbedding(
                task_id=task_id,
                player_user_id=item["player_user_id"],
                player_scope=item["player_scope"],
                source_type=item["source_type"],
                content=item["content"],
                metadata_json=item["metadata_json"],
                embedding=embedding,
            ),
        )
    session.commit()
    return len(documents)


def _top_similar_rows(rows: list, query_embedding: list[float], limit: int) -> list[tuple[float, object]]:
    ranked = []
    for row in rows:
        score = cosine_similarity(getattr(row, "embedding", None), query_embedding)
        ranked.append((score, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:limit]


def _unique_source_key(source: dict) -> tuple:
    return tuple(sorted(source.items()))


async def retrieve_context(
    session: Session,
    *,
    user: User,
    question: str,
    task_id: int | None = None,
) -> dict[str, object]:
    [query_embedding] = await embed_texts([question])
    filters = build_document_filters(question)

    documents_stmt = select(KnowledgeDocument).where(
        KnowledgeDocument.is_active == True,  # noqa: E712
        KnowledgeDocument.ingestion_status == "completed",
    )
    if filters.get("governing_body"):
        documents_stmt = documents_stmt.where(
            KnowledgeDocument.governing_body == filters["governing_body"],
        )
    if filters.get("season_year") is not None:
        documents_stmt = documents_stmt.where(
            KnowledgeDocument.season_year == filters["season_year"],
        )
    documents = session.exec(documents_stmt).all()
    document_ids = [document.id for document in documents]
    document_by_id = {document.id: document for document in documents}
    document_chunks = []
    if document_ids:
        document_chunks = session.exec(
            select(DocumentChunk).where(DocumentChunk.document_id.in_(document_ids)),
        ).all()

    ranked_docs = _top_similar_rows(document_chunks, query_embedding, settings.rag_retrieval_top_k)

    memory_rows = session.exec(
        select(UserMemoryEntry).where(UserMemoryEntry.user_id == user.id),
    ).all()
    ranked_memory = _top_similar_rows(memory_rows, query_embedding, min(3, settings.rag_retrieval_top_k))

    game_rows: list[GameStatEmbedding] = []
    if task_id is not None:
        game_rows = session.exec(
            select(GameStatEmbedding).where(GameStatEmbedding.task_id == task_id),
        ).all()
    ranked_game = _top_similar_rows(game_rows, query_embedding, settings.rag_retrieval_top_k)

    active_prompts = session.exec(
        select(SystemPrompt).where(SystemPrompt.is_active == True),  # noqa: E712
    ).all()
    prompt_sections = [DEFAULT_SYSTEM_PROMPT] + [prompt.content for prompt in active_prompts]

    context_sections = []
    sources = []
    seen_sources: set[tuple] = set()
    seen_memory_summaries: set[str] = set()

    for _, chunk in ranked_docs:
        doc = document_by_id.get(chunk.document_id)
        if doc is None:
            continue
        context_sections.append(
            f"[Document] {doc.title} ({doc.governing_body} {doc.season_year or 'n/a'}): {chunk.content}",
        )
        source = {
            "type": "document",
            "title": doc.title,
            "governingBody": doc.governing_body,
            "competition": doc.competition,
            "seasonYear": doc.season_year,
            "pageStart": chunk.page_start,
            "pageEnd": chunk.page_end,
            "pageRange": _page_range_label(chunk.page_start, chunk.page_end),
            "lineStart": chunk.line_start,
            "lineEnd": chunk.line_end,
            "viewUrl": _knowledge_document_url(doc),
            "downloadUrl": _knowledge_document_url(doc),
        }
        source_key = _unique_source_key(source)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(source)

    for _, memory in ranked_memory:
        if memory.summary_text in seen_memory_summaries:
            continue
        seen_memory_summaries.add(memory.summary_text)
        context_sections.append(f"[UserMemory] {memory.summary_text}")
        source = {
            "type": "user_memory",
            "summary": memory.summary_text,
            "source": (memory.metadata_json or {}).get("source"),
        }
        source_key = _unique_source_key(source)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(source)

    for _, row in ranked_game:
        context_sections.append(f"[GameStats] {row.content}")
        source = {
            "type": "game_stats",
            "taskId": row.task_id,
            "playerScope": row.player_scope,
            "sourceType": row.source_type,
        }
        source_key = _unique_source_key(source)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(source)

    return {
        "system_prompt": "\n\n".join(prompt_sections),
        "context_sections": context_sections,
        "sources": sources,
    }


async def refresh_user_memory(session: Session, user: User, chat_session: ChatSession) -> None:
    recent_messages = session.exec(
        select(ChatMessage)
        .where(ChatMessage.session_id == chat_session.id)
        .order_by(ChatMessage.created_at.asc()),
    ).all()
    transcript = "\n".join(
        f"{message.role.upper()}: {message.content}" for message in recent_messages[-10:]
    )
    if not transcript.strip():
        return

    summary_prompt = [
        {
            "role": "system",
            "content": (
                "Summarize the user's ongoing tennis journey, chat history, and match progress in "
                "3-5 concise sentences. Preserve useful coaching context."
            ),
        },
        {"role": "user", "content": transcript},
    ]
    summary_text = await generate_text(summary_prompt)
    if not summary_text:
        return

    [embedding] = await embed_texts([summary_text])
    user.context_summary = summary_text
    chat_session.summary_text = summary_text
    session.add(user)
    session.add(chat_session)
    session.add(
        UserMemoryEntry(
            user_id=user.id,
            chat_session_id=chat_session.id,
            summary_text=summary_text,
            metadata_json={"source": "chat_session"},
            embedding=embedding,
        ),
    )
    session.commit()


def describe_documents_grouped(session: Session) -> list[dict]:
    rows = session.exec(
        select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc()),
    ).all()
    chunk_counts = defaultdict(int)
    for chunk in session.exec(select(DocumentChunk)).all():
        chunk_counts[chunk.document_id] += 1
    payload = []
    for row in rows:
        payload.append(
            {
                "id": row.id,
                "title": row.title,
                "governingBody": row.governing_body,
                "competition": row.competition,
                "seasonYear": row.season_year,
                "originalFilename": row.original_filename,
                "status": row.ingestion_status,
                "active": row.is_active,
                "chunkCount": chunk_counts[row.id],
                "error": row.ingestion_error,
            },
        )
    return payload


def ensure_storage_dirs() -> None:
    os.makedirs(settings.knowledge_document_dir, exist_ok=True)
    os.makedirs(settings.chat_attachment_dir, exist_ok=True)
