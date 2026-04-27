from __future__ import annotations

import json
import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlmodel import select

from src.config import settings
from src.db.engine import Engine
from src.db.utils import SessionDep
from src.dependencies.auth import AuthContext, get_auth_context
from src.dependencies.ownership import (
    require_chat_session_access,
    require_chat_stream_access,
    require_task_access,
)
from src.models.chat_attachment import ChatAttachment
from src.models.chat_message import ChatMessage
from src.models.chat_session import ChatSession
from src.models.chat_stream import ChatStream
from src.models.game_stat_embedding import GameStatEmbedding
from src.models.user import User
from src.services.ollama_service import stream_chat
from src.services.rag_pipeline import ensure_storage_dirs, ingest_game_stats, refresh_user_memory, retrieve_context
from src.services.rag_text import summarize_title_from_message
from src.utils.response import success_response

router = APIRouter(prefix="/chat", tags=["chat"])


def _attachment_url(attachment: ChatAttachment) -> str:
    return f"/uploads/chat_attachments/{os.path.basename(attachment.file_path)}"


def _serialize_message(message: ChatMessage, attachments: list[ChatAttachment]) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "metadata": message.metadata_json,
        "attachments": [
            {
                "id": attachment.id,
                "type": attachment.attachment_type,
                "filename": attachment.original_filename,
                "mimeType": attachment.mime_type,
                "fileSize": attachment.file_size,
                "url": _attachment_url(attachment),
                "viewUrl": _attachment_url(attachment),
                "downloadUrl": _attachment_url(attachment),
            }
            for attachment in attachments
            if attachment.message_id == message.id
        ],
        "createdAt": message.created_at,
    }


async def _save_image_attachment(
    session,
    *,
    session_id: str,
    message_id: int,
    image: UploadFile | None,
) -> bytes | None:
    if image is None:
        return None
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image attachment must be an image")
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image attachment")
    ensure_storage_dirs()
    extension = os.path.splitext(image.filename or "")[1].lower() or ".bin"
    filename = f"{uuid.uuid4()}{extension}"
    path = os.path.join(settings.chat_attachment_dir, filename)
    with open(path, "wb") as handle:
        handle.write(raw)
    session.add(
        ChatAttachment(
            session_id=session_id,
            message_id=message_id,
            attachment_type="image",
            file_path=path,
            original_filename=image.filename or filename,
            mime_type=image.content_type or "application/octet-stream",
            file_size=len(raw),
        ),
    )
    session.commit()
    return raw


def _create_stream(session, *, chat_session: ChatSession, message: ChatMessage, model_name: str) -> ChatStream:
    chat_stream = ChatStream(
        session_id=chat_session.id,
        user_id=chat_session.user_id,
        prompt_message_id=message.id,
        status="pending",
        model_name=model_name,
    )
    session.add(chat_stream)
    session.commit()
    session.refresh(chat_stream)
    chat_session.last_stream_id = chat_stream.id
    session.add(chat_session)
    session.commit()
    return chat_stream


@router.post("/start")
async def start_chat(
    session: SessionDep,
    message: str = Form(...),
    task_id: int | None = Form(None),
    image: UploadFile | None = File(None),
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    if task_id is not None:
        require_task_access(session, task_id, auth_ctx)

    chat_session = ChatSession(
        user_id=auth_ctx.user_id,
        task_id=task_id,
        title=summarize_title_from_message(message),
        status="active",
    )
    session.add(chat_session)
    session.commit()
    session.refresh(chat_session)

    user_message = ChatMessage(
        session_id=chat_session.id,
        role="user",
        content=message.strip(),
        metadata_json={"taskId": task_id},
    )
    session.add(user_message)
    session.commit()
    session.refresh(user_message)

    await _save_image_attachment(session, session_id=chat_session.id, message_id=user_message.id, image=image)
    chat_stream = _create_stream(session, chat_session=chat_session, message=user_message, model_name=settings.ollama_chat_model)

    return success_response(
        "Chat session created",
        {
            "sessionId": chat_session.id,
            "streamId": chat_stream.id,
            "streamUrl": f"/chat/{chat_stream.id}",
        },
    )


@router.post("/{session_id}/messages")
async def add_chat_message(
    session_id: str,
    session: SessionDep,
    message: str = Form(...),
    image: UploadFile | None = File(None),
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    chat_session = require_chat_session_access(session, session_id, auth_ctx)
    user_message = ChatMessage(
        session_id=chat_session.id,
        role="user",
        content=message.strip(),
        metadata_json={"taskId": chat_session.task_id},
    )
    session.add(user_message)
    session.commit()
    session.refresh(user_message)

    await _save_image_attachment(session, session_id=chat_session.id, message_id=user_message.id, image=image)
    chat_stream = _create_stream(session, chat_session=chat_session, message=user_message, model_name=settings.ollama_chat_model)

    return success_response(
        "Chat message queued",
        {
            "sessionId": chat_session.id,
            "streamId": chat_stream.id,
            "streamUrl": f"/chat/{chat_stream.id}",
        },
    )


@router.get("/history")
def chat_history(
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    sessions = session.exec(
        select(ChatSession)
        .where(ChatSession.user_id == auth_ctx.user_id)
        .order_by(ChatSession.updated_at.desc()),
    ).all()
    data = []
    for row in sessions:
        last_message = session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == row.id)
            .order_by(ChatMessage.created_at.desc()),
        ).first()
        last_attachment = session.exec(
            select(ChatAttachment)
            .where(
                ChatAttachment.session_id == row.id,
                ChatAttachment.attachment_type == "image",
            )
            .order_by(ChatAttachment.created_at.desc()),
        ).first()
        data.append(
            {
                "id": row.id,
                "title": row.title,
                "taskId": row.task_id,
                "summary": row.summary_text,
                "lastStreamId": row.last_stream_id,
                "lastMessagePreview": (last_message.content[:100] if last_message else None),
                "lastAttachmentImageUrl": (
                    _attachment_url(last_attachment)
                    if last_attachment is not None and last_attachment.attachment_type == "image"
                    else None
                ),
                "updatedAt": row.updated_at,
            },
        )
    return success_response("Chat history fetched", {"sessions": data, "contextSummary": auth_ctx.user.context_summary})


@router.get("/history/{session_id}")
def chat_history_detail(
    session_id: str,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    chat_session = require_chat_session_access(session, session_id, auth_ctx)
    messages = session.exec(
        select(ChatMessage)
        .where(ChatMessage.session_id == chat_session.id)
        .order_by(ChatMessage.created_at.asc()),
    ).all()
    attachments = session.exec(
        select(ChatAttachment).where(ChatAttachment.session_id == chat_session.id),
    ).all()
    return success_response(
        "Chat session fetched",
        {
            "session": {
                "id": chat_session.id,
                "title": chat_session.title,
                "taskId": chat_session.task_id,
                "summary": chat_session.summary_text,
                "lastStreamId": chat_session.last_stream_id,
            },
            "messages": [_serialize_message(message, attachments) for message in messages],
        },
    )


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


@router.get("/{stream_id}")
async def stream_chat_response(
    stream_id: str,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    from sqlmodel import Session

    with Session(Engine.instance()) as session:
        chat_stream = require_chat_stream_access(session, stream_id, auth_ctx)
        chat_session = require_chat_session_access(session, chat_stream.session_id, auth_ctx)
        prompt_message = session.exec(
            select(ChatMessage).where(ChatMessage.id == chat_stream.prompt_message_id),
        ).first()
        if prompt_message is None:
            raise HTTPException(status_code=404, detail="Prompt message not found")
        prompt_attachments = session.exec(
            select(ChatAttachment).where(ChatAttachment.message_id == prompt_message.id),
        ).all()
        history = session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == chat_session.id)
            .order_by(ChatMessage.created_at.asc()),
        ).all()

        if chat_stream.status == "completed" and chat_stream.assistant_message_id is not None:
            assistant = session.exec(
                select(ChatMessage).where(ChatMessage.id == chat_stream.assistant_message_id),
            ).first()
            if assistant is None:
                raise HTTPException(status_code=404, detail="Assistant message not found")

            async def replay():
                yield _sse("started", {"streamId": chat_stream.id, "sessionId": chat_session.id})
                yield _sse("delta", {"content": assistant.content})
                yield _sse("completed", {"messageId": assistant.id})

            return StreamingResponse(replay(), media_type="text/event-stream")

        if chat_session.task_id is not None:
            existing_game_vectors = session.exec(
                select(GameStatEmbedding).where(GameStatEmbedding.task_id == chat_session.task_id),
            ).all()
            if not existing_game_vectors:
                await ingest_game_stats(session, chat_session.task_id)

        retrieval = await retrieve_context(
            session,
            user=auth_ctx.user,
            question=prompt_message.content,
            task_id=chat_session.task_id,
        )

        image_bytes = None
        if prompt_attachments:
            with open(prompt_attachments[0].file_path, "rb") as handle:
                image_bytes = handle.read()

        model_messages = [
            {
                "role": "system",
                "content": retrieval["system_prompt"]
                + "\n\nRetrieved context:\n"
                + ("\n".join(retrieval["context_sections"]) if retrieval["context_sections"] else "No supplemental context."),
            },
        ]
        for row in history:
            model_messages.append({"role": row.role, "content": row.content})

        async def event_stream():
            assistant_chunks: list[str] = []
            with Session(Engine.instance()) as stream_session:
                live_stream = stream_session.exec(
                    select(ChatStream).where(ChatStream.id == stream_id),
                ).first()
                if live_stream is None:
                    yield _sse("error", {"message": "Chat stream not found"})
                    return
                live_stream.status = "streaming"
                live_stream.error_message = None
                stream_session.add(live_stream)
                stream_session.commit()
            try:
                yield _sse("started", {"streamId": stream_id, "sessionId": chat_session.id})
                yield _sse("retrieval", {"sources": retrieval["sources"]})
                async for payload in stream_chat(model_messages, image_bytes=image_bytes):
                    piece = ((payload.get("message") or {}).get("content") or "")
                    if piece:
                        assistant_chunks.append(piece)
                        yield _sse("delta", {"content": piece})
                assistant_text = "".join(assistant_chunks).strip()
                with Session(Engine.instance()) as stream_session:
                    live_stream = stream_session.exec(
                        select(ChatStream).where(ChatStream.id == stream_id),
                    ).first()
                    live_session = stream_session.exec(
                        select(ChatSession).where(ChatSession.id == chat_session.id),
                    ).first()
                    live_user = stream_session.exec(
                        select(User).where(User.id == auth_ctx.user_id),
                    ).first()
                    assistant_message = ChatMessage(
                        session_id=chat_session.id,
                        role="assistant",
                        content=assistant_text,
                        metadata_json={"sources": retrieval["sources"]},
                    )
                    stream_session.add(assistant_message)
                    stream_session.commit()
                    stream_session.refresh(assistant_message)
                    if live_stream is not None:
                        live_stream.status = "completed"
                        live_stream.assistant_message_id = assistant_message.id
                        stream_session.add(live_stream)
                    if live_session is not None:
                        live_session.last_stream_id = stream_id
                        stream_session.add(live_session)
                    stream_session.commit()
                    if live_user is not None and live_session is not None:
                        await refresh_user_memory(stream_session, live_user, live_session)
                    yield _sse("completed", {"messageId": assistant_message.id})
            except Exception as exc:
                with Session(Engine.instance()) as stream_session:
                    live_stream = stream_session.exec(
                        select(ChatStream).where(ChatStream.id == stream_id),
                    ).first()
                    if live_stream is not None:
                        live_stream.status = "failed"
                        live_stream.error_message = str(exc)
                        stream_session.add(live_stream)
                        stream_session.commit()
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(event_stream(), media_type="text/event-stream")
