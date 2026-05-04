from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import select

from src.celery.worker import ingest_document_task
from src.config import settings
from src.core.person_detector_backend import (
    PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50,
    PERSON_DETECTOR_BACKEND_YOLO26X,
)
from src.db.utils import SessionDep
from src.dependencies.auth import AuthContext, get_auth_context
from src.dependencies.admin_auth import get_admin_auth_context
from src.dependencies.ownership import is_admin
from src.models.knowledge_document import KnowledgeDocument
from src.models.system_prompt import SystemPrompt
from src.models.user import User, UserRole
from src.services.jwt_service import create_access_token
from src.services.rally_analysis import rebuild_and_save_rally_analysis_from_legacy
from src.services.rag_pipeline import describe_documents_grouped, ensure_storage_dirs
from src.services.runtime_config import (
    get_active_person_detector_backend,
    redis_configured_for_runtime,
    set_active_person_detector_backend,
)

router = APIRouter(tags=["admin"])
templates = Jinja2Templates(directory="src/templates")


def require_admin_api(auth_ctx: AuthContext) -> AuthContext:
    if not is_admin(auth_ctx):
        raise HTTPException(status_code=403, detail="Access denied")
    return auth_ctx


def _dashboard_context(request: Request, session, auth_ctx, notice: str | None = None) -> dict:
    prompts = session.exec(
        select(SystemPrompt).order_by(SystemPrompt.created_at.desc()),
    ).all()
    return {
        "request": request,
        "auth_ctx": auth_ctx,
        "notice": notice,
        "prompts": prompts,
        "documents": describe_documents_grouped(session),
    }


@router.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    return templates.TemplateResponse(
        request,
        "admin/login.html",
        {"request": request, "error": None},
    )


@router.post("/admin/login", response_class=HTMLResponse)
def admin_login(
    request: Request,
    session: SessionDep,
    email: str = Form(...),
    password: str = Form(...),
):
    user = session.exec(
        select(User).where(User.email == email.strip().lower()),
    ).first()
    if user is None or not user.verify_password(password):
        return templates.TemplateResponse(
            request,
            "admin/login.html",
            {"request": request, "error": "Invalid email or password"},
            status_code=400,
        )
    if user.role != UserRole.ADMIN:
        return templates.TemplateResponse(
            request,
            "admin/login.html",
            {"request": request, "error": "Admin access is required"},
            status_code=403,
        )

    token = create_access_token(user_id=user.id, role=user.role.value, email=user.email)
    response = RedirectResponse(url="/admin/chat", status_code=303)
    response.set_cookie(
        settings.admin_session_cookie_name,
        token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=settings.jwt_expires_in_hours * 3600,
    )
    return response


@router.post("/admin/logout")
def admin_logout():
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie(settings.admin_session_cookie_name)
    return response


@router.get("/admin", response_class=HTMLResponse)
def admin_root_redirect(auth_ctx=Depends(get_admin_auth_context)):
    return RedirectResponse(url="/admin/chat", status_code=303)


@router.get("/admin/chat", response_class=HTMLResponse)
def admin_chat_dashboard(
    request: Request,
    session: SessionDep,
    notice: str | None = None,
    auth_ctx=Depends(get_admin_auth_context),
):
    ctx = _dashboard_context(request, session, auth_ctx, notice)
    ctx["active_nav"] = "chat"
    return templates.TemplateResponse(request, "admin/chat.html", ctx)


@router.get("/admin/person-detector", response_class=HTMLResponse)
def admin_person_detector_page(
    request: Request,
    session: SessionDep,
    notice: str | None = None,
    auth_ctx=Depends(get_admin_auth_context),
):
    current = get_active_person_detector_backend()
    ctx = {
        "request": request,
        "auth_ctx": auth_ctx,
        "notice": notice,
        "active_nav": "person_detector",
        "current_backend": current,
        "backend_yolo": PERSON_DETECTOR_BACKEND_YOLO26X,
        "backend_frcnn": PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50,
        "redis_configured": redis_configured_for_runtime(),
        "prompts": [],
        "documents": [],
    }
    return templates.TemplateResponse(request, "admin/person_detector.html", ctx)


@router.post("/admin/person-detector")
def admin_person_detector_save(
    backend: str = Form(...),
    auth_ctx=Depends(get_admin_auth_context),
):
    try:
        set_active_person_detector_backend(backend)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RedirectResponse(
        url="/admin/person-detector?notice=Person+detector+saved",
        status_code=303,
    )


@router.post("/admin/prompts")
def create_prompt(
    session: SessionDep,
    name: str = Form(...),
    content: str = Form(...),
    auth_ctx=Depends(get_admin_auth_context),
):
    prompt_name = name.strip()
    if not prompt_name:
        raise HTTPException(status_code=400, detail="Prompt name is required")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Prompt content is required")
    existing = session.exec(select(SystemPrompt).where(SystemPrompt.name == prompt_name)).first()
    if existing is not None:
        raise HTTPException(status_code=400, detail="Prompt name already exists")

    session.add(
        SystemPrompt(
            name=prompt_name,
            content=content.strip(),
            is_active=True,
            created_by=auth_ctx.user_id,
        ),
    )
    session.commit()
    return RedirectResponse(url="/admin/chat?notice=Prompt+created", status_code=303)


@router.post("/admin/prompts/{prompt_id}/toggle")
def toggle_prompt(
    prompt_id: int,
    session: SessionDep,
    auth_ctx=Depends(get_admin_auth_context),
):
    prompt = session.exec(select(SystemPrompt).where(SystemPrompt.id == prompt_id)).first()
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    prompt.is_active = not prompt.is_active
    session.add(prompt)
    session.commit()
    return RedirectResponse(url="/admin/chat?notice=Prompt+updated", status_code=303)


@router.post("/admin/documents")
async def upload_document(
    session: SessionDep,
    title: str = Form(...),
    governing_body: str = Form(...),
    competition: str | None = Form(None),
    season_year: int | None = Form(None),
    pdf_file: UploadFile = File(...),
    auth_ctx=Depends(get_admin_auth_context),
):
    if pdf_file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    ensure_storage_dirs()
    extension = os.path.splitext(pdf_file.filename or "")[1].lower() or ".pdf"
    filename = f"{uuid.uuid4()}{extension}"
    path = os.path.join(settings.knowledge_document_dir, filename)
    raw = await pdf_file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty PDF file")
    with open(path, "wb") as handle:
        handle.write(raw)

    document = KnowledgeDocument(
        title=title.strip(),
        governing_body=governing_body.strip().upper(),
        competition=competition.strip() if competition and competition.strip() else None,
        season_year=season_year,
        source_file_path=path,
        original_filename=pdf_file.filename or filename,
        mime_type=pdf_file.content_type or "application/pdf",
        file_size=len(raw),
        is_active=True,
        ingestion_status="pending",
        uploaded_by=auth_ctx.user_id,
    )
    session.add(document)
    session.commit()
    session.refresh(document)
    ingest_document_task.delay(int(document.id))
    return RedirectResponse(url="/admin/chat?notice=Document+uploaded", status_code=303)


@router.post("/admin/documents/{document_id}/toggle")
def toggle_document(
    document_id: int,
    session: SessionDep,
    auth_ctx=Depends(get_admin_auth_context),
):
    document = session.exec(
        select(KnowledgeDocument).where(KnowledgeDocument.id == document_id),
    ).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Knowledge document not found")
    document.is_active = not document.is_active
    session.add(document)
    session.commit()
    return RedirectResponse(url="/admin/chat?notice=Document+updated", status_code=303)


@router.post("/admin/tasks/{task_id}/backfill-rally-analysis")
def backfill_rally_analysis(
    task_id: int,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
):
    require_admin_api(auth_ctx)
    payload = rebuild_and_save_rally_analysis_from_legacy(session, task_id)
    return {
        "success": True,
        "data": {
            "task_id": task_id,
            "schema_version": payload.get("schema_version"),
            "summary": payload.get("summary"),
        },
    }
