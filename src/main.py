from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from src.api.auth import router as auth_router
from src.api.friend import router as friend_router
from src.api.friends import router as friends_router
from src.api.misc import router as misc_router
from src.api.stats import router as stats_router
from src.api.stream import router as stream_router
from src.api.tasks import router as tasks_router
from src.api.user import router as user_router
from src.api.users_search import router as users_search_router
from src.celery.worker import process_video_task
from src.config import settings
from src.db.engine import Engine
from src.models.background_task import BackgroundTask
from src.seed.admin import seed_admin_user
from src.utils.response import error_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_admin_user()
    try:
        with Session(Engine.instance()) as session:
            statement = select(BackgroundTask).where(
                BackgroundTask.status.in_(["pending", "processing"]),
                BackgroundTask.is_uploaded_fully == True,
            )
            unprocessed_tasks = session.exec(statement).all()
            requeued_count = 0
            for task in unprocessed_tasks:
                try:
                    process_video_task.delay(int(task.id), task.video_path, task.name)
                    if task.status in ["processing"]:
                        task.status = "pending"
                        task.description = "Re-queued after API restart"
                        session.add(task)
                    requeued_count += 1
                except Exception as e:
                    print(f"[STARTUP ERROR]: Failed to re-queue task {task.id}: {str(e)}")

            if requeued_count > 0:
                session.commit()
                print(f"[STARTUP]: Re-queued {requeued_count} unprocessed task(s)")
            else:
                print("[STARTUP]: No unprocessed tasks to re-queue")
    except Exception as e:
        print(f"[STARTUP ERROR]: Failed to re-queue tasks: {str(e)}")
    yield


openapi_tags = [
    {"name": "auth", "description": "Authentication"},
    {"name": "user", "description": "User Profile"},
    {"name": "users", "description": "User search"},
    {"name": "friends", "description": "Friends"},
    {"name": "tasks", "description": "Tasks"},
    {"name": "stats", "description": "Stats"},
    {"name": "stream", "description": "Streaming"},
    {"name": "misc", "description": "Misc"},
]

app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    docs_url=None if settings.app_env == "prod" else "/docs",
    redoc_url=None if settings.app_env == "prod" else "/redoc",
    description="A Self Hosted Tennis Analytics Platform",
    lifespan=lifespan,
    openapi_tags=openapi_tags,
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    message = "; ".join(error.get("msg", "Validation error") for error in exc.errors())
    return JSONResponse(
        status_code=400,
        content=error_response(message or "Validation error details"),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 401:
        message = "Session expired"
    elif exc.status_code == 403:
        message = "Access denied"
    elif exc.status_code >= 500:
        message = "Internal server error"
    elif isinstance(exc.detail, str):
        message = exc.detail
    else:
        message = "Validation error details"
    return JSONResponse(status_code=exc.status_code, content=error_response(message))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content=error_response("Internal server error"))


app.include_router(auth_router)
app.include_router(user_router)
app.include_router(users_search_router)
app.include_router(friends_router)
app.include_router(friend_router)
app.include_router(tasks_router)
app.include_router(stats_router)
app.include_router(stream_router)
app.include_router(misc_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7000)