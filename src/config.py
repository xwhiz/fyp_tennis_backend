"""Configuration management for GLASS Storage Core."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.person_detector_backend import (
    PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50,
    normalize_person_detector_backend,
)


class Settings(BaseSettings):
    """Application settings."""

    # Database
    database_url: str = "postgresql://postgres:postgres@postgres:5432/acevision"

    # Application
    app_name: str = "AceVision Backend"
    app_env: str = "dev"
    api_version: str = "0.1.0"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Celery
    celery_broker_url: str = "amqp://guest:guest@localhost:5672//"
    celery_result_backend: str = "redis://localhost:6379/0"
    celery_app_name: str = "acevision-backend-tasks"
    celery_worker_concurrency: int = 1
    
    # Flower
    flower_unauthenticated_api: bool = True
    
    # Video Processing
    video_batch_size: int = 200
    person_detector_backend: str = PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50
    
    # Upload Settings
    upload_root_dir: str = "./uploads"
    upload_chunk_size: int = 20 * 1024 * 1024  # 20MB in bytes (default: 20971520)
    profile_image_dir: str = "./uploads/profile_images"
    knowledge_document_dir: str = "./uploads/knowledge_documents"
    chat_attachment_dir: str = "./uploads/chat_attachments"

    # JWT Auth
    jwt_secret: str = "change-me-in-env"
    jwt_algorithm: str = "HS256"
    jwt_expires_in_hours: int = 72
    admin_session_cookie_name: str = "acevision_admin_session"

    # Ollama / RAG
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen3-vl:8b"
    ollama_embedding_model: str = "qwen3-embedding:8b"
    ollama_timeout_seconds: int = 180
    embedding_dimensions: int = 4096
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 200
    rag_retrieval_top_k: int = 5

    # Admin Seeder
    admin_email: str = "admin@example.com"
    admin_password: str = "admin123"
    admin_first_name: str = "Admin"
    admin_last_name: str = "User"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("person_detector_backend", mode="before")
    @classmethod
    def _validate_person_detector_backend(cls, v):
        if v is None or (isinstance(v, str) and not str(v).strip()):
            return PERSON_DETECTOR_BACKEND_FASTER_RCNN_RESNET50
        return normalize_person_detector_backend(str(v))


settings = Settings()
