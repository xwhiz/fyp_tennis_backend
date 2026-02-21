"""Configuration management for GLASS Storage Core."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Database
    database_url: str = "postgresql://postgres:postgres@postgres:5432/acevision"

    # Application
    app_name: str = "AceVision Backend"
    app_env: str = "development"
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Celery
    celery_broker_url: str = "amqp://guest:guest@localhost:5672//"
    celery_result_backend: str = "redis://localhost:6379/0"
    celery_app_name: str = "acevision-backend-tasks"
    celery_worker_concurrency: int = 2
    
    # Flower
    flower_unauthenticated_api: bool = True
    
    # Video Processing
    video_batch_size: int = 500
    
    # Upload Settings
    upload_chunk_size: int = 20 * 1024 * 1024  # 20MB in bytes (default: 20971520)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
