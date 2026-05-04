"""Cross-process runtime settings (Redis). Falls back to env defaults when Redis is unavailable."""

from __future__ import annotations

import logging

import redis

from src.config import settings
from src.core.person_detector_backend import normalize_person_detector_backend

log = logging.getLogger(__name__)

REDIS_KEY_PERSON_DETECTOR = "acevision:person_detector_backend"


def _redis_url() -> str | None:
    url = (settings.celery_result_backend or "").strip()
    if url.startswith("redis://") or url.startswith("rediss://"):
        return url
    return None


def redis_configured_for_runtime() -> bool:
    return _redis_url() is not None


def _redis_client():
    url = _redis_url()
    if not url:
        return None
    try:
        return redis.Redis.from_url(url, decode_responses=True)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Redis client init failed: %s", exc)
        return None


def get_active_person_detector_backend() -> str:
    """Backend for new tasks. Seeds Redis from settings when the key is unset."""
    default = normalize_person_detector_backend(settings.person_detector_backend)
    client = _redis_client()
    if client is None:
        log.debug(
            "person_detector_backend: no redis URL; using settings default %s",
            default,
        )
        return default
    try:
        raw = client.get(REDIS_KEY_PERSON_DETECTOR)
        if raw is None or str(raw).strip() == "":
            client.set(REDIS_KEY_PERSON_DETECTOR, default)
            return default
        return normalize_person_detector_backend(str(raw))
    except Exception as exc:
        log.warning(
            "Redis get %s failed (%s); using settings default %s",
            REDIS_KEY_PERSON_DETECTOR,
            exc,
            default,
        )
        return default


def set_active_person_detector_backend(value: str) -> str:
    """Persist backend for workers and API. Requires a working Redis URL."""
    normalized = normalize_person_detector_backend(value)
    client = _redis_client()
    if client is None:
        raise RuntimeError(
            "celery_result_backend must be a redis:// or rediss:// URL to switch "
            "person detector at runtime.",
        )
    client.set(REDIS_KEY_PERSON_DETECTOR, normalized)
    return normalized
