"""Derive unique Instagram-style @ tags from email (stored without leading @)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlmodel import Session


_MAX_BASE_LEN = 32


def normalize_at_tag_input(raw: str) -> str:
    """Strip @ and whitespace; return lowercase canonical tag for lookup."""
    s = (raw or "").strip().lstrip("@").strip().lower()
    return s


def base_slug_from_email(email: str) -> str:
    local = (email or "").split("@", 1)[0].lower()
    slug = re.sub(r"[^a-z0-9_]+", "_", local)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "user"
    return slug[:_MAX_BASE_LEN]


def allocate_unique_at_tag(session: Session, email: str, exclude_user_id: str | None = None) -> str:
    """Return a unique at_tag for a new user (does not commit)."""
    from sqlmodel import select

    from src.models.user import User

    base = base_slug_from_email(email)
    candidate = base
    suffix = 2
    while True:
        q = select(User).where(User.at_tag == candidate)
        if exclude_user_id:
            q = q.where(User.id != exclude_user_id)
        if session.exec(q).first() is None:
            return candidate
        candidate = f"{base}_{suffix}"[:64]
        suffix += 1


def display_at_tag(stored: str) -> str:
    """API display: prefix with @."""
    if not stored:
        return ""
    return f"@{stored}"


def mask_email(email: str) -> str:
    """e.g. alice@example.com -> a***@example.com"""
    if not email or "@" not in email:
        return "***"
    local, _, domain = email.partition("@")
    if len(local) <= 1:
        return f"*@{domain}"
    return f"{local[0]}***@{domain}"
