from __future__ import annotations

import math
from collections.abc import Sequence

from sqlalchemy import JSON

from src.config import settings

try:
    from pgvector.sqlalchemy import Vector
except Exception:  # pragma: no cover - dependency should exist in runtime
    Vector = None


def embedding_column_type():
    if Vector is None:
        return JSON
    return Vector(settings.embedding_dimensions).with_variant(JSON, "sqlite")


def _coerce_embedding(values: object) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return [float(value) for value in values]
    return []


def cosine_similarity(lhs: object, rhs: object) -> float:
    lhs_values = _coerce_embedding(lhs)
    rhs_values = _coerce_embedding(rhs)
    if len(lhs_values) == 0 or len(rhs_values) == 0:
        return 0.0
    if len(lhs_values) != len(rhs_values):
        return 0.0
    dot = sum(a * b for a, b in zip(lhs_values, rhs_values, strict=False))
    lhs_norm = math.sqrt(sum(a * a for a in lhs_values))
    rhs_norm = math.sqrt(sum(b * b for b in rhs_values))
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0
    return dot / (lhs_norm * rhs_norm)
