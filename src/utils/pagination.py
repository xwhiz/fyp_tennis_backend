from __future__ import annotations

from dataclasses import dataclass

from fastapi import Query


@dataclass(frozen=True)
class PaginationParams:
    start: int
    limit: int


def pagination_dependency(*, default_limit: int, max_limit: int = 100):
    def dependency(
        start: int = Query(0, ge=0),
        limit: int = Query(default_limit, ge=1, le=max_limit),
    ) -> PaginationParams:
        return PaginationParams(start=start, limit=limit)

    return dependency


def pagination_metadata(*, start: int, limit: int, total: int, returned: int) -> dict:
    return {
        "start": start,
        "limit": limit,
        "returned": returned,
        "total": total,
        "hasMore": start + returned < total,
    }


def latest_window_bounds(*, total: int, start: int, limit: int) -> tuple[int, int]:
    if start >= total:
        return total, 0

    remaining = total - start
    window_size = min(limit, remaining)
    offset = max(total - start - window_size, 0)
    return offset, window_size
