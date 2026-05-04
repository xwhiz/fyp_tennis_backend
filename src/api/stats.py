from fastapi import APIRouter, Depends, Query

from src.db.utils import SessionDep
from src.dependencies.auth import AuthContext, get_auth_context
from src.dependencies.ownership import require_task_access
from src.services.rally_analysis import get_rally_analysis_row

router = APIRouter(tags=["stats"])


def _analysis_payload(session, task_id: int, auth_ctx: AuthContext) -> dict | None:
    require_task_access(session, task_id, auth_ctx)
    row = get_rally_analysis_row(session, task_id)
    return row.public_payload if row is not None else None


def _missing_payload(message: str) -> dict:
    return {"success": True, "message": message}


@router.get("/get_video_paths/{task_id}")
async def get_video_paths(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    payload = _analysis_payload(session, task_id, auth_ctx)
    if payload is None:
        return None if not is_api else _missing_payload("Rally-first analysis not found")
    data = {
        "video": payload.get("video"),
        "rally_ranges": [
            {
                "rally_id": rally.get("rally_id"),
                "scene_index": rally.get("scene_index"),
                "start_frame": rally.get("start_frame"),
                "end_frame": rally.get("end_frame"),
                "start_time_sec": rally.get("start_time_sec"),
                "end_time_sec": rally.get("end_time_sec"),
            }
            for rally in payload.get("rallies", [])
        ],
    }
    return data if not is_api else {"success": True, "data": data}


@router.get("/get_speed_stats/{task_id}")
async def get_speed_stats(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    grouped: bool = Query(False),
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    payload = _analysis_payload(session, task_id, auth_ctx)
    if payload is None:
        return None if not is_api else _missing_payload("Rally-first analysis not found")
    rallies = []
    for rally in payload.get("rallies", []):
        entry = {
            "rally_id": rally.get("rally_id"),
            "scene_index": rally.get("scene_index"),
            "players": {
                "p1": rally.get("players", {}).get("p1", {}).get("speed_stats", []),
                "p2": rally.get("players", {}).get("p2", {}).get("speed_stats", []),
            },
        }
        rallies.append(entry)
    data = {"summary": payload.get("summary"), "rallies": rallies}
    return data if not is_api else {"success": True, "data": data}


@router.get("/get_ball_track/{task_id}")
async def get_ball_track(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    payload = _analysis_payload(session, task_id, auth_ctx)
    if payload is None:
        return None if not is_api else _missing_payload("Rally-first analysis not found")
    data = {
        "rallies": [
            {
                "rally_id": rally.get("rally_id"),
                "scene_index": rally.get("scene_index"),
                "ball_track": rally.get("shared", {}).get("ball_track", []),
            }
            for rally in payload.get("rallies", [])
        ],
    }
    return data if not is_api else {"success": True, "data": data}


@router.get("/get_bounces/{task_id}")
async def get_bounces(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    payload = _analysis_payload(session, task_id, auth_ctx)
    if payload is None:
        return None if not is_api else _missing_payload("Rally-first analysis not found")
    data = {
        "rallies": [
            {
                "rally_id": rally.get("rally_id"),
                "scene_index": rally.get("scene_index"),
                "ball_bounces": rally.get("shared", {}).get("ball_bounces", []),
            }
            for rally in payload.get("rallies", [])
        ],
    }
    return data if not is_api else {"success": True, "data": data}


@router.get("/get_direction_change_indices/{task_id}")
async def get_direction_change_indices_api(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    payload = _analysis_payload(session, task_id, auth_ctx)
    if payload is None:
        return None if not is_api else _missing_payload("Rally-first analysis not found")
    data = {
        "rallies": [
            {
                "rally_id": rally.get("rally_id"),
                "scene_index": rally.get("scene_index"),
                "direction_changes": rally.get("shared", {}).get("direction_changes", []),
            }
            for rally in payload.get("rallies", [])
        ],
    }
    return data if not is_api else {"success": True, "data": data}


@router.get("/get_player_positions/{task_id}")
async def get_player_positions(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    grouped: bool = Query(False),
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    payload = _analysis_payload(session, task_id, auth_ctx)
    if payload is None:
        return None if not is_api else _missing_payload("Rally-first analysis not found")
    rallies = []
    for rally in payload.get("rallies", []):
        entry = {
            "rally_id": rally.get("rally_id"),
            "scene_index": rally.get("scene_index"),
            "players": {
                "p1": {"positions": rally.get("players", {}).get("p1", {}).get("positions", [])},
                "p2": {"positions": rally.get("players", {}).get("p2", {}).get("positions", [])},
            },
        }
        rallies.append(entry)
    data = {"rallies": rallies}
    return data if not is_api else {"success": True, "data": data}


@router.get("/thumbnail/{task_id}")
async def get_thumbnail(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    payload = _analysis_payload(session, task_id, auth_ctx)
    if payload is None:
        return None if not is_api else _missing_payload("Rally-first analysis not found")
    data = {"thumbnail_path": payload.get("video", {}).get("thumbnail_path")}
    return data if not is_api else {"success": True, "data": data}


@router.get("/rally_stats/{task_id}")
async def get_rally_stats(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    payload = _analysis_payload(session, task_id, auth_ctx)
    if payload is None:
        return None if not is_api else _missing_payload("Rally-first analysis not found")
    data = {
        "total_rallies": payload.get("summary", {}).get("total_rallies", 0),
        "rallies": [
            {
                "rally_id": rally.get("rally_id"),
                "scene_index": rally.get("scene_index"),
                "start_frame": rally.get("start_frame"),
                "end_frame": rally.get("end_frame"),
                "start_time_sec": rally.get("start_time_sec"),
                "end_time_sec": rally.get("end_time_sec"),
                "duration_sec": rally.get("duration_sec"),
                "summary": rally.get("summary"),
            }
            for rally in payload.get("rallies", [])
        ],
    }
    return data if not is_api else {"success": True, "data": data}


@router.get("/all-stats/{task_id}")
async def get_all_stats(
    task_id: int,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    payload = _analysis_payload(session, task_id, auth_ctx)
    if payload is None:
        return _missing_payload("Rally-first analysis not found")
    return {"success": True, "data": payload}


@router.get("/serve_stats/{task_id}")
async def get_serve_stats(
    task_id: int,
    session: SessionDep,
    grouped: bool = Query(False),
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    payload = _analysis_payload(session, task_id, auth_ctx)
    if payload is None:
        return _missing_payload("Rally-first analysis not found")
    rallies = []
    for rally in payload.get("rallies", []):
        entry = {
            "rally_id": rally.get("rally_id"),
            "scene_index": rally.get("scene_index"),
            "players": {
                "p1": rally.get("players", {}).get("p1", {}).get("serve_stats"),
                "p2": rally.get("players", {}).get("p2", {}).get("serve_stats"),
            },
        }
        rallies.append(entry)
    data = {"rallies": rallies}
    return {"success": True, "data": data}


@router.get("/player_heatmaps/{task_id}")
async def get_player_heatmaps(
    task_id: int,
    session: SessionDep,
    grouped: bool = Query(False),
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    payload = _analysis_payload(session, task_id, auth_ctx)
    if payload is None:
        return _missing_payload("Rally-first analysis not found")
    data = {
        "rallies": [
            {
                "rally_id": rally.get("rally_id"),
                "scene_index": rally.get("scene_index"),
                "players": {
                    "p1": {"heatmap": rally.get("players", {}).get("p1", {}).get("heatmap")},
                    "p2": {"heatmap": rally.get("players", {}).get("p2", {}).get("heatmap")},
                },
            }
            for rally in payload.get("rallies", [])
        ],
    }
    return {"success": True, "data": data}
