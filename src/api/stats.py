import glob
import json
import os

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select

from src.core.utils import classify_serve_type, generate_player_heatmap
from src.db.utils import SessionDep, save_heatmap_data_in_db
from src.dependencies.auth import AuthContext, get_auth_context
from src.models.ball_track import BallTrack
from src.models.bounces import Bounces
from src.models.direction_change_indices import DirectionChangeIndices
from src.models.homography_matrices import HomographyMatrices
from src.models.player_heatmap_data import PlayerHeatmapData
from src.models.player_positions import PlayerPositions
from src.models.speed import Speed
from src.models.thumbnail import Thumbnail
from src.models.user import UserRole
from src.models.video_paths import VideoPaths
from src.schemas.ball_track import BallTrackSchema
from src.schemas.bounces import BouncesSchema
from src.schemas.direction_change_indices import DirectionChangeIndicesSchema
from src.schemas.player_positions import PlayerPositionsSchema
from src.schemas.thumbnail import ThumbnailSchema
from src.schemas.video_paths import VideoPathsSchema

router = APIRouter(tags=["stats"])


def _ensure_admin(auth_ctx: AuthContext) -> None:
    if auth_ctx.role != UserRole.ADMIN.value:
        raise HTTPException(status_code=403, detail="Access denied")


@router.get("/get_video_paths/{task_id}")
async def get_video_paths(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    if is_api:
        _ensure_admin(auth_ctx)
    statement = select(VideoPaths).where(VideoPaths.task_id == task_id)
    video_paths = session.exec(statement).first()
    if video_paths is None:
        return None if not is_api else {"success": True, "message": "Video paths not found"}

    video_paths_dict = VideoPathsSchema.model_validate(video_paths).model_dump()
    return video_paths_dict if not is_api else {"success": True, "data": video_paths_dict}


@router.get("/get_speed_stats/{task_id}")
async def get_speed_stats(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    if is_api:
        _ensure_admin(auth_ctx)
    statement = select(Speed).where(Speed.task_id == task_id)
    speed_stats = session.exec(statement).first()

    if speed_stats is None:
        return None if not is_api else {"success": True, "message": "Speed stats not found"}

    data = json.loads(speed_stats.speeds)
    return data if not is_api else {"success": True, "data": data}


@router.get("/get_ball_track/{task_id}")
async def get_ball_track(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    if is_api:
        _ensure_admin(auth_ctx)
    statement = select(BallTrack).where(BallTrack.task_id == task_id)
    ball_track = session.exec(statement).first()
    if ball_track is None:
        return None if not is_api else {"success": True, "message": "Ball track not found"}

    ball_track.ball_track = json.loads(ball_track.ball_track)
    ball_track_dict = BallTrackSchema.model_validate(ball_track).model_dump()
    return ball_track_dict if not is_api else {"success": True, "data": ball_track_dict}


@router.get("/get_bounces/{task_id}")
async def get_bounces(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    if is_api:
        _ensure_admin(auth_ctx)
    statement = select(Bounces).where(Bounces.task_id == task_id)
    bounces = session.exec(statement).first()
    if bounces is None:
        return None if not is_api else {"success": True, "message": "Bounces not found"}

    bounces.bounces = json.loads(bounces.bounces)
    bounces_dict = BouncesSchema.model_validate(bounces).model_dump()
    return bounces_dict if not is_api else {"success": True, "data": bounces_dict}


@router.get("/get_direction_change_indices/{task_id}")
async def get_direction_change_indices_api(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    if is_api:
        _ensure_admin(auth_ctx)
    statement = select(DirectionChangeIndices).where(DirectionChangeIndices.task_id == task_id)
    direction_change_indices = session.exec(statement).first()
    if direction_change_indices is None:
        return None if not is_api else {"success": True, "message": "Direction change indices not found"}

    direction_change_indices.direction_change_indices = json.loads(direction_change_indices.direction_change_indices)
    direction_change_indices_dict = DirectionChangeIndicesSchema.model_validate(direction_change_indices).model_dump()
    return direction_change_indices_dict if not is_api else {"success": True, "data": direction_change_indices_dict}


@router.get("/get_player_positions/{task_id}")
async def get_player_positions(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    if is_api:
        _ensure_admin(auth_ctx)
    statement = select(PlayerPositions).where(PlayerPositions.task_id == task_id)
    player_positions = session.exec(statement).first()
    if player_positions is None:
        return None if not is_api else {"success": True, "message": "Player positions not found"}

    player_positions.positions = json.loads(player_positions.positions)
    player_positions_dict = PlayerPositionsSchema.model_validate(player_positions).model_dump()
    return player_positions_dict if not is_api else {"success": True, "data": player_positions_dict}


@router.get("/thumbnail/{task_id}")
async def get_thumbnail(
    task_id: int,
    session: SessionDep,
    is_api: bool = True,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    if is_api:
        _ensure_admin(auth_ctx)
    statement = select(Thumbnail).where(Thumbnail.task_id == task_id)
    thumbnail = session.exec(statement).first()
    if thumbnail is None:
        return None if not is_api else {"success": True, "message": "Thumbnail not found"}

    thumbnail_dict = ThumbnailSchema.model_validate(thumbnail).model_dump()
    thumbnail_dict = {
        "id": str(thumbnail.id),
        "thumbnail_path": thumbnail.thumbnail_path,
        "created_at": thumbnail.created_at,
        "updated_at": thumbnail.updated_at,
    }
    return thumbnail_dict if not is_api else {"success": True, "data": thumbnail_dict}


@router.get("/all-stats/{task_id}")
async def get_all_stats(
    task_id: int,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    _ensure_admin(auth_ctx)
    video_paths = await get_video_paths(task_id, session, is_api=False, auth_ctx=auth_ctx)
    speed_stats = await get_speed_stats(task_id, session, is_api=False, auth_ctx=auth_ctx)
    ball_track = await get_ball_track(task_id, session, is_api=False, auth_ctx=auth_ctx)
    bounces = await get_bounces(task_id, session, is_api=False, auth_ctx=auth_ctx)
    direction_change_indices = await get_direction_change_indices_api(task_id, session, is_api=False, auth_ctx=auth_ctx)
    player_positions = await get_player_positions(task_id, session, is_api=False, auth_ctx=auth_ctx)
    thumbnail = await get_thumbnail(task_id, session, is_api=False, auth_ctx=auth_ctx)

    from src.api.tasks import get_task_progress

    progress = get_task_progress(task_id, session, is_api=False, auth_ctx=auth_ctx)

    return {
        "success": True,
        "data": {
            "video_paths": video_paths,
            "speed_stats": speed_stats,
            "ball_track": ball_track,
            "bounces": bounces,
            "direction_change_indices": direction_change_indices,
            "player_positions": player_positions,
            "thumbnail": thumbnail,
        },
        "progress": progress,
    }


@router.get("/serve_stats/{task_id}")
async def get_serve_stats(
    task_id: int,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    _ensure_admin(auth_ctx)
    bounces_row = session.exec(select(Bounces).where(Bounces.task_id == task_id)).first()
    if bounces_row is None:
        return {"success": True, "message": "Bounces not found", "data": {"serves": []}}
    bounces_data = json.loads(bounces_row.bounces)

    dci_row = session.exec(select(DirectionChangeIndices).where(DirectionChangeIndices.task_id == task_id)).first()
    if dci_row is None:
        return {"success": True, "message": "Direction change indices not found", "data": {"serves": []}}
    dci_data = json.loads(dci_row.direction_change_indices)
    dc_frames_sorted = sorted(int(k) for k in dci_data.keys())

    ball_track_row = session.exec(select(BallTrack).where(BallTrack.task_id == task_id)).first()
    if ball_track_row is None:
        return {"success": True, "message": "Ball track not found", "data": {"serves": []}}
    ball_track_data = json.loads(ball_track_row.ball_track)

    serves = []
    for frame_str, bounce_info in bounces_data.items():
        if not bounce_info.get("serve", False):
            continue
        bounce_frame = int(frame_str)
        bounce_position = bounce_info["position"]
        origin_frame = None
        for dc_frame in reversed(dc_frames_sorted):
            if dc_frame < bounce_frame:
                origin_frame = dc_frame
                break
        if origin_frame is None:
            continue

        origin_position = ball_track_data.get(str(origin_frame))
        if origin_position is None:
            continue

        track_segment = []
        for f in range(origin_frame, bounce_frame + 1):
            point = ball_track_data.get(str(f))
            if point is not None:
                track_segment.append(point)

        serve_type = "unknown"
        if bounce_position and bounce_position[0] is not None and bounce_position[1] is not None:
            serve_type = classify_serve_type(bounce_position[0], bounce_position[1])

        serves.append(
            {
                "bounce_frame": bounce_frame,
                "bounce_position": bounce_position,
                "origin_frame": origin_frame,
                "origin_position": origin_position,
                "ball_track": track_segment,
                "serve_type": serve_type,
            },
        )

    serves.sort(key=lambda s: s["bounce_frame"])
    return {"success": True, "data": {"serves": serves}}


@router.get("/player_heatmaps/{task_id}")
async def get_player_heatmaps(
    task_id: int,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    _ensure_admin(auth_ctx)
    heatmap_top_path = f"output/output_{task_id}_*_heatmap_top.png"
    heatmap_bottom_path = f"output/output_{task_id}_*_heatmap_bottom.png"

    heatmap_top_files = glob.glob(heatmap_top_path)
    heatmap_bottom_files = glob.glob(heatmap_bottom_path)
    if len(heatmap_top_files) == 0 or len(heatmap_bottom_files) == 0:
        return {
            "success": True,
            "data": {
                "player_top_heatmap": None,
                "player_bottom_heatmap": None,
            },
        }

    heatmap_top_file = max(heatmap_top_files, key=os.path.getctime)
    heatmap_bottom_file = max(heatmap_bottom_files, key=os.path.getctime)
    return {
        "success": True,
        "data": {
            "player_top_heatmap": f"/{heatmap_top_file}",
            "player_bottom_heatmap": f"/{heatmap_bottom_file}",
        },
    }


def _foot_from_bbox(bbox):
    if not bbox or len(bbox) < 4:
        return None
    return ((float(bbox[0]) + float(bbox[2])) / 2, float(bbox[3]))


@router.post("/player_heatmaps/{task_id}/recreate")
async def recreate_player_heatmaps(
    task_id: int,
    session: SessionDep,
    auth_ctx: AuthContext = Depends(get_auth_context),
) -> object:
    _ensure_admin(auth_ctx)
    heatmap_top_path = f"output/output_{task_id}_heatmap_top.png"
    heatmap_bottom_path = f"output/output_{task_id}_heatmap_bottom.png"

    heatmap_row = session.exec(select(PlayerHeatmapData).where(PlayerHeatmapData.task_id == task_id)).first()
    if heatmap_row is not None:
        top_points = heatmap_row.top_points if isinstance(heatmap_row.top_points, list) else json.loads(heatmap_row.top_points)
        bottom_points = heatmap_row.bottom_points if isinstance(heatmap_row.bottom_points, list) else json.loads(heatmap_row.bottom_points)
        top_tuples = [tuple(p) for p in top_points]
        bottom_tuples = [tuple(p) for p in bottom_points]
    else:
        pos_row = session.exec(select(PlayerPositions).where(PlayerPositions.task_id == task_id)).first()
        hom_row = session.exec(select(HomographyMatrices).where(HomographyMatrices.task_id == task_id)).first()
        if pos_row is None or hom_row is None:
            raise HTTPException(status_code=404, detail="Heatmap data not found; need processed video with player positions and court detection.")
        positions = pos_row.positions if isinstance(pos_row.positions, dict) else json.loads(pos_row.positions)
        matrices = hom_row.matrices if isinstance(hom_row.matrices, list) else json.loads(hom_row.matrices)
        n = min(len(positions), len(matrices))
        top_court_points = []
        bottom_court_points = []
        for i in range(n):
            H = matrices[i]
            if H is None:
                continue
            H = np.array(H, dtype=np.float32)
            frame_pos = positions.get(str(i))
            if not frame_pos:
                continue
            for key, points_list in [("top", top_court_points), ("bottom", bottom_court_points)]:
                bbox = frame_pos.get(key)
                foot = _foot_from_bbox(bbox)
                if foot is None:
                    continue
                try:
                    pt = np.array([[foot]], dtype=np.float32)
                    court_pt = cv2.perspectiveTransform(pt, H)
                    points_list.append((float(court_pt[0, 0, 0]), float(court_pt[0, 0, 1])))
                except Exception:
                    continue
        top_tuples = top_court_points
        bottom_tuples = bottom_court_points
        if top_tuples or bottom_tuples:
            save_heatmap_data_in_db(task_id, top_tuples, bottom_tuples)

    heatmap_top = generate_player_heatmap(top_tuples)
    heatmap_bottom = generate_player_heatmap(bottom_tuples)
    cv2.imwrite(heatmap_top_path, heatmap_top)
    cv2.imwrite(heatmap_bottom_path, heatmap_bottom)

    return {
        "success": True,
        "data": {
            "player_top_heatmap": f"/output/output_{task_id}_heatmap_top.png",
            "player_bottom_heatmap": f"/output/output_{task_id}_heatmap_bottom.png",
        },
    }
