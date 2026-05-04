from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from sqlmodel import Session, select

from src.core.court_reference import CourtReference
from src.core.utils import classify_serve_type, generate_player_heatmap, serve_player_from_bounce_court
from src.db.utils import save_rally_analysis_in_db
from src.models.background_task import BackgroundTask
from src.models.ball_track import BallTrack
from src.models.bounces import Bounces
from src.models.direction_change_indices import DirectionChangeIndices
from src.models.homography_matrices import HomographyMatrices
from src.models.player_positions import PlayerPositions
from src.models.rally_analysis import RallyAnalysis
from src.models.rally_stats import RallyStats
from src.models.speed import Speed
from src.models.thumbnail import Thumbnail
from src.models.video_paths import VideoPaths
from src.services.stats_players import build_player_displays

ANALYSIS_SCHEMA_VERSION = 1
COURT = CourtReference()
NET_Y = float(COURT.net[0][1])
SERVICE_LEFT_X = float(COURT.left_inner_line[0][0])
SERVICE_RIGHT_X = float(COURT.right_inner_line[0][0])
SERVICE_CENTER_X = float(COURT.middle_line[0][0])
TOP_SERVICE_LINE_Y = float(COURT.top_inner_line[0][1])
BOTTOM_SERVICE_LINE_Y = float(COURT.bottom_inner_line[0][1])


def load_json_maybe(raw: Any) -> Any:
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _normalize_fps(fps_raw: Any) -> float:
    try:
        fps = float(fps_raw)
    except (TypeError, ValueError):
        return 30.0
    if fps <= 0 or fps > 240:
        return 30.0
    return fps


def _public_path(path: str | None) -> str | None:
    if not path:
        return None
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("/"):
        return normalized
    if normalized.startswith("uploads/") or normalized.startswith("output/"):
        return f"/{normalized}"
    return normalized


def _frame_time(frame: int, fps: float) -> float:
    return round(frame / float(fps), 3)


def _safe_point(point: Any) -> list[float | None] | None:
    if point is None:
        return None
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    return [
        float(point[0]) if point[0] is not None else None,
        float(point[1]) if point[1] is not None else None,
    ]


def _float_bbox(bbox: Any) -> list[float] | None:
    if not bbox or len(bbox) < 4:
        return None
    return [float(v) for v in bbox[:4]]


def _foot_from_bbox(bbox: list[float] | None) -> tuple[float, float] | None:
    if not bbox or len(bbox) < 4:
        return None
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3]))


def _position_side(point: list[float | None] | None) -> str | None:
    if not point or point[1] is None:
        return None
    return "top" if float(point[1]) < NET_Y else "bottom"


def _player_from_side(side: str | None, opposite: bool = False) -> str | None:
    if side == "top":
        return "p2" if opposite else "p1"
    if side == "bottom":
        return "p1" if opposite else "p2"
    return None


def _extract_fps_and_total_frames(task: BackgroundTask, fallback_total_frames: int) -> tuple[float, int]:
    capture = cv2.VideoCapture(task.video_path)
    if not capture.isOpened():
        return 30.0, fallback_total_frames
    fps = _normalize_fps(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or fallback_total_frames
    capture.release()
    return fps, total_frames


def _sorted_int_keys(mapping: dict | None) -> list[int]:
    if not isinstance(mapping, dict):
        return []
    values: list[int] = []
    for key in mapping.keys():
        try:
            values.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(values)


def _track_events_in_range(track_map: dict, start: int, end: int, fps: float) -> list[dict]:
    events: list[dict] = []
    for frame in _sorted_int_keys(track_map):
        if frame < start or frame >= end:
            continue
        position = _safe_point(track_map.get(str(frame)))
        if not position or position[0] is None or position[1] is None:
            continue
        events.append(
            {
                "frame": frame,
                "time_sec": _frame_time(frame, fps),
                "position": position,
            },
        )
    return events


def _find_prev_direction_change(dc_frames_sorted: list[int], frame: int, fps: float) -> int | None:
    for dc_frame in reversed(dc_frames_sorted):
        if dc_frame >= frame:
            continue
        if frame - dc_frame > int(max(2.0 * fps, 15)):
            break
        return dc_frame
    return None


def _build_track_segment(track_map: dict, start_frame: int | None, end_frame: int, fps: float) -> list[dict]:
    if start_frame is None or start_frame > end_frame:
        return []
    segment: list[dict] = []
    for frame in range(start_frame, end_frame + 1):
        position = _safe_point(track_map.get(str(frame)))
        if not position or position[0] is None or position[1] is None:
            continue
        segment.append(
            {
                "frame": frame,
                "time_sec": _frame_time(frame, fps),
                "position": position,
            },
        )
    return segment


def _ensure_output_dir() -> None:
    os.makedirs("output", exist_ok=True)


def _build_heatmap(
    task_id: int,
    rally_id: str,
    player_key: str,
    points: list[tuple[float, float]],
) -> dict:
    _ensure_output_dir()
    filename = f"output/output_{task_id}_{rally_id}_{player_key}_heatmap.png"
    heatmap = generate_player_heatmap(points)
    cv2.imwrite(filename, heatmap)
    return {
        "image_path": _public_path(filename),
        "point_count": len(points),
    }


def _player_scope_template(display: dict, role: str) -> dict:
    return {
        "role": role,
        "display": display,
        "heatmap": {"image_path": None, "point_count": 0},
        "positions": [],
        "speed_stats": [],
        "serve_stats": {
            "serves": [],
            "summary": {
                "total": 0,
                "t": 0,
                "body": 0,
                "corner": 0,
                "wide": 0,
                "bucket": 0,
                "fault": 0,
            },
        },
        "court_analysis": {
            "ball_bounces": [],
            "ball_points": [],
            "ball_track": [],
            "forehand_backhand": {"forehand": 0, "backhand": 0, "unknown": 0},
        },
    }


def _resolve_assignment(
    original_player: str | None,
    *,
    bounce_position: list[float | None] | None = None,
    origin_position: list[float | None] | None = None,
    destination_position: list[float | None] | None = None,
    opposite_bounce_side: bool = False,
) -> tuple[str, float, str]:
    if original_player in {"p1", "p2"}:
        return original_player, 0.95, "legacy-detected"

    origin_side = _position_side(origin_position)
    if origin_side:
        player = _player_from_side(origin_side, opposite=False)
        if player:
            return player, 0.7, "origin-side"

    if opposite_bounce_side:
        bounce_side = _position_side(bounce_position)
        player = _player_from_side(bounce_side, opposite=True)
        if player:
            return player, 0.6, "opposite-bounce-side"

    destination_side = _position_side(destination_position)
    if destination_side:
        player = _player_from_side(destination_side, opposite=True)
        if player:
            return player, 0.55, "destination-side"

    return "p1", 0.01, "default-fallback"


def _serialize_speed_event(frame: int, speed_data: dict, player: str, confidence: float, method: str) -> dict:
    return {
        "bounce_frame": frame,
        "time_sec": round(float(speed_data.get("timestamp", 0.0)), 3),
        "speed_kmh": round(float(speed_data.get("speed", 0.0)), 3),
        "time_diff_sec": round(float(speed_data.get("time_diff", 0.0)), 3),
        "distance_m": round(float(speed_data.get("distance", 0.0)), 3),
        "shot_type": speed_data.get("shot_type", "unknown"),
        "player": player,
        "attribution_confidence": round(confidence, 3),
        "attribution_method": method,
    }


def _append_internal_record(
    internal_records: list[dict],
    *,
    rally_id: str,
    event_type: str,
    frame: int,
    original_player: str | None,
    assigned_player: str,
    confidence: float,
    method: str,
) -> None:
    if original_player == assigned_player and original_player in {"p1", "p2"}:
        return
    internal_records.append(
        {
            "rally_id": rally_id,
            "event_type": event_type,
            "frame": frame,
            "original_player": original_player or "unknown",
            "assigned_player": assigned_player,
            "confidence": round(confidence, 3),
            "method": method,
        },
    )


def _collect_player_positions(
    positions_map: dict,
    matrices: list,
    start: int,
    end: int,
    fps: float,
) -> tuple[list[dict], list[dict], list[tuple[float, float]], list[tuple[float, float]]]:
    p1_positions: list[dict] = []
    p2_positions: list[dict] = []
    p1_heatmap_points: list[tuple[float, float]] = []
    p2_heatmap_points: list[tuple[float, float]] = []
    for frame in range(start, end):
        frame_value = positions_map.get(str(frame))
        if not isinstance(frame_value, dict):
            continue
        for player_key, box_key, target_list, heatmap_points in [
            ("p1", "top", p1_positions, p1_heatmap_points),
            ("p2", "bottom", p2_positions, p2_heatmap_points),
        ]:
            bbox = _float_bbox(frame_value.get(box_key))
            if bbox is None:
                continue
            foot_point = _foot_from_bbox(bbox)
            court_point = None
            matrix = matrices[frame] if frame < len(matrices) else None
            if foot_point is not None and matrix is not None:
                try:
                    transformed = cv2.perspectiveTransform(
                        np.array([[foot_point]], dtype=np.float32),
                        np.array(matrix, dtype=np.float32),
                    )
                    court_point = (
                        float(transformed[0, 0, 0]),
                        float(transformed[0, 0, 1]),
                    )
                    heatmap_points.append(court_point)
                except Exception:
                    court_point = None
            target_list.append(
                {
                    "frame": frame,
                    "time_sec": _frame_time(frame, fps),
                    "bbox": bbox,
                    "court_position": list(court_point) if court_point is not None else None,
                },
            )
    return p1_positions, p2_positions, p1_heatmap_points, p2_heatmap_points


def _build_serve_summary(serves: list[dict]) -> dict:
    summary = {
        "total": len(serves),
        "t": 0,
        "body": 0,
        "corner": 0,
        "wide": 0,
        "bucket": 0,
        "fault": 0,
    }
    for serve in serves:
        serve_type = serve.get("serve_type")
        if serve_type in summary:
            summary[serve_type] += 1
    return summary


def _dedupe_track_events(track_events: list[dict]) -> list[dict]:
    deduped: dict[int, dict] = {}
    for event in track_events:
        frame = event.get("frame")
        if isinstance(frame, int):
            deduped[frame] = event
    return [deduped[frame] for frame in sorted(deduped)]


def build_rally_analysis_payload(
    *,
    session: Session,
    task: BackgroundTask,
    rally_list: list[dict],
    ball_track_map: dict,
    bounces_map: dict,
    direction_change_map: dict,
    positions_map: dict,
    speed_map: dict,
    matrices: list,
    fps: float,
    total_frames: int,
    output_path: str | None,
    minimap_path: str | None,
    thumbnail_path: str | None,
) -> tuple[dict, dict]:
    p1_display, p2_display = build_player_displays(session, task)
    players_meta = {
        "p1": {"role": "opponent", "display": p1_display},
        "p2": {"role": "owner", "display": p2_display},
    }
    processed_path = _public_path(output_path)
    minimap_public = _public_path(minimap_path)
    thumbnail_public = _public_path(thumbnail_path)
    source_public = _public_path(task.video_path)
    dc_frames_sorted = _sorted_int_keys(direction_change_map)
    rallies: list[dict] = []
    internal_records: list[dict] = []

    for rally in rally_list:
        scene_index = int(rally.get("scene_index", len(rallies)))
        rally_id = f"rally_{scene_index}"
        start_frame = int(rally.get("start_frame", 0))
        end_frame = int(rally.get("end_frame", start_frame))
        p1_scope = _player_scope_template(p1_display, "opponent")
        p2_scope = _player_scope_template(p2_display, "owner")
        player_scopes = {"p1": p1_scope, "p2": p2_scope}

        shared_track = _track_events_in_range(ball_track_map, start_frame, end_frame, fps)
        shared_direction_changes = _track_events_in_range(direction_change_map, start_frame, end_frame, fps)
        p1_positions, p2_positions, p1_heat_points, p2_heat_points = _collect_player_positions(
            positions_map,
            matrices,
            start_frame,
            end_frame,
            fps,
        )
        p1_scope["positions"] = p1_positions
        p2_scope["positions"] = p2_positions
        p1_scope["heatmap"] = _build_heatmap(task.id, rally_id, "p1", p1_heat_points)
        p2_scope["heatmap"] = _build_heatmap(task.id, rally_id, "p2", p2_heat_points)

        bounce_events: list[dict] = []
        for frame in _sorted_int_keys(bounces_map):
            if frame < start_frame or frame >= end_frame:
                continue
            bounce_info = bounces_map.get(str(frame)) or {}
            bounce_position = _safe_point(bounce_info.get("position"))
            is_serve = bool(bounce_info.get("serve"))
            speed_data = speed_map.get(str(frame)) or {}
            dc_frame = _find_prev_direction_change(dc_frames_sorted, frame, fps)
            origin_position = _safe_point(ball_track_map.get(str(dc_frame))) if dc_frame is not None else None
            track_segment = _build_track_segment(ball_track_map, dc_frame, frame, fps)

            original_player = speed_data.get("hitter") if speed_data else None
            if is_serve and original_player not in {"p1", "p2"}:
                server_guess = serve_player_from_bounce_court(
                    bounce_position[0] if bounce_position else None,
                    bounce_position[1] if bounce_position else None,
                )
                if server_guess in {"p1", "p2"}:
                    original_player = server_guess

            assigned_player, confidence, method = _resolve_assignment(
                original_player,
                bounce_position=bounce_position,
                origin_position=origin_position,
                destination_position=bounce_position,
                opposite_bounce_side=not is_serve,
            )
            serve_type = classify_serve_type(
                bounce_position[0] if bounce_position else None,
                bounce_position[1] if bounce_position else None,
            ) if is_serve else None

            bounce_event = {
                "frame": frame,
                "time_sec": _frame_time(frame, fps),
                "position": bounce_position,
                "is_serve": is_serve,
                "serve_type": serve_type,
                "player": assigned_player,
                "attribution_confidence": round(confidence, 3),
                "attribution_method": method,
            }
            bounce_events.append(bounce_event)

            player_scope = player_scopes[assigned_player]["court_analysis"]
            player_scope["ball_bounces"].append(bounce_event)
            if bounce_position:
                player_scope["ball_points"].append(
                    {
                        "frame": frame,
                        "time_sec": _frame_time(frame, fps),
                        "position": bounce_position,
                    },
                )
            player_scope["ball_track"].extend(track_segment)

            if is_serve:
                serve_event = {
                    "bounce_frame": frame,
                    "time_sec": _frame_time(frame, fps),
                    "bounce_position": bounce_position,
                    "origin_frame": dc_frame,
                    "origin_position": origin_position,
                    "ball_track": track_segment,
                    "serve_type": serve_type,
                    "player": assigned_player,
                    "attribution_confidence": round(confidence, 3),
                    "attribution_method": method,
                }
                player_scopes[assigned_player]["serve_stats"]["serves"].append(serve_event)
                _append_internal_record(
                    internal_records,
                    rally_id=rally_id,
                    event_type="serve",
                    frame=frame,
                    original_player=original_player,
                    assigned_player=assigned_player,
                    confidence=confidence,
                    method=method,
                )

        for frame in _sorted_int_keys(speed_map):
            if frame < start_frame or frame >= end_frame:
                continue
            speed_data = speed_map.get(str(frame)) or {}
            bounce_position = _safe_point((bounces_map.get(str(frame)) or {}).get("position"))
            assigned_player, confidence, method = _resolve_assignment(
                speed_data.get("hitter"),
                bounce_position=bounce_position,
                destination_position=bounce_position,
                opposite_bounce_side=True,
            )
            event = _serialize_speed_event(frame, speed_data, assigned_player, confidence, method)
            player_scopes[assigned_player]["speed_stats"].append(event)
            shot_type = event["shot_type"]
            if shot_type not in {"forehand", "backhand"}:
                shot_type = "unknown"
            player_scopes[assigned_player]["court_analysis"]["forehand_backhand"][shot_type] += 1
            _append_internal_record(
                internal_records,
                rally_id=rally_id,
                event_type="speed",
                frame=frame,
                original_player=speed_data.get("hitter"),
                assigned_player=assigned_player,
                confidence=confidence,
                method=method,
            )

        for player_key in ("p1", "p2"):
            scope = player_scopes[player_key]
            scope["serve_stats"]["summary"] = _build_serve_summary(scope["serve_stats"]["serves"])
            scope["court_analysis"]["ball_track"] = _dedupe_track_events(scope["court_analysis"]["ball_track"])

        rally_payload = {
            "rally_id": rally_id,
            "scene_index": scene_index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_sec": _frame_time(start_frame, fps),
            "end_time_sec": _frame_time(end_frame, fps),
            "duration_sec": round((end_frame - start_frame) / float(fps), 3),
            "video": {
                "source_video_path": source_public,
                "processed_video_path": processed_path,
                "minimap_path": minimap_public,
                "thumbnail_path": thumbnail_public,
                "playback_start_frame": start_frame,
                "playback_end_frame": end_frame,
                "playback_start_time_sec": _frame_time(start_frame, fps),
                "playback_end_time_sec": _frame_time(end_frame, fps),
            },
            "summary": {
                "shot_count": int(rally.get("shot_count", 0)),
                "bounce_count": len(bounce_events),
                "serve_count": sum(
                    len(player_scopes[player]["serve_stats"]["serves"]) for player in ("p1", "p2")
                ),
                "direction_change_count": len(shared_direction_changes),
            },
            "shared": {
                "ball_track": shared_track,
                "ball_bounces": bounce_events,
                "direction_changes": shared_direction_changes,
            },
            "players": player_scopes,
        }
        rallies.append(rally_payload)

    public_payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "task": {
            "id": int(task.id),
            "name": task.name,
            "status": task.status,
            "description": task.description,
        },
        "video": {
            "source_video_path": source_public,
            "processed_video_path": processed_path,
            "minimap_path": minimap_public,
            "thumbnail_path": thumbnail_public,
            "fps": round(float(fps), 3),
            "total_frames": int(total_frames),
        },
        "players": players_meta,
        "summary": {
            "total_rallies": len(rallies),
            "total_shots": sum(rally.get("summary", {}).get("shot_count", 0) for rally in rallies),
        },
        "rallies": rallies,
    }
    internal_payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "attribution_records": internal_records,
    }
    return public_payload, internal_payload


def rebuild_rally_analysis_from_legacy(session: Session, task_id: int) -> tuple[dict, dict]:
    task = session.exec(select(BackgroundTask).where(BackgroundTask.id == task_id)).first()
    if task is None:
        raise ValueError(f"Background task {task_id} not found")

    rally_row = session.exec(select(RallyStats).where(RallyStats.task_id == task_id)).first()
    if rally_row is None:
        raise ValueError(f"Rally stats for task {task_id} not found")
    rally_list = load_json_maybe(rally_row.rallies) or []

    ball_track_row = session.exec(select(BallTrack).where(BallTrack.task_id == task_id)).first()
    bounces_row = session.exec(select(Bounces).where(Bounces.task_id == task_id)).first()
    dci_row = session.exec(
        select(DirectionChangeIndices).where(DirectionChangeIndices.task_id == task_id),
    ).first()
    positions_row = session.exec(select(PlayerPositions).where(PlayerPositions.task_id == task_id)).first()
    speed_row = session.exec(select(Speed).where(Speed.task_id == task_id)).first()
    matrices_row = session.exec(
        select(HomographyMatrices).where(HomographyMatrices.task_id == task_id),
    ).first()
    video_paths_row = session.exec(select(VideoPaths).where(VideoPaths.task_id == task_id)).first()
    thumbnail_row = session.exec(select(Thumbnail).where(Thumbnail.task_id == task_id)).first()

    ball_track_map = load_json_maybe(ball_track_row.ball_track) if ball_track_row else {}
    bounces_map = load_json_maybe(bounces_row.bounces) if bounces_row else {}
    dci_map = load_json_maybe(dci_row.direction_change_indices) if dci_row else {}
    positions_map = load_json_maybe(positions_row.positions) if positions_row else {}
    speed_map = load_json_maybe(speed_row.speeds) if speed_row else {}
    matrices = load_json_maybe(matrices_row.matrices) if matrices_row else []

    total_frames_guess = 0
    if ball_track_map:
        total_frames_guess = max(_sorted_int_keys(ball_track_map) or [0]) + 1
    fps, total_frames = _extract_fps_and_total_frames(task, total_frames_guess)

    output_path = video_paths_row.output_path if video_paths_row else None
    minimap_path = video_paths_row.minimap_path if video_paths_row else None
    thumbnail_path = thumbnail_row.thumbnail_path if thumbnail_row else None

    return build_rally_analysis_payload(
        session=session,
        task=task,
        rally_list=rally_list,
        ball_track_map=ball_track_map,
        bounces_map=bounces_map,
        direction_change_map=dci_map,
        positions_map=positions_map,
        speed_map=speed_map,
        matrices=matrices or [],
        fps=fps,
        total_frames=total_frames,
        output_path=output_path,
        minimap_path=minimap_path,
        thumbnail_path=thumbnail_path,
    )


def rebuild_and_save_rally_analysis_from_legacy(session: Session, task_id: int) -> dict:
    public_payload, internal_payload = rebuild_rally_analysis_from_legacy(session, task_id)
    save_rally_analysis_in_db(
        task_id=task_id,
        public_payload=public_payload,
        internal_payload=internal_payload,
        schema_version=ANALYSIS_SCHEMA_VERSION,
    )
    return public_payload


def get_rally_analysis_row(session: Session, task_id: int) -> RallyAnalysis | None:
    return session.exec(select(RallyAnalysis).where(RallyAnalysis.task_id == task_id)).first()
