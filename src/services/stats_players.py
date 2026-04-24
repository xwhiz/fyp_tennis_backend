"""Group per-task analytics for p1 (opponent / top) vs p2 (owner / bottom)."""

from __future__ import annotations

import glob
import json
import os
from typing import Any

from sqlmodel import Session, select

from src.models.background_task import BackgroundTask
from src.models.user import User
from src.utils.at_tag import display_at_tag


def load_json_maybe(raw: Any) -> Any:
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def split_speed_by_hitter(speeds: dict | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not speeds:
        return {}, {}, {}
    p1: dict[str, Any] = {}
    p2: dict[str, Any] = {}
    unassigned: dict[str, Any] = {}
    for k, v in speeds.items():
        if not isinstance(v, dict):
            continue
        key = str(k)
        h = v.get("hitter")
        if h == "p1":
            p1[key] = v
        elif h == "p2":
            p2[key] = v
        else:
            unassigned[key] = v
    return p1, p2, unassigned


def split_positions_for_players(positions_inner: dict | None) -> tuple[dict | None, dict | None]:
    if not positions_inner:
        return None, None
    pos = positions_inner.get("positions") if isinstance(positions_inner, dict) else None
    if pos is None:
        return None, None
    if isinstance(pos, str):
        pos = json.loads(pos)
    if not isinstance(pos, dict):
        return None, None
    p1_frames: dict[str, Any] = {}
    p2_frames: dict[str, Any] = {}
    for frame_key, frame_val in pos.items():
        if not isinstance(frame_val, dict):
            continue
        p1_frames[str(frame_key)] = {"bbox": frame_val.get("top")}
        p2_frames[str(frame_key)] = {"bbox": frame_val.get("bottom")}
    return {"positions": p1_frames}, {"positions": p2_frames}


def heatmap_paths_for_task(task_id: int) -> tuple[str | None, str | None]:
    top_files = glob.glob(f"output/output_{task_id}_*_heatmap_top.png")
    bottom_files = glob.glob(f"output/output_{task_id}_*_heatmap_bottom.png")

    def pick(files: list[str]) -> str | None:
        if not files:
            return None
        return max(files, key=os.path.getctime)

    top = pick(top_files)
    bottom = pick(bottom_files)

    def url(p: str | None) -> str | None:
        return f"/{p}" if p else None

    return url(top), url(bottom)


def build_player_displays(session: Session, task: BackgroundTask) -> tuple[dict, dict]:
    p1: dict[str, Any] = {"kind": "unknown", "userId": None, "atTag": None, "name": None}
    if task.opponent_id:
        ou = session.exec(select(User).where(User.id == task.opponent_id)).first()
        if ou:
            p1 = {
                "kind": "user",
                "userId": ou.id,
                "atTag": display_at_tag(ou.at_tag),
                "name": f"{ou.first_name} {ou.last_name}".strip(),
            }
    p2: dict[str, Any] = {"kind": "unknown", "userId": None, "atTag": None, "name": None}
    owner = session.exec(select(User).where(User.id == task.owner_id)).first()
    if owner:
        p2 = {
            "kind": "user",
            "userId": owner.id,
            "atTag": display_at_tag(owner.at_tag),
            "name": f"{owner.first_name} {owner.last_name}".strip(),
        }
    return p1, p2


def serve_summary_counts(serves: list[dict]) -> dict[str, int]:
    c = {"p1": 0, "p2": 0, "unknown": 0}
    for s in serves:
        srv = s.get("server", "unknown")
        if srv in c:
            c[srv] += 1
        else:
            c["unknown"] += 1
    return c
