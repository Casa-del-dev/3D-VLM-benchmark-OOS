"""Loader for per-frame camera pose and gaze from the Intermediate_data dumps.

`framewise_info.jsonl` contains one JSON record per video frame. Many frames
have null poses — we still return them so that `frame_index` stays aligned
with the list index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class FrameInfo:
    frame_index: int
    camera_position: Vec3 | None       # translation column of T_world_device
    gaze_direction: Vec3 | None        # unit vector in world frame
    gaze_origin: Vec3 | None           # for convenience, same as camera_position


def _extract_translation(T: Sequence[Sequence[float]] | None) -> Vec3 | None:
    if T is None:
        return None
    # T_world_device is 3x4: rows are [R | t]. Translation is last column.
    return (float(T[0][3]), float(T[1][3]), float(T[2][3]))


def load_framewise(framewise_path: Path) -> List[FrameInfo]:
    if not framewise_path.exists():
        return []
    frames: list[FrameInfo] = []
    with open(framewise_path) as f:
        for line in f:
            rec = json.loads(line)
            cam = _extract_translation(rec.get("T_world_device"))
            gdir = rec.get("gaze_direction_in_world")
            frames.append(
                FrameInfo(
                    frame_index=int(rec["frame_index"]),
                    camera_position=cam,
                    gaze_direction=tuple(gdir) if gdir is not None else None,  # type: ignore[arg-type]
                    gaze_origin=cam,
                )
            )
    return frames
