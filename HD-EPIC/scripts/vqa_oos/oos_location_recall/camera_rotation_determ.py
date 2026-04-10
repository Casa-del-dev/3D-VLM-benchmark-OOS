from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

from in_view_determination import load_jsonl, DEFAULT_INTERMEDIATE_ROOT


@dataclass(frozen=True)
class CameraDirectionAnswer:
    choices: list[str]
    correct_idx: int
    signed_rotation_deg: float
    direction: str


def _normalize_angle_deg(x: float) -> float:
    while x > 180.0:
        x -= 360.0
    while x < -180.0:
        x += 360.0
    return x


def _nearest_pose_row(rows: list[dict[str, Any]], time_sec: float, fps: float = 30.0) -> dict[str, Any]:
    target_frame = int(round(float(time_sec) * fps))
    best = None
    best_dist = float("inf")

    for r in rows:
        if r.get("frame_index") is None:
            continue
        fi = int(r["frame_index"])
        d = abs(fi - target_frame)
        if d < best_dist:
            best = r
            best_dist = d

    if best is None:
        raise ValueError(f"No pose row found near time {time_sec}")
    return best


def _rotation_matrix_from_row(row: dict[str, Any]) -> list[list[float]]:
    T = row.get("T_world_device")
    if T is None:
        raise KeyError("Missing T_world_device in framewise row")
    if len(T) != 3 or any(len(r) != 4 for r in T):
        raise ValueError("T_world_device must be 3x4")

    return [
        [float(T[0][0]), float(T[0][1]), float(T[0][2])],
        [float(T[1][0]), float(T[1][1]), float(T[1][2])],
        [float(T[2][0]), float(T[2][1]), float(T[2][2])],
    ]


def _heading_deg_from_row(row: dict[str, Any]) -> float:
    """
    稳定版：用 device 的一个固定轴在水平面上的投影来定义 heading。
    这里先用 -Z 轴作为 forward 候选。
    如果你后面目检发现左右反了，只需要改符号。
    """
    R = _rotation_matrix_from_row(row)

    fx = -R[0][2]
    fy = -R[1][2]

    horiz_norm = math.hypot(fx, fy)
    if horiz_norm < 1e-8:
        raise ValueError("Forward vector horizontal projection is too small")

    return math.degrees(math.atan2(fy, fx))


def load_camera_signed_rotation_deg(
    video_id: str,
    start_time_sec: float,
    end_time_sec: float,
    annotations_root: Path,
    intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT,
) -> float:
    participant_id = video_id.split("-")[0]
    framewise_path = annotations_root / intermediate_root / participant_id / video_id / "framewise_info.jsonl"
    rows = load_jsonl(framewise_path)

    start_row = _nearest_pose_row(rows, start_time_sec)
    end_row = _nearest_pose_row(rows, end_time_sec)

    heading_start = _heading_deg_from_row(start_row)
    heading_end = _heading_deg_from_row(end_row)

    return _normalize_angle_deg(heading_end - heading_start)


def determine_camera_direction_answer(
    video_id: str,
    start_time_sec: float,
    end_time_sec: float,
    annotations_root: Path,
    no_motion_thresh_deg: float = 8.0,
) -> CameraDirectionAnswer:
    signed_deg = load_camera_signed_rotation_deg(
        video_id=video_id,
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        annotations_root=annotations_root,
    )

    choices = ["left", "right", "no significant rotation"]

    if abs(signed_deg) < no_motion_thresh_deg:
        direction = "no significant rotation"
        correct_idx = 2
    elif signed_deg > 0:
        direction = "left"
        correct_idx = 0
    else:
        direction = "right"
        correct_idx = 1

    return CameraDirectionAnswer(
        choices=choices,
        correct_idx=correct_idx,
        signed_rotation_deg=signed_deg,
        direction=direction,
    )