import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


CAMERA_QUADRANT_CHOICES = [
    "Front-left",
    "Front-right",
    "Back-left",
    "Back-right",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_all_steps(trajectory: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    # Legacy flat format
    for step in trajectory.get("steps", []) or []:
        if isinstance(step, dict):
            yield step

    # New branched format
    for step in trajectory.get("incremental_steps", []) or []:
        if isinstance(step, dict):
            yield step

    branch_groups = trajectory.get("branch_groups", {}) or {}
    if isinstance(branch_groups, dict):
        for group_steps in branch_groups.values():
            for step in group_steps or []:
                if isinstance(step, dict):
                    yield step


def get_step(trajectory: Dict[str, Any], step_number: Any) -> Optional[Dict[str, Any]]:
    for step in iter_all_steps(trajectory):
        if step.get("step") == step_number:
            return step
    return None


def get_step_by_class(trajectory: Dict[str, Any], question_class: str) -> Optional[Dict[str, Any]]:
    for step in iter_all_steps(trajectory):
        if step.get("question_class") == question_class:
            return step
    return None

def get_anchor_pixel_from_relation_meta(meta: Dict[str, Any]) -> Any:
    return meta.get("object_y_projected_pixel", meta.get("object_y_pixel"))

def idx_to_choice(choices: List[str], idx: Optional[int]) -> str:
    if idx is None:
        return "N/A"
    if idx < 0 or idx >= len(choices):
        return f"Invalid index: {idx}"
    return choices[idx]


def expected_visibility_label(step1: Optional[Dict[str, Any]]) -> str:
    if not step1:
        return "N/A"
    return idx_to_choice(step1.get("choices", []), step1.get("correct_idx"))


def resolve_video_path(video_arg: Path, video_id: str) -> Path:
    if video_arg.is_file():
        return video_arg

    if not video_arg.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_arg}")

    candidates = []
    for ext in (".mp4", ".MP4", ".avi", ".mov", ".mkv"):
        candidates.append(video_arg / f"{video_id}{ext}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find a video file for video_id='{video_id}' inside {video_arg}. "
        f"Expected one of: {[str(c.name) for c in candidates]}"
    )


def read_frame_at_time(cap: cv2.VideoCapture, time_sec: float, fps_hint: Optional[float] = None) -> Tuple[Optional[Any], int, float]:
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_hint if fps_hint and fps_hint > 0 else video_fps
    if not fps or fps <= 0:
        fps = 30.0

    frame_idx = max(0, int(round(time_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, frame_idx, fps
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, frame_idx, fps


def is_finite_xy(xy: Any) -> bool:
    return (
        isinstance(xy, (list, tuple))
        and len(xy) >= 2
        and xy[0] is not None
        and xy[1] is not None
        and math.isfinite(float(xy[0]))
        and math.isfinite(float(xy[1]))
    )


def point_inside_image(xy: Any, width: int, height: int) -> bool:
    if not is_finite_xy(xy):
        return False
    x, y = float(xy[0]), float(xy[1])
    return 0 <= x < width and 0 <= y < height


def draw_marker(ax, xy: Any, width: int, height: int, color: str, label: str, y_offset_axes: float = 0.04) -> None:
    if not is_finite_xy(xy):
        ax.text(
            0.02,
            y_offset_axes,
            f"{label}: no projected pixel",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
        )
        return

    x, y = float(xy[0]), float(xy[1])
    inside = point_inside_image((x, y), width, height)
    if inside:
        ax.add_patch(Circle((x, y), radius=18, fill=False, linewidth=2.5, color=color))
        ax.text(
            x + 12,
            y - 12,
            label,
            color=color,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor=color),
        )
    else:
        ax.text(
            0.02,
            y_offset_axes,
            f"{label}: off-screen ({x:.1f}, {y:.1f})",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor=color),
        )


def wrap_lines(lines: List[str]) -> str:
    return "\n".join(lines)


def build_text_panel(trajectory_id: str, traj: Dict[str, Any]) -> str:
    step1 = get_step(traj, 1)
    step2 = get_step(traj, 2)
    step3 = get_step(traj, "3") or get_step_by_class(traj, "oos_step3_last_placement")
    step4 = get_step(traj, 4) or get_step_by_class(traj, "oos_step3_fixture")

    step4a = get_step_by_class(traj, "oos_branch_object_camera_relative_position")
    step4b = get_step_by_class(traj, "oos_branch_object_object_relation")
    step4c = get_step_by_class(traj, "oos_branch_object_object_distance")

    s1_meta = (step1 or {}).get("answer_metadata", {}) or {}
    s2_meta = (step2 or {}).get("answer_metadata", {}) or {}
    s3_meta = (step3 or {}).get("answer_metadata", {}) or {}
    s4_meta = (step4 or {}).get("answer_metadata", {}) or {}
    s4a_meta = (step4a or {}).get("answer_metadata", {}) or {}
    s4b_meta = (step4b or {}).get("answer_metadata", {}) or {}
    s4c_meta = (step4c or {}).get("answer_metadata", {}) or {}

    anchor_pixel = s4b_meta.get("object_y_projected_pixel", s4b_meta.get("object_y_pixel", "N/A"))

    branch_groups = traj.get("branch_groups", {}) or {}
    branch_group_names = list(branch_groups.keys()) if isinstance(branch_groups, dict) else []
    branch_count = sum(len(v or []) for v in branch_groups.values()) if isinstance(branch_groups, dict) else 0

    lines = [
        f"Trajectory: {trajectory_id}",
        f"Video: {traj.get('video_id', 'N/A')}",
        f"Object: {traj.get('object_a_name', 'N/A')} ({traj.get('object_a_assoc_id', 'N/A')})",
        "",
        f"Query time: {traj.get('query_time_sec', 'N/A')} s",
        f"Clip: {traj.get('clip_start_time_sec', 'N/A')} -> {traj.get('clip_end_time_sec', 'N/A')} s",
        f"Stop reason: {traj.get('stop_reason', 'N/A')}",
        f"Incremental steps: {traj.get('num_incremental_steps', len(traj.get('incremental_steps', []) or []))}",
        f"Branch steps: {traj.get('num_branch_steps', branch_count)}",
        f"Branch groups: {branch_group_names if branch_group_names else 'N/A'}",
        "",
        "=== Incremental trunk ===",
        "Step 1: visibility",
        f"  expected answer: {expected_visibility_label(step1)}",
        f"  status={s1_meta.get('status')}",
        f"  is_visible={s1_meta.get('is_visible')}",
        f"  is_stably_visible={s1_meta.get('is_stably_visible')}",
        f"  projected_pixel={s1_meta.get('projected_pixel')}",
        f"  camera_coordinates={s1_meta.get('camera_coordinates')}",
        "",
        "Step 2: last visible",
        f"  sampled_last_visible_time_sec: {s2_meta.get('sampled_last_visible_time_sec', 'N/A')}",
        f"  sampled_last_visible_time_token: {s2_meta.get('sampled_last_visible_time_token', 'N/A')}",
        f"  projected_pixel: {s2_meta.get('projected_pixel', 'N/A')}",
        f"  fixture: {s2_meta.get('fixture', 'N/A')}",
        f"  reference_source: {s2_meta.get('reference_source', 'N/A')}",
        "",
        "Step 3: last placement",
        f"  last_placement_time_sec: {s3_meta.get('last_placement_time_sec', 'N/A')}",
        f"  last_placement_time_token: {s3_meta.get('last_placement_time_token', 'N/A')}",
        f"  projected_pixel: {s3_meta.get('projected_pixel', 'N/A')}",
        f"  fixture: {s3_meta.get('fixture', 'N/A')}",
        f"  reference_source: {s3_meta.get('reference_source', 'N/A')}",
        "",
        "Step 4: closest fixture to last placement",
        f"  correct fixture: {s4_meta.get('correct_fixture', 'N/A')}",
        f"  chosen label: {idx_to_choice((step4 or {}).get('choices', []), (step4 or {}).get('correct_idx'))}",
        f"  choices: {(step4 or {}).get('choices', [])}",
        f"  reference_time_sec: {s4_meta.get('reference_time_sec', 'N/A')}",
        "",
        "=== Branches ===",
        "Branch 5a: object-camera relative position",
        f"  chosen label: {idx_to_choice((step4a or {}).get('choices', []), (step4a or {}).get('correct_idx'))}",
        f"  camera_coordinates: {s4a_meta.get('camera_coordinates', 'N/A')}",
        f"  status: {s4a_meta.get('status', 'N/A')}",
        "",
        "Branch 5b: object-object relation",
        f"  chosen label: {idx_to_choice((step4b or {}).get('choices', []), (step4b or {}).get('correct_idx'))}",
        f"  anchor object: {s4b_meta.get('object_y_name', 'N/A')} ({s4b_meta.get('object_y_assoc_id', 'N/A')})",
        f"  anchor pixel: {anchor_pixel}",
        f"  object_x_world_coordinates: {s4b_meta.get('object_x_world_coordinates', 'N/A')}",
        f"  object_y_world_coordinates: {s4b_meta.get('object_y_world_coordinates', 'N/A')}",
        "",
        "Branch 5c: object-object distance",
        f"  chosen label: {idx_to_choice((step4c or {}).get('choices', []), (step4c or {}).get('correct_idx'))}",
        f"  distance_m: {s4c_meta.get('distance_m', 'N/A')}",
        f"  distance_bucket: {s4c_meta.get('distance_bucket', 'N/A')}",
        f"  relative vector: {s4c_meta.get('vector_object_x_relative_to_object_y', 'N/A')}",
    ]
    return wrap_lines(lines)


def draw_quadrant_overlay(ax, label: Optional[str], title: str, box_anchor=(0.72, 0.72)) -> None:
    if not label or label == "N/A":
        return

    ax.text(
        0.98, 0.98,
        f"{title}: {label}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
    )

    box_x, box_y = box_anchor
    box_w, box_h = 0.22, 0.22
    ax.plot([box_x + box_w / 2, box_x + box_w / 2], [box_y, box_y + box_h], transform=ax.transAxes, color="white", linewidth=1.5)
    ax.plot([box_x, box_x + box_w], [box_y + box_h / 2, box_y + box_h / 2], transform=ax.transAxes, color="white", linewidth=1.5)

    quadrant_centers = {
        "Front-left": (box_x + 0.25 * box_w, box_y + 0.75 * box_h),
        "Front-right": (box_x + 0.75 * box_w, box_y + 0.75 * box_h),
        "Back-left": (box_x + 0.25 * box_w, box_y + 0.25 * box_h),
        "Back-right": (box_x + 0.75 * box_w, box_y + 0.25 * box_h),
    }

    for name, (cx, cy) in quadrant_centers.items():
        prefix = "● " if name == label else ""
        ax.text(
            cx, cy, prefix + name.replace("-", "\n"),
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="black"),
        )


def visualize_one(
    trajectory_id: str,
    traj: Dict[str, Any],
    video_root: Path,
    output_dir: Path,
    fps_override: Optional[float],
    show: bool,
) -> Path:
    video_id = str(traj["video_id"])
    video_path = resolve_video_path(video_root, video_id)

    step1 = get_step(traj, 1)
    step2 = get_step(traj, 2)
    step3 = get_step(traj, "3") or get_step_by_class(traj, "oos_step3_last_placement")
    step4 = get_step(traj, 4) or get_step_by_class(traj, "oos_step3_fixture")

    step5a = get_step_by_class(traj, "oos_branch_object_camera_relative_position")
    step5b = get_step_by_class(traj, "oos_branch_object_object_relation")
    step5c = get_step_by_class(traj, "oos_branch_object_object_distance")

    step1_meta = (step1 or {}).get("answer_metadata", {}) or {}
    step2_meta = (step2 or {}).get("answer_metadata", {}) or {}
    step3_meta = (step3 or {}).get("answer_metadata", {}) or {}
    step5b_meta = (step5b or {}).get("answer_metadata", {}) or {}

    query_time = float(traj["query_time_sec"])
    last_visible_time = step2_meta.get("sampled_last_visible_time_sec")
    last_placement_time = step3_meta.get("last_placement_time_sec")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        query_frame, query_idx, fps_used = read_frame_at_time(cap, query_time, fps_override)

        last_visible_frame = None
        last_visible_idx = None
        if last_visible_time is not None:
            last_visible_frame, last_visible_idx, _ = read_frame_at_time(cap, float(last_visible_time), fps_override)

        last_placement_frame = None
        last_placement_idx = None
        if last_placement_time is not None:
            last_placement_frame, last_placement_idx, _ = read_frame_at_time(cap, float(last_placement_time), fps_override)
    finally:
        cap.release()

    if query_frame is None:
        raise RuntimeError(f"Failed to read query frame at {query_time}s from {video_path}")

    qh, qw = query_frame.shape[:2]
    lvh, lvw = last_visible_frame.shape[:2] if last_visible_frame is not None else (qh, qw)
    lph, lpw = last_placement_frame.shape[:2] if last_placement_frame is not None else (qh, qw)

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1])

    gs_left_top = gs[0, 0].subgridspec(1, 2, wspace=0.08)
    ax_last_visible = fig.add_subplot(gs_left_top[0, 0])
    ax_last_placement = fig.add_subplot(gs_left_top[0, 1])

    ax_query = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[:, 1])

    # Top-left: Step 2 last visible
    if last_visible_frame is not None:
        ax_last_visible.imshow(last_visible_frame)
        ax_last_visible.set_title(
            f"Step 2: last visible frame\n"
            f"t={last_visible_time}s | frame={last_visible_idx}"
        )
        draw_marker(ax_last_visible, step2_meta.get("projected_pixel"), lvw, lvh, "red", "last visible")
    else:
        ax_last_visible.text(0.5, 0.5, "No last-visible frame available", ha="center", va="center")
        ax_last_visible.set_title("Step 2: last visible frame")
    ax_last_visible.axis("off")

    # Top-right: Step 3 last placement
    if last_placement_frame is not None:
        ax_last_placement.imshow(last_placement_frame)
        ax_last_placement.set_title(
            f"Step 3: last placement frame\n"
            f"t={last_placement_time}s | frame={last_placement_idx}"
        )
        draw_marker(ax_last_placement, step3_meta.get("projected_pixel"), lpw, lph, "orange", "last placement")
    else:
        ax_last_placement.text(0.5, 0.5, "No last-placement frame available", ha="center", va="center")
        ax_last_placement.set_title("Step 3: last placement frame")
    ax_last_placement.axis("off")

    # Bottom-left: query frame + branches
    ax_query.imshow(query_frame)
    ax_query.set_title(
        f"Query frame with branch overlays\n"
        f"t={query_time}s | frame={query_idx} | fps≈{fps_used:.3f}"
    )

    # target at query time
    draw_marker(ax_query, step1_meta.get("projected_pixel"), qw, qh, "deepskyblue", "target", 0.04)

    # anchor object from relation branch
    anchor_pixel = get_anchor_pixel_from_relation_meta(step5b_meta)
    draw_marker(ax_query, anchor_pixel, qw, qh, "lime", "anchor", 0.10)

    cam_label = idx_to_choice((step5a or {}).get("choices", []), (step5a or {}).get("correct_idx"))
    rel_label = idx_to_choice((step5b or {}).get("choices", []), (step5b or {}).get("correct_idx"))
    distance_label = idx_to_choice((step5c or {}).get("choices", []), (step5c or {}).get("correct_idx"))

    draw_quadrant_overlay(ax_query, cam_label, "Branch 4a", box_anchor=(0.05, 0.72))
    draw_quadrant_overlay(ax_query, rel_label, "Branch 5b", box_anchor=(0.73, 0.72))

    ax_query.text(
        0.02, 0.98,
        f"Branch 5c distance: {distance_label}",
        transform=ax_query.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
    )
    ax_query.axis("off")

    # Right: text summary
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        build_text_panel(trajectory_id, traj),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle(
        f"OOS question check: {trajectory_id}\n"
        f"Incremental trunk (Steps 1-4) + branch overlays on query frame",
        fontsize=14,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{trajectory_id}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def choose_trajectories(
    trajectories: Dict[str, Dict[str, Any]],
    num_samples: int,
    seed: int,
    trajectory_ids: Optional[List[str]],
) -> List[Tuple[str, Dict[str, Any]]]:
    items = list(trajectories.items())

    if trajectory_ids:
        selected = []
        lookup = dict(items)
        for tid in trajectory_ids:
            if tid not in lookup:
                raise KeyError(f"Trajectory id not found: {tid}")
            selected.append((tid, lookup[tid]))
        return selected

    rng = random.Random(seed)
    if num_samples >= len(items):
        return items
    return rng.sample(items, num_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize staged OOS questions for legacy and branched VQA formats."
    )
    parser.add_argument("--questions", type=Path, required=True, help="Path to staged_oos_trajectories.json")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to a single video file, or a directory containing files named like <video_id>.mp4",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("oos_question_visualizations"))
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trajectory_ids",
        nargs="+",
        default=None,
        help="Optional explicit trajectory ids to visualize",
    )
    parser.add_argument(
        "--fps_override",
        type=float,
        default=None,
        help="Optional FPS override if the video metadata FPS is wrong.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open each figure interactively while saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = load_json(args.questions)
    selected = choose_trajectories(
        trajectories=trajectories,
        num_samples=args.num_samples,
        seed=args.seed,
        trajectory_ids=args.trajectory_ids,
    )

    print(f"Loaded {len(trajectories)} trajectories from {args.questions}")
    print(f"Visualizing {len(selected)} trajectories")

    saved_paths = []
    for trajectory_id, traj in selected:
        out_path = visualize_one(
            trajectory_id=trajectory_id,
            traj=traj,
            video_root=args.video,
            output_dir=args.output_dir,
            fps_override=args.fps_override,
            show=args.show,
        )
        saved_paths.append(out_path)
        print(f"Saved: {out_path}")

    print("Done.")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
