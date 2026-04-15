import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

CAMERA_POSE_CHANGE_CHOICES = ["No change", "Rotated left", "Rotated right", "Rotated back"]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_step(trajectory: Dict[str, Any], step_number: int) -> Optional[Dict[str, Any]]:
    for step in trajectory.get("steps", []):
        if step.get("step") == step_number:
            return step
    return None


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


def expected_camera_motion_label(step4: Optional[Dict[str, Any]]) -> str:
    if not step4:
        return "N/A"
    return idx_to_choice(step4.get("choices", CAMERA_POSE_CHANGE_CHOICES), step4.get("correct_idx"))


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


def draw_marker(ax, xy: Any, width: int, height: int, color: str, label: str) -> None:
    if not is_finite_xy(xy):
        ax.text(
            0.02,
            0.04,
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
            0.04,
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
    step3 = get_step(traj, 3)
    step4 = get_step(traj, 4)
    step5 = get_step(traj, 5)   # NEW

    s1_meta = (step1 or {}).get("answer_metadata", {})
    s2_meta = (step2 or {}).get("answer_metadata", {})
    s3_meta = (step3 or {}).get("answer_metadata", {})
    s4_meta = (step4 or {}).get("answer_metadata", {})
    s5_meta = (step5 or {}).get("answer_metadata", {})   # NEW

    lines = [
        f"Trajectory: {trajectory_id}",
        f"Video: {traj.get('video_id', 'N/A')}",
        f"Object: {traj.get('object_a_name', 'N/A')} ({traj.get('object_a_assoc_id', 'N/A')})",
        "",
        f"Query time: {traj.get('query_time_sec', 'N/A')} s",
        f"Clip: {traj.get('clip_start_time_sec', 'N/A')} -> {traj.get('clip_end_time_sec', 'N/A')} s",
        f"Stop reason: {traj.get('stop_reason', 'N/A')}",
        "",
        "Step 1: visibility",
        f"  expected answer: {expected_visibility_label(step1)}",
        f"  metadata in_view={s1_meta.get('in_view')} queryable={s1_meta.get('queryable')} status={s1_meta.get('status')}",
        f"  projected_pixel={s1_meta.get('projected_pixel')}",
        "",
        "Step 2: last visible",
        f"  sampled_last_visible_time_sec: {s2_meta.get('sampled_last_visible_time_sec', 'N/A')}",
        f"  projected_pixel: {s2_meta.get('projected_pixel')}",
        f"  status: {s2_meta.get('status', 'N/A')}",
        "",
        "Step 3: closest fixture",
        f"  correct fixture: {s3_meta.get('correct_fixture', 'N/A')}",
        f"  chosen label: {idx_to_choice((step3 or {}).get('choices', []), (step3 or {}).get('correct_idx'))}",
        f"  choices: {(step3 or {}).get('choices', [])}",
        f"  skipped: {(step3 or {}).get('skipped', False)}",
        f"  error: {s3_meta.get('error', '')}",
        "",
        "Step 4: camera pose change",
        f"  expected answer: {expected_camera_motion_label(step4)}",
        f"  yaw_delta_deg: {s4_meta.get('yaw_delta_deg', 'N/A')}",
        f"  thresholds: no_change<={s4_meta.get('no_change_threshold_deg', 'N/A')}  rotated_back>={s4_meta.get('rotated_back_threshold_deg', 'N/A')}",
        "",
        "Step 5: camera quadrant",   # NEW
        f"  chosen label: {idx_to_choice((step5 or {}).get('choices', []), (step5 or {}).get('correct_idx'))}",
        f"  metadata label: {s5_meta.get('label', 'N/A')}",
        f"  camera_coordinates: {s5_meta.get('camera_coordinates', 'N/A')}",
        f"  skipped: {(step5 or {}).get('skipped', False)}",
        f"  debug: {s5_meta.get('debug', {})}",
    ]
    return wrap_lines(lines)


def draw_quadrant_overlay(ax, label: Optional[str]) -> None:
    if not label:
        return

    ax.text(
        0.98, 0.98,
        f"Step 5: {label}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
    )

    # simple 2x2 direction box
    box_x, box_y, box_w, box_h = 0.72, 0.72, 0.22, 0.22
    ax.plot([box_x + box_w/2, box_x + box_w/2], [box_y, box_y + box_h],
            transform=ax.transAxes, color="white", linewidth=1.5)
    ax.plot([box_x, box_x + box_w], [box_y + box_h/2, box_y + box_h/2],
            transform=ax.transAxes, color="white", linewidth=1.5)

    quadrant_centers = {
        "Front-left":  (box_x + 0.25 * box_w, box_y + 0.75 * box_h),
        "Front-right": (box_x + 0.75 * box_w, box_y + 0.75 * box_h),
        "Back-left":   (box_x + 0.25 * box_w, box_y + 0.25 * box_h),
        "Back-right":  (box_x + 0.75 * box_w, box_y + 0.25 * box_h),
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
    step5 = get_step(traj, 5)
    step1_meta = (step1 or {}).get("answer_metadata", {})
    step2_meta = (step2 or {}).get("answer_metadata", {})
    step5_meta = (step5 or {}).get("answer_metadata", {})

    query_time = float(traj["query_time_sec"])
    last_visible_time = step2_meta.get("sampled_last_visible_time_sec")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        query_frame, query_idx, fps_used = read_frame_at_time(cap, query_time, fps_override)
        last_frame = None
        last_idx = None
        if last_visible_time is not None:
            last_frame, last_idx, _ = read_frame_at_time(cap, float(last_visible_time), fps_override)
    finally:
        cap.release()

    if query_frame is None:
        raise RuntimeError(f"Failed to read query frame at {query_time}s from {video_path}")

    qh, qw = query_frame.shape[:2]

    if last_frame is not None:
        lh, lw = last_frame.shape[:2]
    else:
        lh, lw = qh, qw

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    if last_frame is not None:
        axes[0].imshow(last_frame)
        axes[0].set_title(f"Last visible frame\n t={last_visible_time}s | frame={last_idx}")
        draw_marker(axes[0], step2_meta.get("projected_pixel"), lw, lh, "red", "last visible")
    else:
        axes[0].text(0.5, 0.5, "No last-visible frame available", ha="center", va="center")
        axes[0].set_title("Last visible frame")
    axes[0].axis("off")

    axes[1].imshow(query_frame)
    axes[1].set_title(f"Query frame\n t={query_time}s | frame={query_idx} | fps≈{fps_used:.3f}")
    draw_marker(axes[1], step1_meta.get("projected_pixel"), qw, qh, "deepskyblue", "query projected")
    draw_quadrant_overlay(axes[1], step5_meta.get("label"))
    axes[1].axis("off")

    axes[2].axis("off")
    axes[2].text(
        0.0,
        1.0,
        build_text_panel(trajectory_id, traj),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle(f"OOS question check: {trajectory_id}", fontsize=14)
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
        description="Visualize staged OOS questions to inspect visibility, last-visible position, fixture label, camera motion, and step-5 direction quadrant."    )
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
        help="Optional explicit trajectory ids to visualize, e.g. oos_staged_h2p0_1 oos_staged_h2p0_7",
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
