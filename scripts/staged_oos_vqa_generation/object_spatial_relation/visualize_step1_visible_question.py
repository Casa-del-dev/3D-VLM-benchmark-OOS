import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_all_steps(trajectory: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Support both flat and incremental trajectory formats."""
    for step in trajectory.get("steps", []) or []:
        if isinstance(step, dict):
            yield step

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


def idx_to_choice(choices: List[str], idx: Optional[int]) -> str:
    if idx is None:
        return "N/A"
    if idx < 0 or idx >= len(choices):
        return f"Invalid index: {idx}"
    return str(choices[idx])


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


def read_frame_at_time(
    cap: cv2.VideoCapture,
    time_sec: float,
    fps_hint: Optional[float] = None,
) -> Tuple[Optional[Any], int, float]:
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


def draw_marker(
    ax,
    xy: Any,
    width: int,
    height: int,
    color: str,
    label: str,
    y_offset_axes: float = 0.04,
) -> None:
    if not is_finite_xy(xy):
        ax.text(
            0.02,
            y_offset_axes,
            f"{label}: no projected pixel",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
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
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor=color),
        )
    else:
        ax.text(
            0.02,
            y_offset_axes,
            f"{label}: off-screen ({x:.1f}, {y:.1f})",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor=color),
        )


def normalize_trajectory_container(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Accept either {trajectory_id: traj} or a list of trajectory dicts."""
    if isinstance(raw, dict):
        # Standard format.
        if all(isinstance(v, dict) for v in raw.values()):
            return {str(k): v for k, v in raw.items()}
        # Possible wrapper format.
        for key in ("trajectories", "questions", "data"):
            if key in raw:
                return normalize_trajectory_container(raw[key])

    if isinstance(raw, list):
        out: Dict[str, Dict[str, Any]] = {}
        for i, traj in enumerate(raw):
            if not isinstance(traj, dict):
                continue
            tid = str(traj.get("trajectory_id") or traj.get("id") or f"trajectory_{i:06d}")
            out[tid] = traj
        return out

    raise TypeError("Unsupported question JSON format. Expected dict or list.")


def build_text_panel(trajectory_id: str, traj: Dict[str, Any]) -> str:
    step1 = get_step(traj, 1) or get_step_by_class(traj, "oos_step1_visibility")
    meta = (step1 or {}).get("answer_metadata", {}) or {}

    choices = (step1 or {}).get("choices", [])
    correct_idx = (step1 or {}).get("correct_idx")
    answer = expected_visibility_label(step1)

    lines = [
        f"Trajectory: {trajectory_id}",
        f"Video: {traj.get('video_id', 'N/A')}",
        f"Object: {traj.get('object_a_name', 'N/A')} ({traj.get('object_a_assoc_id', 'N/A')})",
        "",
        f"Query time: {traj.get('query_time_sec', 'N/A')} s",
        f"Query time in clip: {traj.get('query_time_in_clip_sec', 'N/A')} s",
        f"Clip: {traj.get('clip_start_time_sec', 'N/A')} -> {traj.get('clip_end_time_sec', 'N/A')} s",
        f"Clip duration: {traj.get('clip_duration_sec', 'N/A')} s",
        f"Stop reason: {traj.get('stop_reason', 'N/A')}",
        "",
        "=== Step 1: visibility ===",
        f"Question: {(step1 or {}).get('question', 'N/A')}",
        f"Choices: {choices}",
        f"Correct idx: {correct_idx}",
        f"Expected answer: {answer}",
        "",
        "=== Step 1 metadata ===",
        f"status: {meta.get('status', 'N/A')}",
        f"is_visible: {meta.get('is_visible', 'N/A')}",
        f"is_stably_visible: {meta.get('is_stably_visible', 'N/A')}",
        f"projected_pixel: {meta.get('projected_pixel', 'N/A')}",
        f"camera_coordinates: {meta.get('camera_coordinates', 'N/A')}",
        f"frame_index: {meta.get('frame_index', 'N/A')}",
    ]

    gen_info = traj.get("generation_info") or {}
    if isinstance(gen_info, dict):
        lines.extend([
            "",
            "=== Candidate metadata ===",
            f"visibility span: {gen_info.get('oos_span_start_sec', 'N/A')} -> {gen_info.get('oos_span_end_sec', 'N/A')} s",
            f"fixture_at_query: {gen_info.get('fixture_at_query', traj.get('fixture_at_query', 'N/A'))}",
            f"relocation_score: {gen_info.get('relocation_score', traj.get('relocation_score', 'N/A'))}",
        ])

    return "\n".join(lines)


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

    step1 = get_step(traj, 1) or get_step_by_class(traj, "oos_step1_visibility")
    step1_meta = (step1 or {}).get("answer_metadata", {}) or {}

    query_time = float(traj["query_time_sec"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        query_frame, query_idx, fps_used = read_frame_at_time(cap, query_time, fps_override)
    finally:
        cap.release()

    if query_frame is None:
        raise RuntimeError(f"Failed to read query frame at {query_time}s from {video_path}")

    qh, qw = query_frame.shape[:2]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax_frame = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    ax_frame.imshow(query_frame)
    ax_frame.set_title(
        f"Visible Step-1 query frame\n"
        f"t={query_time}s | frame={query_idx} | fps≈{fps_used:.3f} | answer={expected_visibility_label(step1)}"
    )
    draw_marker(
        ax_frame,
        step1_meta.get("projected_pixel"),
        qw,
        qh,
        "deepskyblue",
        "visible target",
        0.04,
    )
    ax_frame.axis("off")

    ax_frame.text(
        0.02,
        0.98,
        f"Object: {traj.get('object_a_name', 'N/A')}\n"
        f"Visibility answer: {expected_visibility_label(step1)}\n"
        f"Clip: {traj.get('clip_start_time_sec', 'N/A')} → {traj.get('clip_end_time_sec', 'N/A')} s",
        transform=ax_frame.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
    )

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

    fig.suptitle(f"Visible question check: {trajectory_id}", fontsize=14)
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
    only_yes: bool,
) -> List[Tuple[str, Dict[str, Any]]]:
    items = list(trajectories.items())

    if only_yes:
        filtered = []
        for tid, traj in items:
            step1 = get_step(traj, 1) or get_step_by_class(traj, "oos_step1_visibility")
            if expected_visibility_label(step1).lower() == "yes":
                filtered.append((tid, traj))
        items = filtered

    if trajectory_ids:
        lookup = dict(items)
        selected = []
        for tid in trajectory_ids:
            if tid not in lookup:
                raise KeyError(f"Trajectory id not found or filtered out: {tid}")
            selected.append((tid, lookup[tid]))
        return selected

    rng = random.Random(seed)
    if num_samples >= len(items):
        return items
    return rng.sample(items, num_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize generated visible step-1 VQA questions."
    )
    parser.add_argument("--questions", type=Path, required=True, help="Path to generated visible question JSON")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to a single video file, or a directory containing files named like <video_id>.mp4",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("visible_question_visualizations"))
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
    parser.add_argument(
        "--no_only_yes",
        action="store_true",
        help="Do not filter to trajectories whose step-1 answer is Yes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_json(args.questions)
    trajectories = normalize_trajectory_container(raw)
    selected = choose_trajectories(
        trajectories=trajectories,
        num_samples=args.num_samples,
        seed=args.seed,
        trajectory_ids=args.trajectory_ids,
        only_yes=not args.no_only_yes,
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


# python scripts/staged_oos_vqa_generation/object_spatial_relation/visualize_step1_visible_question.py \
#   --questions scripts/staged_oos_vqa_generation/object_spatial_relation/outputs/generated_vqa/P01-20240202-110250_vqa_visible_30_visible_step1.json \
#   --video data/HD-EPIC/Videos/P01/P01-20240202-110250.mp4 \
#   --output_dir scripts/staged_oos_vqa_generation/object_spatial_relation/outputs/visualizations/P01/P01-20240202-110250_vqa_visible_30_visible_step1 \
#   --num_samples 30