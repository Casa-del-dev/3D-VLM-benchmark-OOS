"""Debug overlay: burn visibility status onto the source video.

Reads `visibility_track.jsonl` (stage 4) and `in_view_tracks.jsonl` (stage 2)
for one video, then writes an MP4 where each sampled frame shows:

    * a coloured dot + label on every drawable object state
        (green=in_view/observed_visible_in_open_fixture,
        red=geometrically occluded / closed-fixture-occluded / observed-not-visible)
    * a right-side sidebar listing names and statuses of currently in-sight and
        out-of-sight objects

Output lives next to the other stage outputs as
`visibility_track_overlay.mp4`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.visibility_track.common import PipelineConfig, load_config, read_jsonl  # noqa: E402
else:
    from .common import PipelineConfig, load_config, read_jsonl


DEFAULT_CONFIG = Path(__file__).resolve().parent / "visibility_track_config.yaml"

GREEN = (0, 200, 0)
RED = (0, 0, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (140, 140, 140)
DARK_GRAY = (40, 40, 40)
SIDEBAR_WIDTH = 420

IN_SIGHT_STATUSES = {
    "in_view",
    "in_motion",
    "observed_visible_in_open_fixture",
}

OUT_OF_SIGHT_STATUSES = {
    "out_of_view",
    "unobservable_no_data",
    "occluded_inside_closed_fixture",
    "observed_not_visible_in_open_fixture",
    "assumed_not_visible_in_open_fixture",
    "geometrically_occluded",
}

GREEN_POINT_STATUSES = {
    "in_view",
    "observed_visible_in_open_fixture",
}

RED_POINT_STATUSES = {
    "occluded_inside_closed_fixture",
    "geometrically_occluded",
    "observed_not_visible_in_open_fixture",
    "assumed_not_visible_in_open_fixture",
}


def _status_at(intervals: List[dict], time_sec: float) -> str | None:
    for interval in intervals:
        if float(interval["start_sec"]) <= time_sec <= float(interval["end_sec"]):
            return interval["status"]
    return None


def _color_for_status(status: str | None) -> Tuple[int, int, int] | None:
    if status in GREEN_POINT_STATUSES:
        return GREEN
    if status in RED_POINT_STATUSES:
        return RED
    return None


def _build_interval_index(rows: List[dict]) -> Dict[str, List[dict]]:
    intervals_by_object: Dict[str, List[dict]] = {}
    for row in rows:
        intervals_by_object.setdefault(row["assoc_id"], []).append(row)
    for intervals in intervals_by_object.values():
        intervals.sort(key=lambda item: float(item["start_sec"]))
    return intervals_by_object


def _build_name_index(in_view_rows: List[dict], visibility_rows: List[dict]) -> Dict[str, str]:
    names_by_object: Dict[str, str] = {}
    for row in in_view_rows:
        assoc_id = row.get("assoc_id")
        name = row.get("name")
        if assoc_id and name:
            names_by_object[assoc_id] = name
    for row in visibility_rows:
        assoc_id = row.get("assoc_id")
        # visibility_track.jsonl stores the display name under `object_name`.
        name = row.get("object_name") or row.get("name")
        if assoc_id and name and assoc_id not in names_by_object:
            names_by_object[assoc_id] = name
    return names_by_object


def _draw_label(frame, text: str, x: int, y: int, color: Tuple[int, int, int], max_x: int) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    text_x = max(2, min(max_x - text_w - 4, x))
    text_y = max(text_h + 4, y)
    cv2.rectangle(
        frame,
        (text_x - 3, text_y - text_h - 3),
        (text_x + text_w + 3, text_y + 4),
        BLACK,
        -1,
    )
    cv2.putText(frame, text, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_sidebar_section(
    frame,
    *,
    title: str,
    entries: List[Tuple[str, str]],
    x0: int,
    x1: int,
    y_top: int,
    y_bottom: int,
) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_y = y_top + 20
    line_y = title_y + 10

    cv2.putText(frame, title, (x0, title_y), font, 0.7, WHITE, 2, cv2.LINE_AA)
    cv2.line(frame, (x0, line_y), (x1, line_y), GRAY, 1)

    if not entries:
        cv2.putText(frame, "(none)", (x0, line_y + 24), font, 0.6, GRAY, 1, cv2.LINE_AA)
        return

    line_height = 20
    y = line_y + 22
    max_lines = max(1, (y_bottom - y - 4) // line_height)
    shown_entries = entries[:max_lines]
    hidden_count = len(entries) - len(shown_entries)

    for name, status in shown_entries:
        full_text = f"- {name}: {status}"
        text = full_text
        (text_w, _), _ = cv2.getTextSize(text, font, 0.55, 1)
        if x0 + text_w > x1:
            text = "..."
            for cut in range(len(full_text), 0, -1):
                candidate = full_text[:cut].rstrip() + "..."
                (candidate_w, _), _ = cv2.getTextSize(candidate, font, 0.55, 1)
                if x0 + candidate_w <= x1:
                    text = candidate
                    break
        cv2.putText(frame, text, (x0, y), font, 0.55, WHITE, 1, cv2.LINE_AA)
        y += line_height

    if hidden_count > 0 and y <= y_bottom:
        cv2.putText(frame, f"... and {hidden_count} more", (x0, y), font, 0.55, GRAY, 1, cv2.LINE_AA)


def _draw_sidebar(
    frame,
    video_width: int,
    in_sight_entries: List[Tuple[str, str]],
    out_of_sight_entries: List[Tuple[str, str]],
) -> None:
    import cv2

    height, total_width = frame.shape[:2]
    x0 = video_width + 12
    x1 = total_width - 12
    split_y = height // 2

    cv2.rectangle(frame, (video_width, 0), (total_width - 1, height - 1), DARK_GRAY, -1)
    cv2.line(frame, (video_width, 0), (video_width, height - 1), GRAY, 2)
    cv2.line(frame, (x0, split_y), (x1, split_y), GRAY, 1)

    _draw_sidebar_section(
        frame,
        title="In sight",
        entries=in_sight_entries,
        x0=x0,
        x1=x1,
        y_top=10,
        y_bottom=split_y - 10,
    )
    _draw_sidebar_section(
        frame,
        title="Out of sight",
        entries=out_of_sight_entries,
        x0=x0,
        x1=x1,
        y_top=split_y + 10,
        y_bottom=height - 10,
    )


def render_overlay(
    cfg: PipelineConfig,
    video_id: str,
    output_path: Optional[Path] = None,
    render_fps: float = 1.0,
) -> Path:
    import cv2

    out_dir = cfg.video_output_dir(video_id)
    vis_path = out_dir / "visibility_track.jsonl"
    in_view_path = out_dir / "in_view_tracks.jsonl"
    if not vis_path.exists():
        raise FileNotFoundError(f"Missing {vis_path}. Run stage 4 first.")
    if not in_view_path.exists():
        raise FileNotFoundError(f"Missing {in_view_path}. Run stage 2 first.")

    visibility_rows = read_jsonl(vis_path)
    in_view_rows = read_jsonl(in_view_path)
    intervals_by_object = _build_interval_index(visibility_rows)
    names_by_object = _build_name_index(in_view_rows, visibility_rows)

    video_path = cfg.video_file(video_id)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        output_path = out_dir / "visibility_track_overlay.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.video_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (frame_count / src_fps) if src_fps > 0 else 0.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        render_fps,
        (width + SIDEBAR_WIDTH, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer: {output_path}")

    step = 1.0 / render_fps
    times: list[float] = []
    t = 0.0
    while t <= duration + 1e-9:
        times.append(round(t, 6))
        t += step

    try:
        for time_sec in times:
            frame_idx = int(round(time_sec * src_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            canvas = cv2.copyMakeBorder(
                frame,
                top=0,
                bottom=0,
                left=0,
                right=SIDEBAR_WIDTH,
                borderType=cv2.BORDER_CONSTANT,
                value=BLACK,
            )

            in_sight_entries: list[Tuple[str, str]] = []
            out_of_sight_entries: list[Tuple[str, str]] = []

            for track in in_view_rows:
                assoc_id = track["assoc_id"]
                intervals = intervals_by_object.get(assoc_id, [])
                status = _status_at(intervals, time_sec)
                display_name = names_by_object.get(assoc_id, track.get("name", assoc_id))

                if status in IN_SIGHT_STATUSES:
                    in_sight_entries.append((display_name, status))
                elif status in OUT_OF_SIGHT_STATUSES:
                    out_of_sight_entries.append((display_name, status))

                if status in {"out_of_view", "unobservable_no_data", "in_motion"}:
                    continue

                sample = next(
                    (
                        row
                        for row in track.get("samples", [])
                        if abs(float(row["time_sec"]) - time_sec) < 1e-6 and row.get("projected_uv") is not None
                    ),
                    None,
                )
                if sample is None:
                    continue

                color = _color_for_status(status)
                if color is None:
                    continue

                u, v = sample["projected_uv"]
                u_i = max(0, min(width - 1, int(round(float(u)))))
                v_i = max(0, min(height - 1, int(round(float(v)))))
                cv2.circle(canvas, (u_i, v_i), 9, BLACK, -1)
                cv2.circle(canvas, (u_i, v_i), 7, color, -1)
                _draw_label(canvas, track.get("name", assoc_id), u_i + 12, v_i - 10, color, width)

            in_sight_entries.sort(key=lambda item: (item[0].lower(), item[1]))
            out_of_sight_entries.sort(key=lambda item: (item[0].lower(), item[1]))
            _draw_sidebar(canvas, width, in_sight_entries, out_of_sight_entries)

            cv2.putText(
                canvas,
                f"t={time_sec:.1f}s",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                WHITE,
                2,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        cap.release()
        writer.release()

    return output_path


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", action="append", default=None,
                        help="Video ID to overlay. May be repeated.")
    parser.add_argument("--participant", action="append", default=None,
                        help="Process all videos for a participant. May be repeated.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path override (only used when processing a single video).")
    parser.add_argument("--fps", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config)

    video_ids: List[str] = list(args.video or [])

    if args.participant:
        for participant in args.participant:
            for v in cfg.videos_for_participant(participant):
                if v not in video_ids:
                    video_ids.append(v)

    if not video_ids:
        # No explicit flags — fall back to config
        for participant in cfg.participants:
            for v in cfg.videos_for_participant(participant):
                if v not in video_ids:
                    video_ids.append(v)

    if not video_ids:
        video_ids = cfg.videos

    if not video_ids:
        raise ValueError(
            "No videos to process. Pass --video, --participant, or populate inputs in the config."
        )

    if args.output is not None and len(video_ids) > 1:
        raise ValueError("--output can only be used when processing a single video.")

    for video_id in video_ids:
        out_path = render_overlay(
            cfg=cfg,
            video_id=video_id,
            output_path=args.output if len(video_ids) == 1 else None,
            render_fps=args.fps,
        )
        print(f"[overlay] wrote {out_path}")


if __name__ == "__main__":
    main()