from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from pathlib import Path
from typing import Any


# Visible "now" for step 1.
VISIBLE_STATUSES = {
    "in_view",
    "observed_visible_in_open_fixture",
    "in_motion",
}

# Stable, localized visibility for step 2 ("last visible when & where?").
STABLE_VISIBLE_STATUSES = {
    "in_view",
    "observed_visible_in_open_fixture",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collapse_spans(times: list[float], flags: list[bool]) -> list[dict[str, Any]]:
    if not times:
        return []

    spans: list[dict[str, Any]] = []
    run_start = 0
    run_value = flags[0]

    for i in range(1, len(times)):
        if flags[i] != run_value:
            spans.append(
                {
                    "start_sec": float(times[run_start]),
                    "end_sec": float(times[i - 1]),
                    "in_view": bool(run_value),
                }
            )
            run_start = i
            run_value = flags[i]

    spans.append(
        {
            "start_sec": float(times[run_start]),
            "end_sec": float(times[-1]),
            "in_view": bool(run_value),
        }
    )
    return spans


def compute_last_true_index(flags: list[bool]) -> list[int | None]:
    out: list[int | None] = []
    last_idx: int | None = None
    for i, flag in enumerate(flags):
        if flag:
            last_idx = i
        out.append(last_idx)
    return out


def build_span_index(spans: list[dict[str, Any]]) -> tuple[list[float], list[dict[str, Any]]]:
    spans = sorted(
        spans,
        key=lambda x: (float(x["start_sec"]), float(x["end_sec"]))
    )
    starts = [float(s["start_sec"]) for s in spans]
    return starts, spans


def find_span_status(time_sec: float, starts: list[float], spans: list[dict[str, Any]]) -> str | None:
    if not spans:
        return None

    idx = bisect_right(starts, time_sec) - 1
    if idx < 0:
        return None

    sp = spans[idx]
    start_sec = float(sp["start_sec"])
    end_sec = float(sp["end_sec"])
    if start_sec <= time_sec <= end_sec:
        status = sp.get("status")
        return str(status) if status is not None else None
    return None


def normalize_in_view_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Expected Stage-2 row shape:
      {
        "assoc_id": ...,
        "name": ...,
        "sampled_times_sec": [...],
        "samples": [
          {
            "time_sec": ...,
            "status": ...,
            "in_view": ...,
            "projected_uv": ...,
            "fixture": ...,
            "frame_index": ...,
            "world_coordinates": ...,
            "camera_coordinates": ...
          },
          ...
        ]
      }
    """
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        assoc_id = str(row["assoc_id"])
        out[assoc_id] = row
    return out


def normalize_visibility_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Expected occlusion/visibility rows:
      {
        "assoc_id": ...,
        "start_sec": ...,
        "end_sec": ...,
        "status": ...
      }

    Extra keys are ignored.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        assoc_id = str(row["assoc_id"])
        out.setdefault(assoc_id, []).append(row)
    return out


def infer_status_from_in_view_sample(sample: dict[str, Any]) -> str:
    """
    Fallback only when no visibility span covers a sampled time.
    This preserves the geometric in-view signal from stage 2.
    """
    status = sample.get("status")
    if status is not None:
        return str(status)

    in_view = sample.get("in_view")
    if in_view is True:
        return "in_view"
    if in_view is False:
        return "out_of_view"
    return "unknown"


def merge_tracks(
    video_id: str,
    in_view_rows: list[dict[str, Any]],
    visibility_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    in_view_by_assoc = normalize_in_view_rows(in_view_rows)
    visibility_by_assoc = normalize_visibility_rows(visibility_rows)

    object_tracks: dict[str, dict[str, Any]] = {}

    for assoc_id, inv in in_view_by_assoc.items():
        name = str(inv.get("name", assoc_id))
        times = [float(t) for t in inv["sampled_times_sec"]]
        samples = inv["samples"]

        if len(times) != len(samples):
            raise ValueError(
                f"{assoc_id}: sampled_times_sec has length {len(times)} "
                f"but samples has length {len(samples)}"
            )

        vis_spans_raw = visibility_by_assoc.get(assoc_id, [])
        starts, spans = build_span_index(vis_spans_raw)

        status_samples: list[str] = []
        visibility_samples: list[bool] = []
        stable_visibility_samples: list[bool] = []
        projected_pixel_samples: list[list[float] | None] = []
        camera_coordinate_samples: list[list[float] | None] = []
        frame_index_samples: list[int | None] = []
        fixture_samples: list[str | None] = []
        world_coordinate_samples: list[list[float] | None] = []

        for t, s in zip(times, samples):
            span_status = find_span_status(t, starts, spans)
            merged_status = span_status if span_status is not None else infer_status_from_in_view_sample(s)

            visible_now = merged_status in VISIBLE_STATUSES
            stable_visible = merged_status in STABLE_VISIBLE_STATUSES

            projected_uv = s.get("projected_uv")
            camera_coordinates = s.get("camera_coordinates")
            frame_index = s.get("frame_index")
            fixture = s.get("fixture")
            world_coordinates = s.get("world_coordinates")

            status_samples.append(merged_status)
            visibility_samples.append(visible_now)
            stable_visibility_samples.append(stable_visible)
            projected_pixel_samples.append(projected_uv)
            camera_coordinate_samples.append(camera_coordinates)
            frame_index_samples.append(int(frame_index) if frame_index is not None else None)
            fixture_samples.append(str(fixture) if fixture is not None else None)
            world_coordinate_samples.append(world_coordinates)

        last_visible_index_before_each_sample = compute_last_true_index(stable_visibility_samples)
        merged_spans = collapse_spans(times, visibility_samples)

        object_tracks[assoc_id] = {
            "assoc_id": assoc_id,
            "name": name,
            "sampled_times_sec": times,
            "visibility_samples": visibility_samples,
            # kept for clarity/debugging; current question generator can ignore it
            "stable_visibility_samples": stable_visibility_samples,
            "status_samples": status_samples,
            "projected_pixel_samples": projected_pixel_samples,
            "camera_coordinate_samples": camera_coordinate_samples,
            "frame_index_samples": frame_index_samples,
            "fixture_samples": fixture_samples,
            "world_coordinate_samples": world_coordinate_samples,
            "last_visible_index_before_each_sample": last_visible_index_before_each_sample,
            "spans": merged_spans,
        }

    return {
        "video_id": video_id,
        "object_tracks": object_tracks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge stage-2 in-view tracks with a visibility/occlusion track into the "
            "JSON format expected by staged_oos_question_generator.py"
        )
    )
    parser.add_argument("--video_id", required=True, help="Video id for the output payload")
    parser.add_argument("--in_view_jsonl", type=Path, required=True, help="Path to stage-2 in_view_tracks.jsonl")
    parser.add_argument(
        "--visibility_jsonl",
        type=Path,
        required=True,
        help="Path to visibility/occlusion track JSONL with assoc_id/start_sec/end_sec/status rows",
    )
    parser.add_argument("--output_json", type=Path, required=True, help="Output merged JSON path")
    args = parser.parse_args()

    in_view_rows = load_jsonl(args.in_view_jsonl)
    visibility_rows = load_jsonl(args.visibility_jsonl)

    merged = merge_tracks(
        video_id=args.video_id,
        in_view_rows=in_view_rows,
        visibility_rows=visibility_rows,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Saved merged track file to {args.output_json}")


if __name__ == "__main__":
    main()

# python scripts/staged_oos_vqa_generation/object_camera_relation/prepare_tracks_for_generation_via_merging.py \
#   --video_id P04-20240413-142619 \
#   --in_view_jsonl outputs/visibility_track/P04-20240413-142619/in_view_tracks.jsonl \
#   --visibility_jsonl outputs/visibility_track/P04-20240413-142619/visibility_track.jsonl \
#   --output_json scripts/staged_oos_vqa_generation/object_camera_relation/outputs/merged_tracks/P04/P04-20240413-142619_merged_visibility_track.json