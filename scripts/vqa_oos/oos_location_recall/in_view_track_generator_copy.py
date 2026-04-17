from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import json

from in_view_determination import DEFAULT_INTERMEDIATE_ROOT, determine_in_view_objects, load_jsonl


@dataclass(frozen=True)
class VisibilitySpan:
    start_sec: float
    end_sec: float
    in_view: bool


@dataclass(frozen=True)
class ObjectVisibilityTrack:
    assoc_id: str
    name: str
    sampled_times_sec: list[float]

    # Visible in the scene.
    visibility_samples: list[bool]

    # Safe for location-based querying.
    queryable_samples: list[bool]

    # Per-sample raw status from determine_in_view_objects or an external visibility file.
    status_samples: list[str]

    # Optional per-sample geometry metadata.
    projected_pixel_samples: list[list[float] | None]
    camera_coordinate_samples: list[list[float] | None]
    frame_index_samples: list[int | None]

    # Fast lookup helpers for downstream question generation.
    last_visible_index_before_each_sample: list[int | None]
    last_queryable_index_before_each_sample: list[int | None]

    spans: list[VisibilitySpan]


# Only these count as visible. EVERYTHING ELSE is treated as out-of-sight / not visible.
VISIBLE_STATUSES = {"in_view", "in_motion", "observed_visible_in_open_fixture"}

# Only stable visible states are queryable.
QUERYABLE_STATUSES = {"in_view", "observed_visible_in_open_fixture"}


def _normalize_external_status(status: Any) -> str:
    status = str(status).strip().lower()
    aliases = {
        "visible": "in_view",
        "visible_in_open_fixture": "observed_visible_in_open_fixture",
        "not_visible_in_open_fixture": "observed_not_visible_in_open_fixture",
        "occluded": "geometrically_occluded",
        "inside_closed_fixture": "occluded_inside_closed_fixture",
    }
    return aliases.get(status, status)


def _status_is_visible(status: str) -> bool:
    return _normalize_external_status(status) in VISIBLE_STATUSES


def _status_is_queryable(status: str) -> bool:
    return _normalize_external_status(status) in QUERYABLE_STATUSES


def _infer_time_window_sec(
    video_id: str,
    annotations_root: str | Path,
    fps_for_frame_lookup: float,
    intermediate_root: str,
) -> tuple[float, float]:
    """Infer the video sampling window from min/max frame indices."""
    annotations_root = Path(annotations_root)
    participant_id = video_id.split("-")[0]
    framewise_path = annotations_root / intermediate_root / participant_id / video_id / "framewise_info.jsonl"
    rows = load_jsonl(framewise_path)

    frame_indices = [int(r["frame_index"]) for r in rows if r.get("frame_index") is not None]
    if not frame_indices:
        raise ValueError(f"No valid frame_index entries found for video {video_id}")

    min_frame = min(frame_indices)
    max_frame = max(frame_indices)
    return min_frame / fps_for_frame_lookup, max_frame / fps_for_frame_lookup


def _build_sample_times(start_sec: float, end_sec: float, sampling_fps: float) -> list[float]:
    """Build an inclusive, uniformly sampled timestamp list."""
    if sampling_fps <= 0:
        raise ValueError("sampling_fps must be > 0")
    if end_sec < start_sec:
        raise ValueError("end_sec must be >= start_sec")

    step = 1.0 / sampling_fps
    times: list[float] = []
    t = start_sec
    while t <= end_sec + 1e-9:
        times.append(round(t, 6))
        t += step
    return times


def _collapse_visibility(sampled_times_sec: list[float], visibility_samples: list[bool]) -> list[VisibilitySpan]:
    """Collapse per-time visibility booleans into contiguous spans."""
    if not sampled_times_sec:
        return []

    spans: list[VisibilitySpan] = []
    run_start = 0
    run_value = visibility_samples[0]

    for i in range(1, len(visibility_samples)):
        if visibility_samples[i] != run_value:
            spans.append(
                VisibilitySpan(
                    start_sec=sampled_times_sec[run_start],
                    end_sec=sampled_times_sec[i - 1],
                    in_view=run_value,
                )
            )
            run_start = i
            run_value = visibility_samples[i]

    spans.append(
        VisibilitySpan(
            start_sec=sampled_times_sec[run_start],
            end_sec=sampled_times_sec[-1],
            in_view=run_value,
        )
    )
    return spans


def _compute_last_true_index_before_each_sample(samples: list[bool]) -> list[int | None]:
    """For each sample i, store the latest earlier index j < i with samples[j] == True."""
    out: list[int | None] = []
    last_true_idx: int | None = None

    for is_true in samples:
        out.append(last_true_idx)
        if is_true:
            last_true_idx = len(out) - 1

    return out


def _normalize_geometry_status(status: str, in_view: bool | None) -> str:
    status = str(status).strip().lower()
    if status == "ok" and in_view:
        return "in_view"
    if status == "in_motion":
        return "in_motion"
    if status == "ok" and not in_view:
        return "out_of_view"
    return status


def generate_in_view_tracks(
    video_id: str,
    annotations_root: str | Path,
    sampling_fps: float = 2.0,
    fps_for_frame_lookup: float = 30.0,
    intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT,
    start_time_sec: float | None = None,
    end_time_sec: float | None = None,
) -> dict[str, ObjectVisibilityTrack]:
    """Generate per-object tracks from geometric visibility only.

    This remains useful as a geometry backbone. Higher-level scripts can then
    override status/visibility from an external visibility span file.
    """
    if start_time_sec is None or end_time_sec is None:
        inferred_start, inferred_end = _infer_time_window_sec(
            video_id=video_id,
            annotations_root=annotations_root,
            fps_for_frame_lookup=fps_for_frame_lookup,
            intermediate_root=intermediate_root,
        )
        if start_time_sec is None:
            start_time_sec = inferred_start
        if end_time_sec is None:
            end_time_sec = inferred_end

    sampled_times_sec = _build_sample_times(start_time_sec, end_time_sec, sampling_fps)
    if not sampled_times_sec:
        return {}

    per_object_name: dict[str, str] = {}
    per_object_visibility: dict[str, list[bool]] = {}
    per_object_queryable: dict[str, list[bool]] = {}
    per_object_status: dict[str, list[str]] = {}
    per_object_pixels: dict[str, list[list[float] | None]] = {}
    per_object_camera_coords: dict[str, list[list[float] | None]] = {}
    per_object_frame_indices: dict[str, list[int | None]] = {}

    for t_sec in sampled_times_sec:
        states = determine_in_view_objects(
            video_id=video_id,
            time_sec=t_sec,
            annotations_root=annotations_root,
            fps=fps_for_frame_lookup,
            intermediate_root=intermediate_root,
        )

        for state in states:
            if state.assoc_id not in per_object_name:
                per_object_name[state.assoc_id] = state.name
                per_object_visibility[state.assoc_id] = []
                per_object_queryable[state.assoc_id] = []
                per_object_status[state.assoc_id] = []
                per_object_pixels[state.assoc_id] = []
                per_object_camera_coords[state.assoc_id] = []
                per_object_frame_indices[state.assoc_id] = []

            status = _normalize_geometry_status(str(state.status), state.in_view)
            is_visible = _status_is_visible(status)
            is_queryable = _status_is_queryable(status)

            per_object_visibility[state.assoc_id].append(is_visible)
            per_object_queryable[state.assoc_id].append(is_queryable)
            per_object_status[state.assoc_id].append(status)
            per_object_pixels[state.assoc_id].append(
                [float(v) for v in state.projected_pixel] if state.projected_pixel is not None else None
            )
            per_object_camera_coords[state.assoc_id].append(
                [float(v) for v in state.camera_coordinates] if state.camera_coordinates is not None else None
            )
            per_object_frame_indices[state.assoc_id].append(
                int(state.frame_number) if state.frame_number is not None else None
            )

    out: dict[str, ObjectVisibilityTrack] = {}
    for assoc_id in per_object_name.keys():
        visibility_samples = per_object_visibility[assoc_id]
        queryable_samples = per_object_queryable[assoc_id]
        last_visible_index_before_each_sample = _compute_last_true_index_before_each_sample(visibility_samples)
        last_queryable_index_before_each_sample = _compute_last_true_index_before_each_sample(queryable_samples)
        spans = _collapse_visibility(sampled_times_sec, visibility_samples)

        out[assoc_id] = ObjectVisibilityTrack(
            assoc_id=assoc_id,
            name=per_object_name[assoc_id],
            sampled_times_sec=sampled_times_sec,
            visibility_samples=visibility_samples,
            queryable_samples=queryable_samples,
            status_samples=per_object_status[assoc_id],
            projected_pixel_samples=per_object_pixels[assoc_id],
            camera_coordinate_samples=per_object_camera_coords[assoc_id],
            frame_index_samples=per_object_frame_indices[assoc_id],
            last_visible_index_before_each_sample=last_visible_index_before_each_sample,
            last_queryable_index_before_each_sample=last_queryable_index_before_each_sample,
            spans=spans,
        )
    return out


def load_visibility_span_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def apply_external_visibility_spans(
    tracks: dict[str, ObjectVisibilityTrack],
    span_rows: list[dict[str, Any]],
    video_id: str,
) -> dict[str, ObjectVisibilityTrack]:
    """Override track visibility/status using external span labels.

    Source-of-truth rule:
    - Only in_motion, observed_visible_in_open_fixture, and in_view are visible.
    - EVERYTHING ELSE is not visible / out-of-sight.
    - If external rows exist for an object, they are authoritative.
    - If multiple external spans overlap, prefer the stronger OOS status.
    - If no external span matches a sampled time for that object, default to out_of_view.
    """

    status_priority = {
        "occluded_inside_closed_fixture": 70,
        "geometrically_occluded": 60,
        "observed_not_visible_in_open_fixture": 50,
        "out_of_view": 40,
        "in_motion": 30,
        "observed_visible_in_open_fixture": 20,
        "in_view": 10,
    }

    spans_by_assoc: dict[str, list[dict[str, Any]]] = {}
    for row in span_rows:
        if str(row.get("video_id")) != str(video_id):
            continue
        assoc_id = row.get("assoc_id")
        if assoc_id is None:
            continue
        normalized = dict(row)
        normalized["status"] = _normalize_external_status(row.get("status", "out_of_view"))
        spans_by_assoc.setdefault(str(assoc_id), []).append(normalized)

    for assoc_id in spans_by_assoc:
        spans_by_assoc[assoc_id].sort(
            key=lambda r: (
                float(r.get("start_sec", 0.0)),
                float(r.get("end_sec", r.get("start_sec", 0.0))),
            )
        )

    def choose_status(rows: list[dict[str, Any]], t: float) -> str:
        matches: list[dict[str, Any]] = []
        for row in rows:
            start = float(row.get("start_sec", 0.0))
            end = float(row.get("end_sec", start))
            if start - 1e-9 <= t <= end + 1e-9:
                matches.append(row)

        if not matches:
            return "out_of_view"

        def rank(row: dict[str, Any]) -> tuple[int, float, float]:
            status = _normalize_external_status(row.get("status", "out_of_view"))
            start = float(row.get("start_sec", 0.0))
            end = float(row.get("end_sec", start))
            span_len = end - start
            return (
                status_priority.get(status, 0),  # higher is stronger
                -span_len,                       # shorter span is more specific
                start,                           # later start is more specific
            )

        best = max(matches, key=rank)
        return _normalize_external_status(best.get("status", "out_of_view"))

    out: dict[str, ObjectVisibilityTrack] = {}
    for assoc_id, tr in tracks.items():
        rows = spans_by_assoc.get(assoc_id)
        if not rows:
            out[assoc_id] = tr
            continue

        new_status: list[str] = []
        new_visible: list[bool] = []
        new_queryable: list[bool] = []

        for t in tr.sampled_times_sec:
            status = choose_status(rows, float(t))
            new_status.append(status)
            new_visible.append(_status_is_visible(status))
            new_queryable.append(_status_is_queryable(status))

        last_visible_index_before_each_sample = _compute_last_true_index_before_each_sample(new_visible)
        last_queryable_index_before_each_sample = _compute_last_true_index_before_each_sample(new_queryable)
        new_spans = _collapse_visibility(tr.sampled_times_sec, new_visible)

        out[assoc_id] = replace(
            tr,
            visibility_samples=new_visible,
            queryable_samples=new_queryable,
            status_samples=new_status,
            last_visible_index_before_each_sample=last_visible_index_before_each_sample,
            last_queryable_index_before_each_sample=last_queryable_index_before_each_sample,
            spans=new_spans,
        )

    return out


def tracks_to_dict(tracks: dict[str, ObjectVisibilityTrack]) -> dict[str, Any]:
    """Convert generated tracks to a JSON-serializable dictionary."""
    out: dict[str, Any] = {}
    for assoc_id, tr in tracks.items():
        out[assoc_id] = {
            "name": tr.name,
            "sampled_times_sec": tr.sampled_times_sec,
            "visibility_samples": tr.visibility_samples,
            "queryable_samples": tr.queryable_samples,
            "status_samples": tr.status_samples,
            "projected_pixel_samples": tr.projected_pixel_samples,
            "camera_coordinate_samples": tr.camera_coordinate_samples,
            "frame_index_samples": tr.frame_index_samples,
            "last_visible_index_before_each_sample": tr.last_visible_index_before_each_sample,
            "last_queryable_index_before_each_sample": tr.last_queryable_index_before_each_sample,
            "spans": [
                {
                    "start_sec": sp.start_sec,
                    "end_sec": sp.end_sec,
                    "in_view": sp.in_view,
                }
                for sp in tr.spans
            ],
        }
    return out
