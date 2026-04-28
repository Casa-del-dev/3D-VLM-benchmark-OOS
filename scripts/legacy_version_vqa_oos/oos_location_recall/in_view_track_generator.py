from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
	visibility_samples: list[bool]
	projected_pixel_samples: list[list[float] | None]
	camera_coordinate_samples: list[list[float] | None]
	spans: list[VisibilitySpan]


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


def generate_in_view_tracks(
	video_id: str,
	annotations_root: str | Path,
	sampling_fps: float = 2.0,
	fps_for_frame_lookup: float = 30.0,
	intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT,
	start_time_sec: float | None = None,
	end_time_sec: float | None = None,
	per_object_pixels: dict[str, list[list[float] | None]] = {},
	per_object_camera_coords: dict[str, list[list[float] | None]] = {}
) -> dict[str, ObjectVisibilityTrack]:
	"""Generate per-object in-view/out-of-view tracks over uniformly sampled times.

	Inputs: video id, annotations root, sampling fps, frame-lookup fps, optional time window.
	Computes: visibility at each sampled timestamp using determine_in_view_objects.
	Outputs: one ObjectVisibilityTrack per assoc_id with samples and collapsed spans.
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
	per_object_samples: dict[str, list[bool]] = {}

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
				per_object_samples[state.assoc_id] = []
				per_object_pixels[state.assoc_id] = []
				per_object_camera_coords[state.assoc_id] = []   # ← NEW		

			# Track generation only needs binary visibility; non-ok states are treated as out-of-view.
			is_visible = bool(state.status == "ok" and state.in_view)
			per_object_samples[state.assoc_id].append(is_visible)

			per_object_pixels[state.assoc_id].append(
				state.projected_pixel if state.status == "ok" else None
			)
			per_object_camera_coords[state.assoc_id].append(
				state.camera_coordinates if state.status == "ok" else None
			)

	out: dict[str, ObjectVisibilityTrack] = {}
	for assoc_id, samples in per_object_samples.items():
		spans = _collapse_visibility(sampled_times_sec, samples)
		out[assoc_id] = ObjectVisibilityTrack(
			assoc_id=assoc_id,
			name=per_object_name[assoc_id],
			sampled_times_sec=sampled_times_sec,
			visibility_samples=samples,
			projected_pixel_samples=per_object_pixels[assoc_id],
			camera_coordinate_samples=per_object_camera_coords[assoc_id],
			spans=spans,
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
			"projected_pixel_samples": tr.projected_pixel_samples,
			"camera_coordinate_samples": tr.camera_coordinate_samples,
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
