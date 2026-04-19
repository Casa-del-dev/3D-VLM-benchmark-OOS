from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Any


DEFAULT_INTERMEDIATE_ROOT = "Intermediate_data"


@dataclass(frozen=True)
class FrameContext:
	"""Per-query frame context used for visibility projection.

	Inputs (via load_frame_context): video_id, query time, annotations root, fps.
	Computes: nearest frame pose, camera projection settings, per-video object/mask maps.
	Outputs: immutable bundle consumed by determine_in_view_objects.
	"""
	video_id: str
	frame_index: int
	image_width: int
	image_height: int
	model_name: str
	projection_params: list[float]
	T_camera_world: list[list[float]]
	assoc_objects: dict[str, dict[str, Any]]
	mask_info_video: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ObjectState:
	"""Visibility/state summary for one object at a queried time.

	Inputs: assoc object entry + selected track/mask + frame/camera context.
	Computes: stable 3D point selection, camera coordinates, pixel projection, in-view flag.
	Outputs: immutable per-object result used by downstream OOS modules.
	"""
	assoc_id: str
	name: str
	status: str
	selection_mode: str | None
	track_id: str | None
	time_segment: list[float] | None
	mask_id: str | None
	frame_number: int | None
	fixture: str | None
	world_coordinates: list[float] | None
	camera_coordinates: list[float] | None
	projected_pixel: list[float] | None
	depth_in_camera: float | None
	in_view: bool | None
	next_closest_possible_time: float | None
	comment: str
	mask_bbox: list[float] | None = None  # xyxy from the chosen mask frame


def load_json(path: Path) -> Any:
	"""Load and return a JSON object from disk."""
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
	"""Load a JSONL file into a list of parsed dictionaries."""
	rows: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				rows.append(json.loads(line))
	return rows


def to_homogeneous_4x4(T_3x4: list[list[float]]) -> list[list[float]]:
	"""Convert a rigid 3x4 transform to homogeneous 4x4 form."""
	return [
		[T_3x4[0][0], T_3x4[0][1], T_3x4[0][2], T_3x4[0][3]],
		[T_3x4[1][0], T_3x4[1][1], T_3x4[1][2], T_3x4[1][3]],
		[T_3x4[2][0], T_3x4[2][1], T_3x4[2][2], T_3x4[2][3]],
		[0.0, 0.0, 0.0, 1.0],
	]


def mat4_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
	"""Multiply two 4x4 matrices."""
	out = [[0.0] * 4 for _ in range(4)]
	for i in range(4):
		for j in range(4):
			out[i][j] = sum(A[i][k] * B[k][j] for k in range(4))
	return out


def invert_rigid_4x4(T: list[list[float]]) -> list[list[float]]:
	"""Invert a rigid 4x4 transform using R^T and translated origin."""
	R = [[T[i][j] for j in range(3)] for i in range(3)]
	t = [T[0][3], T[1][3], T[2][3]]
	Rt = [[R[j][i] for j in range(3)] for i in range(3)]
	t_inv = [-(Rt[i][0] * t[0] + Rt[i][1] * t[1] + Rt[i][2] * t[2]) for i in range(3)]
	return [
		[Rt[0][0], Rt[0][1], Rt[0][2], t_inv[0]],
		[Rt[1][0], Rt[1][1], Rt[1][2], t_inv[1]],
		[Rt[2][0], Rt[2][1], Rt[2][2], t_inv[2]],
		[0.0, 0.0, 0.0, 1.0],
	]


def transform_point(T: list[list[float]], p: list[float]) -> list[float]:
	"""Transform a 3D point by a 4x4 pose matrix."""
	x, y, z = p
	return [
		T[0][0] * x + T[0][1] * y + T[0][2] * z + T[0][3],
		T[1][0] * x + T[1][1] * y + T[1][2] * z + T[1][3],
		T[2][0] * x + T[2][1] * y + T[2][2] * z + T[2][3],
	]


def project_fisheye624(point_cam: list[float], params: list[float]) -> tuple[list[float] | None, float, bool]:
	"""Project a camera-frame 3D point to pixels with the FISHEYE624 model."""
	x, y, z = point_cam
	if z <= 1e-9:
		return None, z, False

	f, cu, cv = [float(v) for v in params[0:3]]
	k0, k1, k2, k3, k4, k5 = [float(v) for v in params[3:9]]
	p0, p1 = [float(v) for v in params[9:11]]
	s0, s1, s2, s3 = [float(v) for v in params[11:15]]

	a = x / z
	b = y / z
	r = math.sqrt(a * a + b * b)
	if r < 1e-12:
		return [cu, cv], z, True

	theta = math.atan(r)
	theta2 = theta * theta
	theta4 = theta2 * theta2
	theta6 = theta4 * theta2
	theta8 = theta4 * theta4
	theta10 = theta8 * theta2
	theta12 = theta6 * theta6
	theta_d = theta * (1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8 + k4 * theta10 + k5 * theta12)

	scale = theta_d / r
	xr = a * scale
	yr = b * scale
	rr = xr * xr + yr * yr
	rr2 = rr * rr
	x_tan = (2.0 * xr * xr + rr) * p0 + 2.0 * xr * yr * p1
	y_tan = (2.0 * yr * yr + rr) * p1 + 2.0 * xr * yr * p0
	x_prism = s0 * rr + s1 * rr2
	y_prism = s2 * rr + s3 * rr2
	return [f * (xr + x_tan + x_prism) + cu, f * (yr + y_tan + y_prism) + cv], z, True


def choose_track_for_time(tracks: list[dict[str, Any]], time_sec: float) -> tuple[dict[str, Any] | None, str | None, float | None]:
	"""Choose track around query time and return selection mode metadata."""
	past_tracks: list[dict[str, Any]] = []
	future_tracks: list[dict[str, Any]] = []
	for tr in tracks:
		start_t, end_t = tr["time_segment"]
		if start_t <= time_sec <= end_t:
			return tr, "in_motion", None
		if end_t < time_sec:
			past_tracks.append(tr)
		elif start_t > time_sec:
			future_tracks.append(tr)
	if past_tracks:
		chosen = max(past_tracks, key=lambda tr: tr["time_segment"][1])
		return chosen, "past", None
	if future_tracks:
		chosen = min(future_tracks, key=lambda tr: tr["time_segment"][0])
		return chosen, "future", chosen["time_segment"][0]
	return None, None, None


def get_mask_from_track(mask_info_video: dict[str, Any], track: dict[str, Any], pick: str = "latest") -> dict[str, Any] | None:
	"""Fetch first/latest valid mask observation associated with a track."""
	masks: list[dict[str, Any]] = []
	for mask_id in track.get("masks", []):
		if mask_id not in mask_info_video:
			continue
		entry = mask_info_video[mask_id]
		masks.append(
			{
				"mask_id": mask_id,
				"frame_number": entry["frame_number"],
				"3d_location": entry["3d_location"],
				"bbox": entry.get("bbox"),
				"fixture": entry.get("fixture"),
			}
		)
	if not masks:
		return None
	if pick == "latest":
		return max(masks, key=lambda m: m["frame_number"])
	if pick == "first":
		return min(masks, key=lambda m: m["frame_number"])
	raise ValueError(f"Unknown pick mode: {pick}")


def point_in_image(pixel_xy: list[float] | None, width: int, height: int) -> bool:
	"""Return True when pixel coordinates lie within image bounds."""
	if pixel_xy is None:
		return False
	u, v = pixel_xy
	return 0.0 <= u < width and 0.0 <= v < height


def find_closest_frame_entry(framewise_rows: list[dict[str, Any]], time_sec: float, fps: float) -> dict[str, Any]:
	"""Find frame metadata closest to query time using frame index distance."""
	target_frame = int(round(time_sec * fps))
	candidates = [r for r in framewise_rows if r.get("frame_index") is not None and r.get("T_world_device") is not None]
	if not candidates:
		raise ValueError("No valid frame entries with T_world_device.")
	return min(candidates, key=lambda r: abs(int(r["frame_index"]) - target_frame))


@dataclass
class VideoCache:
	"""Per-video data loaded once and reused across all time steps.

	Holds the large shared JSON blobs (assoc_info, mask_info) and the
	per-video calibration + framewise rows so they are not re-read from
	disk on every call to load_frame_context.
	"""
	video_id: str
	assoc_objects: dict[str, dict[str, Any]]
	mask_info_video: dict[str, dict[str, Any]]
	calibration_rgb: dict[str, Any]
	framewise_rows: list[dict[str, Any]]
	# Pre-built index for O(log N) frame lookup (populated by build())
	framewise_sorted_indices: list[int]
	framewise_index: dict[int, dict[str, Any]]

	@classmethod
	def build(
		cls,
		video_id: str,
		annotations_root: str | Path,
		intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT,
	) -> "VideoCache":
		annotations_root = Path(annotations_root)
		participant_id = video_id.split("-")[0]

		assoc_info = load_json(annotations_root / "scene-and-object-movements" / "assoc_info.json")
		mask_info = load_json(annotations_root / "scene-and-object-movements" / "mask_info.json")
		if video_id not in assoc_info:
			raise KeyError(f"Video {video_id} not found in assoc_info.json")
		if video_id not in mask_info:
			raise KeyError(f"Video {video_id} not found in mask_info.json")

		video_dir = annotations_root / intermediate_root / participant_id / video_id
		calibration = load_json(video_dir / "device_calibration.json")
		framewise_rows = load_jsonl(video_dir / "framewise_info.jsonl")

		# Build sorted index over valid rows for O(log N) closest-frame lookup.
		valid_rows = [
			r for r in framewise_rows
			if r.get("frame_index") is not None and r.get("T_world_device") is not None
		]
		framewise_sorted_indices = sorted(int(r["frame_index"]) for r in valid_rows)
		framewise_index = {int(r["frame_index"]): r for r in valid_rows}

		return cls(
			video_id=video_id,
			assoc_objects=assoc_info[video_id],
			mask_info_video=mask_info[video_id],
			calibration_rgb=calibration["cameras"]["camera-rgb"],
			framewise_rows=framewise_rows,
			framewise_sorted_indices=framewise_sorted_indices,
			framewise_index=framewise_index,
		)


def load_frame_context(
	video_id: str,
	time_sec: float,
	annotations_root: str | Path,
	fps: float,
	intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT,
	cache: "VideoCache | None" = None,
) -> FrameContext:
	"""Load and assemble all frame-level inputs needed for visibility checks.

	Inputs: video id, query time in seconds, annotation root path, fps, intermediate-data folder name.
	Computes: nearest frame entry, camera extrinsics/intrinsics, and T_camera_world transform.
	Outputs: FrameContext with calibration, frame index, and object/mask dictionaries for the video.

	Pass a pre-built VideoCache to avoid re-reading the large JSON files on
	every call (critical when processing many time steps for the same video).
	"""
	if cache is None:
		cache = VideoCache.build(video_id, annotations_root, intermediate_root)

	# O(log N) lookup using pre-built sorted index.
	target_frame = int(round(time_sec * fps))
	indices = cache.framewise_sorted_indices
	if not indices:
		raise ValueError("No valid frame entries with T_world_device.")
	pos = bisect.bisect_left(indices, target_frame)
	best = indices[min(pos, len(indices) - 1)]
	if pos > 0 and abs(indices[pos - 1] - target_frame) < abs(best - target_frame):
		best = indices[pos - 1]
	frame_entry = cache.framewise_index[best]
	rgb = cache.calibration_rgb
	T_device_camera_raw = rgb["T_device_camera"]
	T_device_camera = to_homogeneous_4x4(T_device_camera_raw) if len(T_device_camera_raw) == 3 else T_device_camera_raw
	T_world_device_raw = frame_entry["T_world_device"]
	T_world_device = to_homogeneous_4x4(T_world_device_raw) if len(T_world_device_raw) == 3 else T_world_device_raw
	T_camera_world = invert_rigid_4x4(mat4_mul(T_world_device, T_device_camera))

	return FrameContext(
		video_id=video_id,
		frame_index=int(frame_entry["frame_index"]),
		image_width=int(rgb["image_size"][0]),
		image_height=int(rgb["image_size"][1]),
		model_name=rgb["model_name"],
		projection_params=[float(v) for v in rgb["projection_params"]],
		T_camera_world=T_camera_world,
		assoc_objects=cache.assoc_objects,
		mask_info_video=cache.mask_info_video,
	)


def determine_in_view_objects(
	video_id: str,
	time_sec: float,
	annotations_root: str | Path,
	fps: float = 30.0,
	intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT,
	cache: "VideoCache | None" = None,
) -> list[ObjectState]:
	"""Estimate in-view status for all objects at a given query time.

	Inputs: video id, query time, annotation root, fps, and intermediate-data folder name.
	Computes: track selection around time t, mask sampling, camera projection, and in/out-of-view decision.
	Outputs: list of ObjectState entries (one per object), including failure/status cases.

	Pass a pre-built VideoCache to avoid re-reading large JSON files per call.
	"""
	# Reuses the same track-selection policy as preprocessing scripts.
	ctx = load_frame_context(video_id, time_sec, annotations_root, fps, intermediate_root, cache=cache)
	out: list[ObjectState] = []

	for assoc_id, obj in ctx.assoc_objects.items():
		track, mode, next_time = choose_track_for_time(obj["tracks"], time_sec)
		if track is None:
			out.append(
				ObjectState(
					assoc_id=assoc_id,
					name=obj["name"],
					status="no_track_available",
					selection_mode=None,
					track_id=None,
					time_segment=None,
					mask_id=None,
					frame_number=None,
					fixture=None,
					world_coordinates=None,
					camera_coordinates=None,
					projected_pixel=None,
					depth_in_camera=None,
					in_view=None,
					next_closest_possible_time=next_time,
					comment="No track around queried time.",
				)
			)
			continue
		if mode == "in_motion":
			out.append(
				ObjectState(
					assoc_id=assoc_id,
					name=obj["name"],
					status="in_motion",
					selection_mode=mode,
					track_id=track["track_id"],
					time_segment=track["time_segment"],
					mask_id=None,
					frame_number=None,
					fixture=None,
					world_coordinates=None,
					camera_coordinates=None,
					projected_pixel=None,
					depth_in_camera=None,
					in_view=None,
					next_closest_possible_time=next_time,
					comment="Inside movement segment; no stable location used.",
				)
			)
			continue

		pick = "latest" if mode == "past" else "first"
		mask = get_mask_from_track(ctx.mask_info_video, track, pick=pick)
		if mask is None:
			out.append(
				ObjectState(
					assoc_id=assoc_id,
					name=obj["name"],
					status="no_valid_mask",
					selection_mode=mode,
					track_id=track["track_id"],
					time_segment=track["time_segment"],
					mask_id=None,
					frame_number=None,
					fixture=None,
					world_coordinates=None,
					camera_coordinates=None,
					projected_pixel=None,
					depth_in_camera=None,
					in_view=None,
					next_closest_possible_time=next_time,
					comment="Track found but no valid mask found in mask_info.",
				)
			)
			continue

		world_xyz = mask["3d_location"]
		if world_xyz is None:
			out.append(
				ObjectState(
					assoc_id=assoc_id,
					name=obj["name"],
					status="no_valid_mask",
					selection_mode=mode,
					track_id=track["track_id"],
					time_segment=track["time_segment"],
					mask_id=mask["mask_id"],
					frame_number=int(mask["frame_number"]),
					fixture=mask["fixture"],
					world_coordinates=None,
					camera_coordinates=None,
					projected_pixel=None,
					depth_in_camera=None,
					in_view=None,
					next_closest_possible_time=next_time,
					comment="Mask found but 3d_location is None.",
				)
			)
			continue
		cam_xyz = transform_point(ctx.T_camera_world, world_xyz)
		pixel_xy, depth, valid = project_fisheye624(cam_xyz, ctx.projection_params) if ctx.model_name == "CameraModelType.FISHEYE624" else (None, cam_xyz[2], False)
		in_view = valid and point_in_image(pixel_xy, ctx.image_width, ctx.image_height)
		comment = "Projected inside RGB image bounds." if in_view else "Projected outside RGB image bounds or behind camera."

		out.append(
			ObjectState(
				assoc_id=assoc_id,
				name=obj["name"],
				status="ok",
				selection_mode=mode,
				track_id=track["track_id"],
				time_segment=track["time_segment"],
				mask_id=mask["mask_id"],
				frame_number=int(mask["frame_number"]),
				fixture=mask["fixture"],
				world_coordinates=world_xyz,
				camera_coordinates=cam_xyz,
				projected_pixel=pixel_xy,
				depth_in_camera=depth,
				in_view=in_view,
				next_closest_possible_time=next_time,
				comment=comment,
				mask_bbox=list(mask["bbox"]) if mask.get("bbox") is not None else None,
			)
		)

	return out
