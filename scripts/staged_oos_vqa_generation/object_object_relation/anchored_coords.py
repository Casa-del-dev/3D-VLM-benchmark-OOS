from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from in_view_determination import determine_in_view_objects, load_frame_context


@dataclass(frozen=True)
class AnchoredRelation:
	"""Relative relation of object A to anchor object B at one query time.

	Inputs: ObjectState for A and anchor B from in_view_determination.
	Computes: A-B displacement in B-centric camera-aligned coordinates.
	Outputs: immutable relation record used for relative-question answering.
	"""
	assoc_id: str
	name: str
	status: str
	a_relative_to_b_egocentric: list[float] | None
	visible_in_anchor_frame: bool | None


def _transform_point_3x4(T: list[list[float]], p: list[float]) -> list[float]:
	"""Apply a 3x4 affine transform to a 3D point."""
	x, y, z = [float(v) for v in p]
	return [
		T[0][0] * x + T[0][1] * y + T[0][2] * z + T[0][3],
		T[1][0] * x + T[1][1] * y + T[1][2] * z + T[1][3],
		T[2][0] * x + T[2][1] * y + T[2][2] * z + T[2][3],
	]


def _world_to_anchor_local(T_camera_world: list[list[float]], xyz_b_world: list[float]) -> list[list[float]]:
	"""Build the same world->anchor-local 3x4 transform as relative_relation_with_viz.

	Axes follow the queried camera frame; origin is translated to anchor B.
	"""
	R = [
		[T_camera_world[0][0], T_camera_world[0][1], T_camera_world[0][2]],
		[T_camera_world[1][0], T_camera_world[1][1], T_camera_world[1][2]],
		[T_camera_world[2][0], T_camera_world[2][1], T_camera_world[2][2]],
	]
	bx, by, bz = [float(v) for v in xyz_b_world]
	return [
		[R[0][0], R[0][1], R[0][2], -(R[0][0] * bx + R[0][1] * by + R[0][2] * bz)],
		[R[1][0], R[1][1], R[1][2], -(R[1][0] * bx + R[1][1] * by + R[1][2] * bz)],
		[R[2][0], R[2][1], R[2][2], -(R[2][0] * bx + R[2][1] * by + R[2][2] * bz)],
	]


def pick_central_anchor(
	video_id: str,
	time_sec: float,
	annotations_root: str | Path,
	fps: float = 30.0,
) -> dict | None:
	"""Pick anchor B as the most central currently visible object.

	Inputs: video id, query time, annotation root, fps.
	Computes: per-object visibility then minimizes pixel distance to image center.
	Outputs: dict with anchor assoc_id/name/pixel, or None if no visible object exists.
	"""
	states = determine_in_view_objects(video_id=video_id, time_sec=time_sec, annotations_root=annotations_root, fps=fps)
	visible = [s for s in states if s.status == "ok" and s.in_view and s.projected_pixel is not None]
	if not visible:
		return None

	# Most-central object is defined by 2D distance to image center at query time.
	ctx = load_frame_context(video_id=video_id, time_sec=time_sec, annotations_root=annotations_root, fps=fps)
	width = float(ctx.image_width)
	height = float(ctx.image_height)
	cx, cy = width / 2.0, height / 2.0

	best = min(visible, key=lambda s: (s.projected_pixel[0] - cx) ** 2 + (s.projected_pixel[1] - cy) ** 2)
	return {"assoc_id": best.assoc_id, "name": best.name, "pixel": best.projected_pixel}


def compute_anchored_relations(
	video_id: str,
	time_sec: float,
	anchor_assoc_id: str,
	annotations_root: str | Path,
	fps: float = 30.0,
) -> list[AnchoredRelation]:
	"""Compute A-relative-to-B vectors for all objects in a video at time t.

	Inputs: video id, query time, chosen anchor assoc id, annotation root, fps.
	Computes: camera-coordinate displacement (A - B) for each non-anchor object.
	Outputs: list of AnchoredRelation entries, preserving invalid-status cases where needed.
	"""
	states = determine_in_view_objects(video_id=video_id, time_sec=time_sec, annotations_root=annotations_root, fps=fps)
	by_id = {s.assoc_id: s for s in states}
	if anchor_assoc_id not in by_id:
		raise KeyError(f"Anchor assoc_id {anchor_assoc_id} not found in video {video_id}")

	anchor = by_id[anchor_assoc_id]
	if anchor.status != "ok" or anchor.world_coordinates is None:
		raise ValueError("Anchor object is not in a valid stable state at queried time")
	if not bool(anchor.in_view):
		raise ValueError("Anchor object is not visible in the queried camera frame")

	ctx = load_frame_context(video_id=video_id, time_sec=time_sec, annotations_root=annotations_root, fps=fps)
	T_local_world = _world_to_anchor_local(ctx.T_camera_world, anchor.world_coordinates)

	out: list[AnchoredRelation] = []
	for state in states:
		if state.assoc_id == anchor_assoc_id:
			continue
		if state.status != "ok" or state.world_coordinates is None:
			out.append(AnchoredRelation(state.assoc_id, state.name, state.status, None, state.in_view))
			continue

		rel = _transform_point_3x4(T_local_world, state.world_coordinates)
		out.append(AnchoredRelation(state.assoc_id, state.name, "ok", rel, state.in_view))
	return out


def relation_for_pair(
	video_id: str,
	time_sec: float,
	object_a_assoc_id: str,
	object_b_assoc_id: str,
	annotations_root: str | Path,
	fps: float = 30.0,
) -> list[float]:
	"""Return the single egocentric vector of object A relative to anchor B.

	Inputs: video id, query time, object A assoc id, object B assoc id, annotation root, fps.
	Computes: anchored relations with B as anchor, then selects object A.
	Outputs: [dx, dy, dz] in B-centric camera-aligned coordinates, or raises if unavailable.
	"""
	relations = compute_anchored_relations(
		video_id=video_id,
		time_sec=time_sec,
		anchor_assoc_id=object_b_assoc_id,
		annotations_root=annotations_root,
		fps=fps,
	)
	for rel in relations:
		if rel.assoc_id == object_a_assoc_id:
			if rel.a_relative_to_b_egocentric is None:
				raise ValueError(f"Object A ({object_a_assoc_id}) has no stable relative coordinate at queried time")
			return rel.a_relative_to_b_egocentric
	raise KeyError(f"Object A ({object_a_assoc_id}) not found in relations")
