from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from anchored_coords import relation_for_pair


RELATIVE_CHOICES = [
	"above",
	"below",
	"to the right",
	"to the left",
	"in front of",
	"behind",
]


@dataclass(frozen=True)
class RelativeAnswer:
	"""Resolved relative answer labels from an anchored displacement vector.

	Inputs: displacement d = p_A - p_B and border tolerance.
	Computes: dominant-axis label and optional boundary-tolerant second label.
	Outputs: immutable answer payload with fixed-choice indices.
	"""
	vector: list[float]
	choices: list[str]
	correct_idx: int
	acceptable_idxs: list[int]


def _label_for_axis(axis: str, value: float) -> str:
	"""Map signed axis value to the corresponding relative-direction label."""
	if axis == "x":
		return "to the right" if value > 0 else "to the left"
	if axis == "y":
		return "above" if value > 0 else "below"
	if axis == "z":
		return "in front of" if value > 0 else "behind"
	raise ValueError(f"Unknown axis {axis}")


def _choice_index(label: str) -> int:
	"""Return the fixed multiple-choice index for a direction label."""
	return RELATIVE_CHOICES.index(label)


def _dominant_axis(dx: float, dy: float, dz: float) -> str:
	"""Pick dominant axis by absolute magnitude with deterministic tie-breaking."""
	axes = {
		"x": abs(dx),
		"y": abs(dy),
		"z": abs(dz),
	}
	# README priority wording is y, then x, then z under ties.
	priority = {"y": 0, "x": 1, "z": 2}
	return sorted(axes.keys(), key=lambda a: (-axes[a], priority[a]))[0]


def determine_relative_answer_from_vector(
	a_relative_to_b_egocentric: list[float],
	border_tolerance_deg: float = 10.0,
) -> RelativeAnswer:
	"""Determine relative answer from anchored vector"""
	if len(a_relative_to_b_egocentric) != 3:
		raise ValueError("Expected 3D vector [dx, dy, dz]")

	dx, dy, dz = [float(v) for v in a_relative_to_b_egocentric]
	if abs(dx) < 1e-12 and abs(dy) < 1e-12 and abs(dz) < 1e-12:
		raise ValueError("Relative vector is near zero; direction is undefined")

	axis_values = {
		"x": dx,
		"y": dy,
		"z": dz,
	}
	abs_by_axis = {
		"x": abs(dx),
		"y": abs(dy),
		"z": abs(dz),
	}

	dom_axis = _dominant_axis(dx, dy, dz)
	primary_label = _label_for_axis(dom_axis, axis_values[dom_axis])
	correct_idx = _choice_index(primary_label)
	acceptable: set[int] = {correct_idx}

	ordered_axes = sorted(abs_by_axis.keys(), key=lambda a: abs_by_axis[a], reverse=True)
	a1, a2 = ordered_axes[0], ordered_axes[1]
	m1, m2 = abs_by_axis[a1], abs_by_axis[a2]
	if m1 > 1e-12:
		theta = math.degrees(math.atan(m2 / m1))
		if abs(45.0 - theta) <= border_tolerance_deg + 1e-12:
			secondary_label = _label_for_axis(a2, axis_values[a2])
			acceptable.add(_choice_index(secondary_label))

	acceptable_idxs = sorted(acceptable)
	return RelativeAnswer(
		vector=[dx, dy, dz],
		choices=RELATIVE_CHOICES,
		correct_idx=correct_idx,
		acceptable_idxs=acceptable_idxs,
	)


def determine_relative_answer_for_pair(
	video_id: str,
	time_sec: float,
	object_a_assoc_id: str,
	object_b_assoc_id: str,
	annotations_root: str | Path,
	fps: float = 30.0, # Not actually used for now
	border_tolerance_deg: float = 10.0,
) -> RelativeAnswer:
	"""Resolve relative answer for object pair (A, B) at query time."""
	vector = relation_for_pair(
		video_id=video_id,
		time_sec=time_sec,
		object_a_assoc_id=object_a_assoc_id,
		object_b_assoc_id=object_b_assoc_id,
		annotations_root=Path(annotations_root),
		fps=fps,
	)
	return determine_relative_answer_from_vector(
		a_relative_to_b_egocentric=vector,
		border_tolerance_deg=border_tolerance_deg,
	)
