from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

from in_view_determination import choose_track_for_time, get_mask_from_track, load_json


@dataclass(frozen=True)
class AbsoluteAnswer:
	"""Resolved absolute fixture answer and multiple-choice set for object A.

	Inputs: video/time/object A plus fixture vocabulary and choice count.
	Computes: stable fixture at query time and distractor sampling.
	Outputs: immutable answer payload with choices and correct index.
	"""
	correct_fixture: str
	choices: list[str]
	correct_idx: int


def build_fixture_vocabulary(
	annotations_root: str | Path,
	video_ids: Iterable[str] | None = None,
) -> list[str]:
	"""Build a sorted fixture vocabulary from mask annotations.

	When video_ids is provided, vocabulary is restricted to those videos.
	"""
	annotations_root = Path(annotations_root)
	mask_info = load_json(annotations_root / "scene-and-object-movements" / "mask_info.json")

	vocab: set[str] = set()
	selected_video_ids = set(video_ids) if video_ids is not None else None
	for video_id, masks in mask_info.items():
		if selected_video_ids is not None and video_id not in selected_video_ids:
			continue
		for entry in masks.values():
			fixture = entry.get("fixture")
			if fixture:
				vocab.add(str(fixture))
	return sorted(vocab)


def resolve_fixture_for_object_at_time(
	video_id: str,
	time_sec: float,
	object_a_assoc_id: str,
	annotations_root: str | Path,
) -> str:
	"""Resolve object A's fixture at query time using the shared track policy.

	Raises ValueError when a stable fixture cannot be resolved.
	"""
	annotations_root = Path(annotations_root)
	assoc_info = load_json(annotations_root / "scene-and-object-movements" / "assoc_info.json")
	mask_info = load_json(annotations_root / "scene-and-object-movements" / "mask_info.json")

	if video_id not in assoc_info:
		raise KeyError(f"Video {video_id} not found in assoc_info.json")
	if video_id not in mask_info:
		raise KeyError(f"Video {video_id} not found in mask_info.json")

	video_objects = assoc_info[video_id]
	if object_a_assoc_id not in video_objects:
		raise KeyError(f"Object A assoc_id {object_a_assoc_id} not found in video {video_id}")

	obj = video_objects[object_a_assoc_id]
	track, mode, _ = choose_track_for_time(obj.get("tracks", []), time_sec)
	if track is None:
		raise ValueError("No track available around queried time for object A")
	if mode == "in_motion":
		raise ValueError("Object A is in motion at queried time; fixture is not treated as stable")

	pick = "latest" if mode == "past" else "first"
	mask = get_mask_from_track(mask_info[video_id], track, pick=pick)
	if mask is None:
		raise ValueError("No valid mask available for object A in selected track")

	fixture = mask.get("fixture")
	if not fixture:
		raise ValueError("Fixture is missing for object A at queried context")
	return str(fixture)


def build_absolute_choices(
	correct_fixture: str,
	fixture_vocabulary: list[str],
	num_choices: int = 5,
	rng: random.Random | None = None,
) -> tuple[list[str], int]:
	"""Build unique choices with exactly one correct fixture."""
	if num_choices < 2:
		raise ValueError("num_choices must be >= 2")

	rng = rng or random.Random()
	vocab_set = {str(x) for x in fixture_vocabulary if str(x)}
	vocab_set.discard(correct_fixture)
	distractors = sorted(vocab_set)

	if len(distractors) < num_choices - 1:
		raise ValueError(
			f"Not enough distractors: need {num_choices - 1}, found {len(distractors)}"
		)

	chosen_distractors = rng.sample(distractors, k=num_choices - 1)
	choices = [correct_fixture] + chosen_distractors
	rng.shuffle(choices)
	correct_idx = choices.index(correct_fixture)
	return choices, correct_idx


def determine_absolute_answer(
	video_id: str,
	time_sec: float,
	object_a_assoc_id: str,
	annotations_root: str | Path,
	num_choices: int = 5,
	fixture_vocabulary: list[str] | None = None,
	vocabulary_video_ids: Iterable[str] | None = None,
	rng: random.Random | None = None,
) -> AbsoluteAnswer:
	"""Resolve absolute fixture answer for one keyframe/object pair."""
	correct_fixture = resolve_fixture_for_object_at_time(
		video_id=video_id,
		time_sec=time_sec,
		object_a_assoc_id=object_a_assoc_id,
		annotations_root=annotations_root,
	)

	if fixture_vocabulary is None:
		fixture_vocabulary = build_fixture_vocabulary(
			annotations_root=annotations_root,
			video_ids=vocabulary_video_ids,
		)

	choices, correct_idx = build_absolute_choices(
		correct_fixture=correct_fixture,
		fixture_vocabulary=fixture_vocabulary,
		num_choices=num_choices,
		rng=rng,
	)
	return AbsoluteAnswer(correct_fixture=correct_fixture, choices=choices, correct_idx=correct_idx)
