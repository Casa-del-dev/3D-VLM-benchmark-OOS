from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import random
from typing import Any

from abs_answer_determ import build_fixture_vocabulary, determine_absolute_answer
from anchored_coords import pick_central_anchor
from in_view_determination import DEFAULT_INTERMEDIATE_ROOT, determine_in_view_objects, load_json, load_jsonl
from key_frame_generator import KeyFrameCandidate, generate_key_frames_for_videos
from relative_answer_determ import determine_relative_answer_for_pair
from camera_rotation_determ import determine_camera_rotation_answer,is_rotation_sample_stable
@dataclass(frozen=True)
class GenerationConfig:
	annotations_root: Path
	sampling_fps: float
	out_of_sight_horizon_sec: float
	max_questions_per_video: int
	absolute_enabled: bool
	relative_enabled: bool
	camera_rotation_enabled: bool
	relative_border_tolerance_deg: float
	absolute_num_choices: int
	random_seed: int
	videos: list[str]
	participants: list[str]
	output_json: Path
	fixed_clip_start_earlier_sec: float
	force_clip_start_to_video_start: bool
	camera_rotation_axis: str
	camera_rotation_no_motion_thresh_deg: float


def _load_yaml(path: Path) -> dict[str, Any]:
	"""Load YAML configuration with a lazy PyYAML dependency check."""
	try:
		yaml_module = importlib.import_module("yaml")
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"PyYAML is required to read oos_location_recall_config.yaml. "
			"Install it with: pip install pyyaml"
		) from exc
	with path.open("r", encoding="utf-8") as f:
		return yaml_module.safe_load(f) or {}


def _format_horizon_token(horizon_sec: float) -> str:
	"""Format horizon seconds into ID-safe token form (e.g., h2p0)."""
	return f"h{horizon_sec:.1f}".replace(".", "p")


def _format_time_hms_1dp(time_sec: float) -> str:
	"""Format seconds as HH:MM:SS.s with one decimal place."""
	t = max(0.0, float(time_sec))
	hours = int(t // 3600)
	minutes = int((t % 3600) // 60)
	seconds = t - (hours * 3600 + minutes * 60)
	return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"


def _time_token(time_sec: float, input_key: str = "video 1") -> str:
	"""Build the HD-EPIC style <TIME ...> token for prompts."""
	return f"<TIME {_format_time_hms_1dp(time_sec)} {input_key}>"




def _infer_video_time_window_sec(
	video_id: str,
	annotations_root: Path,
	fps_for_frame_lookup: float = 30.0,
	intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT,
) -> tuple[float, float]:
	"""Infer the valid video time window from framewise metadata."""
	participant_id = video_id.split("-")[0]
	framewise_path = annotations_root / intermediate_root / participant_id / video_id / "framewise_info.jsonl"
	rows = load_jsonl(framewise_path)
	frame_indices = [int(r["frame_index"]) for r in rows if r.get("frame_index") is not None]
	if not frame_indices:
		raise ValueError(f"No valid frame_index entries found for video {video_id}")
	min_frame = min(frame_indices)
	max_frame = max(frame_indices)
	return min_frame / fps_for_frame_lookup, max_frame / fps_for_frame_lookup


def _load_config(config_path: Path) -> GenerationConfig:
	"""Parse YAML and materialize a typed generation configuration."""
	root = config_path.parent
	raw = _load_yaml(config_path)

	inputs = raw.get("inputs", {})
	question_classes = raw.get("question_classes", {})
	relative_cfg = raw.get("relative", {})
	absolute_cfg = raw.get("absolute", {})
	camera_cfg = raw.get("camera_rotation", {})

	return GenerationConfig(
		annotations_root=(root / raw["annotations_root"]).resolve(),
		sampling_fps=float(raw.get("sampling_fps", 2.0)),
		out_of_sight_horizon_sec=float(raw.get("out_of_sight_horizon_sec", 2.0)),
		max_questions_per_video=int(raw.get("max_questions_per_video", 20)),
		absolute_enabled=bool(question_classes.get("absolute", True)),
		relative_enabled=bool(question_classes.get("relative", True)),
		camera_rotation_enabled=bool(question_classes.get("camera_rotation", True)),

		relative_border_tolerance_deg=float(relative_cfg.get("border_tolerance_deg", 10.0)),
		absolute_num_choices=int(absolute_cfg.get("num_choices", 5)),
		random_seed=int(raw.get("random_seed", 42)),
		videos=[str(v) for v in inputs.get("videos", [])],
		participants=[str(p) for p in inputs.get("participants", [])],
		output_json=(root / raw.get("output_json", "oos_location_recall_questions.json")).resolve(),
		
		fixed_clip_start_earlier_sec=float(raw.get("fixed_clip_start_earlier_sec", 0.0)),
		force_clip_start_to_video_start=bool(raw.get("force_clip_start_to_video_start", False)),
		camera_rotation_axis=str(camera_cfg.get("axis", "yaw")),
		camera_rotation_no_motion_thresh_deg=float(camera_cfg.get("no_motion_thresh_deg", 5.0)),
	)


def _select_video_ids(cfg: GenerationConfig) -> list[str]:
	"""Select target video IDs from explicit list or participant filter."""
	if cfg.videos:
		return sorted(cfg.videos)

	assoc_info = load_json(cfg.annotations_root / "scene-and-object-movements" / "assoc_info.json")
	all_video_ids = sorted(assoc_info.keys())
	if not cfg.participants:
		return all_video_ids

	participants = set(cfg.participants)
	return [vid for vid in all_video_ids if vid.split("-")[0] in participants]


def _is_object_out_of_view(candidate: KeyFrameCandidate, annotations_root: Path) -> bool:
	"""Re-check that the candidate object is out of view at query time."""
	states = determine_in_view_objects(
		video_id=candidate.video_id,
		time_sec=candidate.query_time_sec,
		annotations_root=annotations_root,
		fps=30.0,
	)
	for state in states:
		if state.assoc_id == candidate.assoc_id:
			return not bool(state.status == "ok" and state.in_view)
	return False


def generate_questions_from_config(cfg: GenerationConfig) -> dict[str, dict[str, Any]]:
	"""Generate absolute/relative OOS questions from pipeline configuration."""
	video_ids = _select_video_ids(cfg)
	if not video_ids:
		return {}

	horizon = cfg.out_of_sight_horizon_sec
	horizon_token = _format_horizon_token(horizon)
	rng = random.Random(cfg.random_seed)

	keyframes_by_video = generate_key_frames_for_videos(
		video_ids=video_ids,
		annotations_root=cfg.annotations_root,
		horizon_sec=horizon,
		max_questions_per_video=cfg.max_questions_per_video,
		sampling_fps=cfg.sampling_fps,
	)

	fixture_vocab: list[str] | None = None
	if cfg.absolute_enabled:
		fixture_vocab = build_fixture_vocabulary(
			annotations_root=cfg.annotations_root,
			video_ids=video_ids,
		)

	results: dict[str, dict[str, Any]] = {}
	abs_idx = 0
	rel_idx = 0
	cam_idx = 0

	for video_id in video_ids:
		candidates = sorted(keyframes_by_video.get(video_id, []), key=lambda c: c.query_time_sec)
		video_start_sec, _ = _infer_video_time_window_sec(video_id=video_id, annotations_root=cfg.annotations_root)
		for cand in candidates:
			if cand.oos_duration_sec + 1e-9 < horizon:
				continue
			if not _is_object_out_of_view(cand, cfg.annotations_root):
				continue

			if cfg.force_clip_start_to_video_start:
				shifted_clip_start_time_sec = video_start_sec
			else:
				shifted_clip_start_time_sec = cand.clip_start_time_sec - cfg.fixed_clip_start_earlier_sec
				if shifted_clip_start_time_sec < video_start_sec - 1e-9:
					continue
			
			query_time_in_clip_sec = cand.query_time_sec - shifted_clip_start_time_sec
			clip_duration_sec = cand.clip_end_time_sec - shifted_clip_start_time_sec

			base_fields = {
				"inputs": {"video 1": {"id": cand.video_id}},
				"video_id": cand.video_id,
				"query_time_sec": cand.query_time_sec,
				"query_time_in_clip_sec": query_time_in_clip_sec,
				"horizon_sec": horizon,
				"clip_start_time_sec": shifted_clip_start_time_sec,
				"clip_end_time_sec": cand.clip_end_time_sec,
				"clip_duration_sec": clip_duration_sec,
				"object_a_assoc_id": cand.assoc_id,
				"object_a_name": cand.object_name,
				"generation_info": {
					"oos_span_start_sec": cand.oos_span_start_sec,
					"oos_span_end_sec": cand.oos_span_end_sec,
					"oos_duration_sec": cand.oos_duration_sec,
					"sampling_fps": cfg.sampling_fps,
					"random_seed": cfg.random_seed,
					"relocation_score": cand.relocation_score,
					"fixed_clip_start_earlier_sec": cfg.fixed_clip_start_earlier_sec,
					"original_clip_start_time_sec": cand.clip_start_time_sec,
				},
			}

			if cfg.absolute_enabled and fixture_vocab is not None:
				try:
					abs_answer = determine_absolute_answer(
						video_id=cand.video_id,
						time_sec=cand.query_time_sec,
						object_a_assoc_id=cand.assoc_id,
						annotations_root=cfg.annotations_root,
						num_choices=cfg.absolute_num_choices,
						fixture_vocabulary=fixture_vocab,
						rng=rng,
					)
					time_tok = _time_token(query_time_in_clip_sec, input_key="video 1")
					qid = f"oos_abs_fixture_location_{horizon_token}_{abs_idx}"
					results[qid] = {
						**base_fields,
						"question": (
							f"The {cand.object_name} was seen earlier in the video. "
							f"At the current time {time_tok} what fixture is the object nearest to?"
						),
						"choices": abs_answer.choices,
						"correct_idx": abs_answer.correct_idx,
						"object_b_assoc_id": None,
						"object_b_name": None,
						"question_class": "oos_abs_fixture_location",
					}
					abs_idx += 1
				except (KeyError, ValueError):
					pass

			if cfg.relative_enabled:
				anchor = pick_central_anchor(
					video_id=cand.video_id,
					time_sec=cand.query_time_sec,
					annotations_root=cfg.annotations_root,
					fps=30.0,
				)
				if anchor is None:
					continue
				try:
					rel_answer = determine_relative_answer_for_pair(
						video_id=cand.video_id,
						time_sec=cand.query_time_sec,
						object_a_assoc_id=cand.assoc_id,
						object_b_assoc_id=str(anchor["assoc_id"]),
						annotations_root=cfg.annotations_root,
						fps=30.0,
						border_tolerance_deg=cfg.relative_border_tolerance_deg,
					)
					if rel_answer.correct_idx not in rel_answer.acceptable_idxs:
						continue
					time_tok = _time_token(query_time_in_clip_sec, input_key="video 1")
					qid = f"oos_rel_anchor_location_{horizon_token}_{rel_idx}"
					results[qid] = {
						**base_fields,
						"question": (
							f"{cand.object_name} was seen earlier in the video. "
							f"At the current time {time_tok} what is its position relative to {anchor['name']}?"
						),
						"choices": rel_answer.choices,
						"correct_idx": rel_answer.correct_idx,
						"acceptable_idxs": rel_answer.acceptable_idxs,
						"object_b_assoc_id": str(anchor["assoc_id"]),
						"object_b_name": str(anchor["name"]),
						"question_class": "oos_rel_anchor_location",
					}
					rel_idx += 1
				except (KeyError, ValueError):
					pass
			if cfg.camera_rotation_enabled:
				try:
					cam_answer = determine_camera_rotation_answer(
						video_id=cand.video_id,
						start_time_sec=shifted_clip_start_time_sec,
						end_time_sec=cand.query_time_sec,
						annotations_root=cfg.annotations_root,
						no_motion_thresh_deg=cfg.camera_rotation_no_motion_thresh_deg,
					)
					if not is_rotation_sample_stable(
						cam_answer.signed_rotation_deg,
						no_motion_thresh_deg=cfg.camera_rotation_no_motion_thresh_deg,
					):
						continue

					if abs(cam_answer.signed_rotation_deg) > 20.0: continue
			
					time_tok = _time_token(query_time_in_clip_sec, input_key="video 1")
					qid = f"camera_rotation_{horizon_token}_{cam_idx}"
					results[qid] = {
						**base_fields,
						"question": (
							f"Between the start of the clip and the current time {time_tok}, "
							f"how did the camera mainly rotate?"
						),
						"choices": cam_answer.choices,
						"correct_idx": cam_answer.correct_idx,
						"camera_signed_rotation_deg": cam_answer.signed_rotation_deg,
						"camera_rotation_direction": cam_answer.direction,
						"camera_rotation_angle_bucket": cam_answer.angle_bucket,
						"object_b_assoc_id": None,
						"object_b_name": None,
						"question_class": "camera_rotation",
					}
					cam_idx += 1
				except (KeyError, ValueError, FileNotFoundError):
					pass

	return results

def save_questions_json(questions: dict[str, dict[str, Any]], output_path: Path) -> None:
	"""Write generated questions to a UTF-8 JSON file."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(questions, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for question generation entrypoint."""
	parser = argparse.ArgumentParser(description="Generate HD-EPIC OOS location recall VQA questions")
	parser.add_argument(
		"--config",
		type=Path,
		default=Path(__file__).resolve().parent / "oos_location_recall_config.yaml",
		help="Path to generator YAML config",
	)
	parser.add_argument(
		"--output_json",
		type=Path,
		default=None,
		help="Optional output JSON path override",
	)
	parser.add_argument(
		"--videoStart",
		action="store_true",
		help="Override YAML: force clip to start at video beginning",
	)
	parser.add_argument(
		"--clipOffset",
		type=float,
		default=None,
		help="Override YAML: set fixed_clip_start_earlier_sec",
	)
	return parser.parse_args()


def main() -> None:
	"""CLI entrypoint: load config, generate questions, and persist JSON."""
	args = parse_args()
	base_cfg = _load_config(args.config.resolve())

	force_video_start = base_cfg.force_clip_start_to_video_start
	clip_offset = base_cfg.fixed_clip_start_earlier_sec

	if args.videoStart:
		force_video_start = True

	if args.clipOffset is not None:
		clip_offset = args.clipOffset

	output_json = base_cfg.output_json
	if args.output_json is not None:
		output_json = args.output_json.resolve()

	cfg = GenerationConfig(
		annotations_root=base_cfg.annotations_root,
		sampling_fps=base_cfg.sampling_fps,
		out_of_sight_horizon_sec=base_cfg.out_of_sight_horizon_sec,
		max_questions_per_video=base_cfg.max_questions_per_video,
		absolute_enabled=base_cfg.absolute_enabled,
		relative_enabled=base_cfg.relative_enabled,
		camera_rotation_enabled=base_cfg.camera_rotation_enabled,
		relative_border_tolerance_deg=base_cfg.relative_border_tolerance_deg,
		absolute_num_choices=base_cfg.absolute_num_choices,
		random_seed=base_cfg.random_seed,
		videos=base_cfg.videos,
		participants=base_cfg.participants,
		output_json=output_json,
		fixed_clip_start_earlier_sec=clip_offset,
		force_clip_start_to_video_start=force_video_start,
		camera_rotation_axis=base_cfg.camera_rotation_axis,
		camera_rotation_no_motion_thresh_deg=base_cfg.camera_rotation_no_motion_thresh_deg,
	)

	questions = generate_questions_from_config(cfg)
	save_questions_json(questions, cfg.output_json)
	print(f"Generated {len(questions)} questions")
	print(f"Saved JSON: {cfg.output_json}")


if __name__ == "__main__":
	main()
