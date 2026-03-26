from __future__ import annotations

import argparse
import shutil
from io import BytesIO
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


# Allow running this script directly from the tests folder.
SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
if str(MODULE_DIR) not in sys.path:
	sys.path.insert(0, str(MODULE_DIR))

from in_view_determination import determine_in_view_objects, load_frame_context, project_fisheye624  # noqa: E402
from in_view_track_generator import generate_in_view_tracks  # noqa: E402
from question_generator import GenerationConfig, _load_config, generate_questions_from_config  # noqa: E402

"""
EREN SETUP:
DEFAULT_CONFIG = MODULE_DIR / "oos_location_recall_config.yaml"
DEFAULT_DATA_ROOT_REL = "../../../../../"
DEFAULT_VIDEO_REL_TEMPLATE = "HD-EPIC/Videos/{participant}/{video_id}.mp4"
DEFAULT_OUTPUT_DIR_REL = "../../../outputs/oos_location_recall_debug"
"""

"""
IVO SETUP:
DEFAULT_CONFIG = MODULE_DIR / "oos_location_recall_config.yaml"
DEFAULT_DATA_ROOT_REL = "../../../../../data"
DEFAULT_VIDEO_REL_TEMPLATE = "HD-EPIC/Videos/{participant}/{video_id}.mp4"
DEFAULT_OUTPUT_DIR_REL = "../../../outputs/oos_location_recall_debug"
"""

DEFAULT_CONFIG = MODULE_DIR / "oos_location_recall_config.yaml"
DEFAULT_DATA_ROOT_REL = "../../../../../"
DEFAULT_VIDEO_REL_TEMPLATE = "HD-EPIC/Videos/{participant}/{video_id}.mp4"
DEFAULT_OUTPUT_DIR_REL = "../../../outputs/oos_location_recall_debug"


def _select_evenly(values: list[float], n: int) -> list[float]:
	"""Select up to n approximately evenly spaced values preserving order."""
	if len(values) <= n:
		return values
	if n <= 1:
		return [values[len(values) // 2]]

	picks: list[float] = []
	for i in range(n):
		idx = round(i * (len(values) - 1) / (n - 1))
		picks.append(values[idx])

	# Preserve order while removing accidental duplicates from rounding.
	out: list[float] = []
	seen: set[float] = set()
	for v in picks:
		if v not in seen:
			out.append(v)
			seen.add(v)
	return out


def _load_pillow_modules() -> tuple[Any, Any, Any]:
	"""Import Pillow modules lazily with a clear dependency error."""
	try:
		image_module = importlib.import_module("PIL.Image")
		draw_module = importlib.import_module("PIL.ImageDraw")
		ops_module = importlib.import_module("PIL.ImageOps")
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"Pillow is required for visual outputs. Install it with: pip install pillow"
		) from exc
	return image_module, draw_module, ops_module


def _read_frame_at_time(video_path: Path, t_sec: float) -> Any:
	"""Decode a single RGB frame at the requested timestamp using ffmpeg."""
	image_module, _, _ = _load_pillow_modules()
	cmd = [
		"ffmpeg",
		"-hide_banner",
		"-loglevel",
		"error",
		"-ss",
		f"{t_sec:.3f}",
		"-i",
		str(video_path),
		"-frames:v",
		"1",
		"-f",
		"image2pipe",
		"-vcodec",
		"png",
		"pipe:1",
	]
	proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if proc.returncode != 0 or not proc.stdout:
		raise RuntimeError(f"Failed to read frame at t={t_sec:.3f}s: {proc.stderr.decode('utf-8', errors='ignore')}")
	return image_module.open(BytesIO(proc.stdout)).convert("RGB")


def _get_object_state(video_id: str, t_sec: float, assoc_id: str, annotations_root: Path) -> Any | None:
	"""Return one object's state at time t, or None when absent."""
	states = determine_in_view_objects(
		video_id=video_id,
		time_sec=t_sec,
		annotations_root=annotations_root,
		fps=30.0,
	)
	return next((s for s in states if s.assoc_id == assoc_id), None)


def _draw_multiline_text(draw: Any, x: int, y: int, lines: list[str], fill: tuple[int, int, int]) -> None:
	"""Draw stacked text lines with fixed line spacing."""
	line_h = 20
	for i, line in enumerate(lines):
		draw.text((x, y + i * line_h), line, fill=fill)


def _scale_to_frame(pixel_xy: list[float], frame_w: int, frame_h: int, calib_width: int, calib_height: int) -> tuple[int, int]:
	"""Scale calibration-space pixel coordinates into rendered frame size."""
	px = int(round(pixel_xy[0] * (frame_w / float(calib_width))))
	py = int(round(pixel_xy[1] * (frame_h / float(calib_height))))
	return px, py


def _draw_arrow(draw: Any, p0: tuple[int, int], p1: tuple[int, int], color: tuple[int, int, int], width: int = 3) -> None:
	"""Draw a line arrow with a small triangular head."""
	draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)
	# Small arrow head.
	dx = p1[0] - p0[0]
	dy = p1[1] - p0[1]
	mag = (dx * dx + dy * dy) ** 0.5
	if mag < 1e-6:
		return
	ux, uy = dx / mag, dy / mag
	left = (int(p1[0] - 10 * ux - 5 * uy), int(p1[1] - 10 * uy + 5 * ux))
	right = (int(p1[0] - 10 * ux + 5 * uy), int(p1[1] - 10 * uy - 5 * ux))
	draw.polygon([p1, left, right], fill=color)


def _draw_relative_query_overlay(
	draw: Any,
	frame: Any,
	video_id: str,
	time_sec: float,
	question_obj: dict[str, Any],
	assoc_id: str,
	annotations_root: Path,
	calib_width: int,
	calib_height: int,
) -> None:
	"""Overlay anchor axes and A-direction cue for relative-query debug frames."""
	if question_obj.get("question_class") != "oos_rel_anchor_location":
		return

	anchor_assoc_id = question_obj.get("object_b_assoc_id")
	if anchor_assoc_id is None:
		return

	anchor_state = _get_object_state(video_id, time_sec, str(anchor_assoc_id), annotations_root)
	a_state = _get_object_state(video_id, time_sec, assoc_id, annotations_root)
	if anchor_state is None or anchor_state.projected_pixel is None or anchor_state.camera_coordinates is None:
		return

	w, h = frame.size
	b_px = _scale_to_frame(anchor_state.projected_pixel, w, h, calib_width, calib_height)

	# Mark anchor B.
	draw.ellipse((b_px[0] - 12, b_px[1] - 12, b_px[0] + 12, b_px[1] + 12), outline=(0, 255, 255), width=3)
	draw.text((b_px[0] + 16, b_px[1] - 12), "B(anchor)", fill=(0, 180, 180))

	ctx = load_frame_context(video_id=video_id, time_sec=time_sec, annotations_root=annotations_root, fps=30.0)
	bx, by, bz = [float(v) for v in anchor_state.camera_coordinates]
	axis_len_m = 0.08
	axis_points_cam = {
		"x": [bx + axis_len_m, by, bz],
		"y": [bx, by + axis_len_m, bz],
		"z": [bx, by, bz + axis_len_m],
	}
	axis_colors = {
		"x": (255, 60, 60),
		"y": (60, 220, 60),
		"z": (60, 120, 255),
	}
	for axis, p_cam in axis_points_cam.items():
		pix, _, ok = project_fisheye624(p_cam, ctx.projection_params)
		if not ok or pix is None:
			continue
		p_tip = _scale_to_frame(pix, w, h, calib_width, calib_height)
		_draw_arrow(draw, b_px, p_tip, axis_colors[axis], width=3)
		draw.text((p_tip[0] + 4, p_tip[1] + 2), axis, fill=axis_colors[axis])

	# Draw a yellow direction arrow from B toward A with 2x the basis-vector length.
	if a_state is not None and a_state.camera_coordinates is not None:
		ax, ay, az = [float(v) for v in a_state.camera_coordinates]
		dx, dy, dz = ax - bx, ay - by, az - bz
		mag = (dx * dx + dy * dy + dz * dz) ** 0.5
		if mag > 1e-9:
			target_len_m = 2.0 * axis_len_m
			target_cam = [
				bx + target_len_m * (dx / mag),
				by + target_len_m * (dy / mag),
				bz + target_len_m * (dz / mag),
			]
			tip_pix, _, ok = project_fisheye624(target_cam, ctx.projection_params)
			if ok and tip_pix is not None:
				a_tip = _scale_to_frame(tip_pix, w, h, calib_width, calib_height)
				_draw_arrow(draw, b_px, a_tip, (255, 215, 0), width=3)
				draw.text((a_tip[0] + 8, a_tip[1] + 6), "A dir (2x)", fill=(220, 180, 0))


def _render_object_grid(
	video_path: Path,
	video_id: str,
	assoc_id: str,
	object_name: str,
	times_sec: list[float],
	annotations_root: Path,
	calib_width: int,
	calib_height: int,
	title: str,
) -> Any:
	"""Render a 2x5 grid of frames for one object over selected times."""
	image_module, draw_module, ops_module = _load_pillow_modules()
	cell_w, cell_h = 512, 288
	cells: list[Any] = []

	for t_sec in times_sec:
		frame = _read_frame_at_time(video_path, t_sec)
		state = _get_object_state(video_id, t_sec, assoc_id, annotations_root)
		in_view = bool(state is not None and state.status == "ok" and state.in_view)
		color = (0, 180, 0) if in_view else (200, 0, 0)
		draw = draw_module.Draw(frame)

		if state is not None and state.projected_pixel is not None:
			w, h = frame.size
			px = int(round(state.projected_pixel[0] * (w / float(calib_width))))
			py = int(round(state.projected_pixel[1] * (h / float(calib_height))))
			draw.ellipse((px - 12, py - 12, px + 12, py + 12), fill=(255, 255, 0), outline=(0, 0, 0), width=2)

		draw.text((12, 10), f"t={t_sec:.2f}s", fill=color)
		draw.text((12, 30), f"{object_name} ({assoc_id})", fill=color)
		draw.text((12, 50), f"in_view={in_view}", fill=color)

		frame = frame.resize((cell_w, cell_h), image_module.Resampling.BILINEAR)
		frame = ops_module.expand(frame, border=3, fill=color)
		cells.append(frame)

	while len(cells) < 10:
		blank = image_module.new("RGB", (cell_w + 6, cell_h + 6), color=(255, 255, 255))
		draw_module.Draw(blank).text((cell_w // 2 - 20, cell_h // 2 - 8), "N/A", fill=(0, 0, 0))
		cells.append(blank)

	cells = cells[:10]
	row1 = image_module.new("RGB", (sum(im.width for im in cells[:5]), max(im.height for im in cells[:5])), color=(255, 255, 255))
	x = 0
	for im in cells[:5]:
		row1.paste(im, (x, 0))
		x += im.width

	row2 = image_module.new("RGB", (sum(im.width for im in cells[5:10]), max(im.height for im in cells[5:10])), color=(255, 255, 255))
	x = 0
	for im in cells[5:10]:
		row2.paste(im, (x, 0))
		x += im.width

	grid = image_module.new("RGB", (row1.width, row1.height + row2.height), color=(255, 255, 255))
	grid.paste(row1, (0, 0))
	grid.paste(row2, (0, row1.height))

	banner = image_module.new("RGB", (grid.width, 60), color=(255, 255, 255))
	draw_module.Draw(banner).text((16, 20), title, fill=(20, 20, 20))
	full = image_module.new("RGB", (grid.width, grid.height + banner.height), color=(255, 255, 255))
	full.paste(banner, (0, 0))
	full.paste(grid, (0, banner.height))
	return full


def _find_last_seen_time(track_times: list[float], visibility: list[bool], query_time_sec: float) -> float | None:
	"""Find latest sampled visible timestamp strictly before query time."""
	last_seen: float | None = None
	for t, vis in zip(track_times, visibility):
		if t >= query_time_sec:
			break
		if vis:
			last_seen = t
	return last_seen


def _render_question_frame(
	video_path: Path,
	video_id: str,
	question_id: str,
	question_obj: dict[str, Any],
	assoc_id: str,
	object_name: str,
	time_sec: float,
	annotations_root: Path,
	calib_width: int,
	calib_height: int,
	label: str,
) -> Any:
	"""Render one annotated frame panel for a generated question instance."""
	image_module, draw_module, _ = _load_pillow_modules()
	frame = _read_frame_at_time(video_path, time_sec)
	draw = draw_module.Draw(frame)

	state = _get_object_state(video_id, time_sec, assoc_id, annotations_root)
	if state is not None and state.projected_pixel is not None:
		w, h = frame.size
		px, py = _scale_to_frame(state.projected_pixel, w, h, calib_width, calib_height)
		draw.ellipse((px - 14, py - 14, px + 14, py + 14), fill=(255, 255, 0), outline=(0, 0, 0), width=2)

	if label == "QUERY FRAME":
		_draw_relative_query_overlay(
			draw=draw,
			frame=frame,
			video_id=video_id,
			time_sec=time_sec,
			question_obj=question_obj,
			assoc_id=assoc_id,
			annotations_root=annotations_root,
			calib_width=calib_width,
			calib_height=calib_height,
		)

	choices = question_obj.get("choices", [])
	correct_idx = int(question_obj.get("correct_idx", -1))
	acceptable_idxs = question_obj.get("acceptable_idxs")

	text_lines = [
		f"{label}: t={time_sec:.2f}s",
		f"qid={question_id}",
		f"object={object_name} ({assoc_id})",
		f"question={question_obj.get('question', '')}",
	]
	for i, ch in enumerate(choices):
		mark = "*" if i == correct_idx else " "
		text_lines.append(f"{mark} choice[{i}]={ch}")
	if acceptable_idxs is not None:
		text_lines.append(f"acceptable_idxs={acceptable_idxs}")

	# Draw a white strip for text for readability.
	panel_h = min(340, 20 * (len(text_lines) + 1))
	panel = image_module.new("RGB", (frame.width, panel_h), color=(255, 255, 255))
	panel_draw = draw_module.Draw(panel)
	_draw_multiline_text(panel_draw, 12, 8, text_lines, fill=(10, 10, 10))

	out = image_module.new("RGB", (frame.width, frame.height + panel_h), color=(255, 255, 255))
	out.paste(panel, (0, 0))
	out.paste(frame, (0, panel_h))
	return out


def _ensure_ffmpeg() -> None:
	"""Ensure ffmpeg exists in PATH before visualization work."""
	if shutil.which("ffmpeg") is None and shutil.which("ffmpeg.exe") is None:
		raise RuntimeError("ffmpeg is required but was not found in PATH.")


def parse_args() -> argparse.Namespace:
	"""Parse CLI args for the OOS question visual review script."""
	parser = argparse.ArgumentParser(description="Manual visual tester for OOS location recall generation")
	parser.add_argument("--video_id", type=str, required=True, help="Single video id to test, e.g. P01-20240203-184045")
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to oos_location_recall_config.yaml")
	parser.add_argument("--data_root", type=Path, default=(SCRIPT_DIR / DEFAULT_DATA_ROOT_REL).resolve(), help="Root containing HD-EPIC/Videos")
	parser.add_argument("--video_path", type=Path, default=None, help="Optional direct mp4 path override")
	parser.add_argument("--output_dir", type=Path, default=(SCRIPT_DIR / DEFAULT_OUTPUT_DIR_REL).resolve(), help="Folder for debug outputs")
	parser.add_argument("--samples_per_class", type=int, default=10, help="How many in-view and out-of-view frames per object")
	return parser.parse_args()


def main() -> None:
	"""Run end-to-end debug generation and visualization for one video."""
	args = parse_args()
	_ensure_ffmpeg()

	video_id = args.video_id
	participant = video_id.split("-")[0]
	video_path = args.video_path or (args.data_root / DEFAULT_VIDEO_REL_TEMPLATE.format(participant=participant, video_id=video_id))
	if not video_path.exists():
		raise FileNotFoundError(f"Video file not found: {video_path}")

	base_cfg = _load_config(args.config.resolve())
	cfg = GenerationConfig(
		annotations_root=base_cfg.annotations_root,
		sampling_fps=base_cfg.sampling_fps,
		out_of_sight_horizon_sec=base_cfg.out_of_sight_horizon_sec,
		max_questions_per_video=base_cfg.max_questions_per_video,
		absolute_enabled=base_cfg.absolute_enabled,
		relative_enabled=base_cfg.relative_enabled,
		relative_border_tolerance_deg=base_cfg.relative_border_tolerance_deg,
		absolute_num_choices=base_cfg.absolute_num_choices,
		random_seed=base_cfg.random_seed,
		videos=[video_id],
		participants=[],
		output_json=(args.output_dir / f"{video_id}_questions.json").resolve(),
	)

	questions = generate_questions_from_config(cfg)
	if not questions:
		raise RuntimeError("No questions generated for the provided video.")

	args.output_dir.mkdir(parents=True, exist_ok=True)
	(cfg.output_json.parent).mkdir(parents=True, exist_ok=True)
	with cfg.output_json.open("w", encoding="utf-8") as f:
		json.dump(questions, f, indent=2, ensure_ascii=False)

	tracks = generate_in_view_tracks(
		video_id=video_id,
		annotations_root=cfg.annotations_root,
		sampling_fps=cfg.sampling_fps,
		fps_for_frame_lookup=30.0,
	)
	if not tracks:
		raise RuntimeError("No visibility tracks generated for requested video.")

	# Use one context read for calibration dimensions.
	any_track = next(iter(tracks.values()))
	ctx = load_frame_context(
		video_id=video_id,
		time_sec=any_track.sampled_times_sec[0],
		annotations_root=cfg.annotations_root,
		fps=30.0,
	)

	object_grid_dir = args.output_dir / "object_grids"
	question_frame_dir = args.output_dir / "question_frames"
	object_grid_dir.mkdir(parents=True, exist_ok=True)
	question_frame_dir.mkdir(parents=True, exist_ok=True)

	# Generate object grids for all object A instances appearing in generated questions.
	objects: dict[str, str] = {}
	for q in questions.values():
		objects[str(q["object_a_assoc_id"])] = str(q["object_a_name"])

	object_grid_manifest: dict[str, dict[str, str]] = {}
	for assoc_id, object_name in objects.items():
		tr = tracks.get(assoc_id)
		if tr is None:
			continue

		in_times = [t for t, v in zip(tr.sampled_times_sec, tr.visibility_samples) if v]
		out_times = [t for t, v in zip(tr.sampled_times_sec, tr.visibility_samples) if not v]
		chosen_in = _select_evenly(in_times, args.samples_per_class)
		chosen_out = _select_evenly(out_times, args.samples_per_class)

		in_grid = _render_object_grid(
			video_path=video_path,
			video_id=video_id,
			assoc_id=assoc_id,
			object_name=object_name,
			times_sec=chosen_in,
			annotations_root=cfg.annotations_root,
			calib_width=ctx.image_width,
			calib_height=ctx.image_height,
			title=f"IN-VIEW samples ({object_name}, {assoc_id})",
		)
		out_grid = _render_object_grid(
			video_path=video_path,
			video_id=video_id,
			assoc_id=assoc_id,
			object_name=object_name,
			times_sec=chosen_out,
			annotations_root=cfg.annotations_root,
			calib_width=ctx.image_width,
			calib_height=ctx.image_height,
			title=f"OUT-OF-VIEW samples ({object_name}, {assoc_id})",
		)

		in_path = object_grid_dir / f"{video_id}_{assoc_id}_in_view_grid.png"
		out_path = object_grid_dir / f"{video_id}_{assoc_id}_out_of_view_grid.png"
		in_grid.save(in_path)
		out_grid.save(out_path)

		object_grid_manifest[assoc_id] = {
			"object_name": object_name,
			"in_view_grid": str(in_path),
			"out_of_view_grid": str(out_path),
		}

	question_manifest: dict[str, Any] = {}
	for qid, qobj in questions.items():
		assoc_id = str(qobj["object_a_assoc_id"])
		object_name = str(qobj["object_a_name"])
		query_t = float(qobj["query_time_sec"])

		query_img = _render_question_frame(
			video_path=video_path,
			video_id=video_id,
			question_id=qid,
			question_obj=qobj,
			assoc_id=assoc_id,
			object_name=object_name,
			time_sec=query_t,
			annotations_root=cfg.annotations_root,
			calib_width=ctx.image_width,
			calib_height=ctx.image_height,
			label="QUERY FRAME",
		)
		query_path = question_frame_dir / f"{qid}_query.png"
		query_img.save(query_path)

		last_seen_t: float | None = None
		tr = tracks.get(assoc_id)
		if tr is not None:
			last_seen_t = _find_last_seen_time(tr.sampled_times_sec, tr.visibility_samples, query_t)

		last_seen_path: str | None = None
		if last_seen_t is not None:
			last_seen_img = _render_question_frame(
				video_path=video_path,
				video_id=video_id,
				question_id=qid,
				question_obj=qobj,
				assoc_id=assoc_id,
				object_name=object_name,
				time_sec=last_seen_t,
				annotations_root=cfg.annotations_root,
				calib_width=ctx.image_width,
				calib_height=ctx.image_height,
				label="LAST SEEN (sampled)",
			)
			last_seen_file = question_frame_dir / f"{qid}_last_seen.png"
			last_seen_img.save(last_seen_file)
			last_seen_path = str(last_seen_file)

		question_manifest[qid] = {
			"question": qobj["question"],
			"choices": qobj["choices"],
			"correct_idx": qobj["correct_idx"],
			"acceptable_idxs": qobj.get("acceptable_idxs"),
			"query_time_sec": query_t,
			"query_frame": str(query_path),
			"last_seen_time_sec_sampled": last_seen_t,
			"last_seen_frame": last_seen_path,
		}

	report = {
		"video_id": video_id,
		"video_path": str(video_path),
		"questions_json": str(cfg.output_json),
		"num_questions": len(questions),
		"object_grids": object_grid_manifest,
		"questions": question_manifest,
	}

	report_path = args.output_dir / f"{video_id}_review_manifest.json"
	with report_path.open("w", encoding="utf-8") as f:
		json.dump(report, f, indent=2, ensure_ascii=False)

	print("Done.")
	print(f"Video: {video_path}")
	print(f"Questions generated: {len(questions)}")
	print(f"Question JSON: {cfg.output_json}")
	print(f"Review manifest: {report_path}")
	print(f"Object grids dir: {object_grid_dir}")
	print(f"Question frames dir: {question_frame_dir}")


if __name__ == "__main__":
	main()
