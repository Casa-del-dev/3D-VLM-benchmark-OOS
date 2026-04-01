from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from abs_answer_determ import build_fixture_vocabulary, determine_absolute_answer
from in_view_determination import DEFAULT_INTERMEDIATE_ROOT, determine_in_view_objects, load_frame_context
from key_frame_generator import KeyFrameCandidate, generate_key_frames_for_videos


@dataclass(frozen=True)
class BenchmarkConfig:
    annotations_root: Path
    video_ids: list[str]
    sampling_fps: float = 2.0
    fps_for_frame_lookup: float = 30.0
    out_of_sight_horizon_sec: float = 2.0
    max_questions_per_video: int = 20
    absolute_num_choices: int = 5
    random_seed: int = 42
    back_threshold_deg: float = 135.0
    camera_turn_small_deg: float = 30.0
    camera_turn_back_deg: float = 135.0
    intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT
    output_json: Path | None = None
    pre_context_sec: float = 2.0


@dataclass(frozen=True)
class LastVisibleInfo:
    sampled_time_sec: float
    projected_pixel: list[float] | None
    frame_index: int | None
    status: str


LEFT_RIGHT_BACK_CHOICES = ["left", "right", "back"]
CAMERA_POSE_CHANGE_CHOICES = ["No change", "Rotated left", "Rotated right", "Rotated back"]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load the benchmark config. Install it with: pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _format_horizon_token(horizon_sec: float) -> str:
    return f"h{horizon_sec:.1f}".replace(".", "p")


def _format_time_hms_1dp(time_sec: float) -> str:
    t = max(0.0, float(time_sec))
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = t - (hours * 3600 + minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"


def _time_token(time_sec: float, input_key: str = "video 1") -> str:
    return f"<TIME {_format_time_hms_1dp(time_sec)} {input_key}>"


def _states_by_assoc_id(video_id: str, time_sec: float, annotations_root: Path, fps: float, intermediate_root: str) -> dict[str, Any]:
    states = determine_in_view_objects(
        video_id=video_id,
        time_sec=time_sec,
        annotations_root=annotations_root,
        fps=fps,
        intermediate_root=intermediate_root,
    )
    return {s.assoc_id: s for s in states}


def _find_last_visible_info(candidate: KeyFrameCandidate, cfg: BenchmarkConfig) -> LastVisibleInfo | None:
    step = 1.0 / cfg.sampling_fps
    t = candidate.query_time_sec - step
    while t >= candidate.clip_start_time_sec - 1e-9:
        state = _states_by_assoc_id(
            video_id=candidate.video_id,
            time_sec=t,
            annotations_root=cfg.annotations_root,
            fps=cfg.fps_for_frame_lookup,
            intermediate_root=cfg.intermediate_root,
        ).get(candidate.assoc_id)
        if state is not None and state.status == "ok" and state.in_view:
            return LastVisibleInfo(
                sampled_time_sec=float(t),
                projected_pixel=[float(v) for v in state.projected_pixel] if state.projected_pixel is not None else None,
                frame_index=int(state.frame_number) if state.frame_number is not None else None,
                status="ok",
            )
        t -= step
    return None


def _classify_camera_relative_direction(camera_coordinates: list[float], back_threshold_deg: float = 135.0) -> int:
    x, _, z = [float(v) for v in camera_coordinates]
    angle_deg = math.degrees(math.atan2(abs(x), z)) if abs(z) > 1e-12 or abs(x) > 1e-12 else 0.0
    if z < 0.0 and angle_deg >= (180.0 - back_threshold_deg):
        return 2
    return 0 if x < 0.0 else 1


def _camera_forward_world(ctx) -> tuple[float, float]:
    R = ctx.T_camera_world
    fx, fz = float(R[2][0]), float(R[2][2])
    norm = math.hypot(fx, fz)
    if norm < 1e-12:
        return (0.0, 1.0)
    return (fx / norm, fz / norm)


def _signed_yaw_delta_deg(forward_a: tuple[float, float], forward_b: tuple[float, float]) -> float:
    ax, az = forward_a
    bx, bz = forward_b
    dot = max(-1.0, min(1.0, ax * bx + az * bz))
    det = ax * bz - az * bx
    return math.degrees(math.atan2(det, dot))


def _classify_camera_pose_change(last_visible_time_sec: float, query_time_sec: float, candidate: KeyFrameCandidate, cfg: BenchmarkConfig) -> tuple[int, float]:
    ctx_last = load_frame_context(
        video_id=candidate.video_id,
        time_sec=last_visible_time_sec,
        annotations_root=cfg.annotations_root,
        fps=cfg.fps_for_frame_lookup,
        intermediate_root=cfg.intermediate_root,
    )
    ctx_query = load_frame_context(
        video_id=candidate.video_id,
        time_sec=query_time_sec,
        annotations_root=cfg.annotations_root,
        fps=cfg.fps_for_frame_lookup,
        intermediate_root=cfg.intermediate_root,
    )

    yaw_delta = _signed_yaw_delta_deg(_camera_forward_world(ctx_last), _camera_forward_world(ctx_query))
    abs_delta = abs(yaw_delta)
    if abs_delta <= cfg.camera_turn_small_deg:
        return 0, yaw_delta
    if abs_delta >= cfg.camera_turn_back_deg:
        return 3, yaw_delta
    return (1, yaw_delta) if yaw_delta > 0.0 else (2, yaw_delta)


def _load_config(path: Path) -> BenchmarkConfig:
    raw = _load_yaml(path)
    root = path.parent
    inputs = raw.get("inputs", {})
    return BenchmarkConfig(
        annotations_root=(root / raw["annotations_root"]).resolve(),
        video_ids=[str(v) for v in inputs.get("videos", [])],
        sampling_fps=float(raw.get("sampling_fps", 2.0)),
        fps_for_frame_lookup=float(raw.get("fps_for_frame_lookup", 30.0)),
        out_of_sight_horizon_sec=float(raw.get("out_of_sight_horizon_sec", 2.0)),
        max_questions_per_video=int(raw.get("max_questions_per_video", 20)),
        absolute_num_choices=int(raw.get("absolute", {}).get("num_choices", 5)),
        random_seed=int(raw.get("random_seed", 42)),
        back_threshold_deg=float(raw.get("camera_relative", {}).get("back_threshold_deg", 135.0)),
        camera_turn_small_deg=float(raw.get("camera_pose_change", {}).get("no_change_threshold_deg", 30.0)),
        camera_turn_back_deg=float(raw.get("camera_pose_change", {}).get("rotated_back_threshold_deg", 135.0)),
        intermediate_root=str(raw.get("intermediate_root", DEFAULT_INTERMEDIATE_ROOT)),
        output_json=(root / raw.get("output_json", "staged_oos_questions.json")).resolve(),
        pre_context_sec=float(raw.get("pre_context_sec", 2.0)),
    )


def generate_staged_benchmark(cfg: BenchmarkConfig) -> dict[str, dict[str, Any]]:
    if not cfg.video_ids:
        raise ValueError("No input videos were provided in the config.")

    rng = random.Random(cfg.random_seed)
    horizon_token = _format_horizon_token(cfg.out_of_sight_horizon_sec)
    fixture_vocab = build_fixture_vocabulary(cfg.annotations_root, video_ids=cfg.video_ids)
    keyframes_by_video = generate_key_frames_for_videos(
        video_ids=cfg.video_ids,
        annotations_root=cfg.annotations_root,
        horizon_sec=cfg.out_of_sight_horizon_sec,
        max_questions_per_video=cfg.max_questions_per_video,
        sampling_fps=cfg.sampling_fps,
        fps_for_frame_lookup=cfg.fps_for_frame_lookup,
        intermediate_root=cfg.intermediate_root,
        random_seed=cfg.random_seed,
        pre_context_sec=cfg.pre_context_sec,
    )

    results: dict[str, dict[str, Any]] = {}
    running_idx = 0

    for video_id in cfg.video_ids:
        for candidate in sorted(keyframes_by_video.get(video_id, []), key=lambda c: c.query_time_sec):
            states = _states_by_assoc_id(
                video_id=candidate.video_id,
                time_sec=candidate.query_time_sec,
                annotations_root=cfg.annotations_root,
                fps=cfg.fps_for_frame_lookup,
                intermediate_root=cfg.intermediate_root,
            )
            object_state = states.get(candidate.assoc_id)
            if object_state is None:
                continue

            query_time_in_clip_sec = candidate.query_time_sec - candidate.clip_start_time_sec
            time_tok = _time_token(query_time_in_clip_sec, input_key="video 1")
            common = {
                "inputs": {"video 1": {"id": candidate.video_id}},
                "video_id": candidate.video_id,
                "object_a_assoc_id": candidate.assoc_id,
                "object_a_name": candidate.object_name,
                "query_time_sec": candidate.query_time_sec,
                "query_time_in_clip_sec": query_time_in_clip_sec,
                "clip_start_time_sec": candidate.clip_start_time_sec,
                "clip_end_time_sec": candidate.clip_end_time_sec,
                "clip_duration_sec": candidate.clip_duration_sec,
                "horizon_sec": candidate.horizon_sec,
                "generation_info": asdict(candidate),
            }

            qid = f"oos_step1_visibility_{horizon_token}_{running_idx}"
            results[qid] = {
                **common,
                "question_class": "oos_step1_visibility",
                "question": f"At the current time {time_tok}, is the {candidate.object_name} visible in the camera view?",
                "choices": ["Yes", "No"],
                "correct_idx": 1 if not bool(object_state.status == "ok" and object_state.in_view) else 0,
                "answer_metadata": {
                    "status": object_state.status,
                    "in_view": object_state.in_view,
                    "projected_pixel": object_state.projected_pixel,
                },
            }
            running_idx += 1

            last_visible = _find_last_visible_info(candidate, cfg)
            if last_visible is not None:
                qid = f"oos_step2_last_visible_{horizon_token}_{running_idx}"
                results[qid] = {
                    **common,
                    "question_class": "oos_step2_last_visible",
                    "question": (
                        f"At the current time {time_tok}, the {candidate.object_name} is not visible. "
                        "When was it last visible, and where was it located in the image?"
                    ),
                    "choices": [],
                    "correct_idx": None,
                    "answer_metadata": {
                        "sampled_last_visible_time_sec": last_visible.sampled_time_sec,
                        "sampled_last_visible_time_in_clip_sec": last_visible.sampled_time_sec - candidate.clip_start_time_sec,
                        "projected_pixel": last_visible.projected_pixel,
                        "frame_index": last_visible.frame_index,
                        "note": "This answer uses the last sampled visible timestamp from the generated visibility track.",
                    },
                }
                running_idx += 1

            try:
                abs_answer = determine_absolute_answer(
                    video_id=candidate.video_id,
                    time_sec=candidate.query_time_sec,
                    object_a_assoc_id=candidate.assoc_id,
                    annotations_root=cfg.annotations_root,
                    num_choices=cfg.absolute_num_choices,
                    fixture_vocabulary=fixture_vocab,
                    rng=rng,
                )
                qid = f"oos_step3_fixture_{horizon_token}_{running_idx}"
                results[qid] = {
                    **common,
                    "question_class": "oos_step3_fixture",
                    "question": (
                        f"At the current time {time_tok}, the {candidate.object_name} is not visible. "
                        "Based on where it was last placed, what is the nearest fixture?"
                    ),
                    "choices": abs_answer.choices,
                    "correct_idx": abs_answer.correct_idx,
                    "answer_metadata": {"correct_fixture": abs_answer.correct_fixture},
                }
                running_idx += 1
            except Exception:
                pass

            if object_state.status == "ok" and object_state.camera_coordinates is not None:
                direction_idx = _classify_camera_relative_direction(
                    object_state.camera_coordinates,
                    back_threshold_deg=cfg.back_threshold_deg,
                )
                qid = f"oos_cam_relative_{horizon_token}_{running_idx}"
                results[qid] = {
                    **common,
                    "question_class": "oos_cam_relative",
                    "question": (
                        f"At the current time {time_tok}, the {candidate.object_name} is not visible. "
                        "Relative to the camera, which direction best describes the object's position?"
                    ),
                    "choices": LEFT_RIGHT_BACK_CHOICES,
                    "correct_idx": direction_idx,
                    "answer_metadata": {
                        "camera_coordinates": object_state.camera_coordinates,
                        "back_threshold_deg": cfg.back_threshold_deg,
                    },
                }
                running_idx += 1

            if last_visible is not None:
                try:
                    pose_idx, yaw_delta_deg = _classify_camera_pose_change(
                        last_visible_time_sec=last_visible.sampled_time_sec,
                        query_time_sec=candidate.query_time_sec,
                        candidate=candidate,
                        cfg=cfg,
                    )
                    qid = f"oos_step4_camera_pose_change_{horizon_token}_{running_idx}"
                    results[qid] = {
                        **common,
                        "question_class": "oos_step4_camera_pose_change",
                        "question": (
                            f"At the current time {time_tok}, compared with when the {candidate.object_name} was last visible, "
                            "how has the camera changed?"
                        ),
                        "choices": CAMERA_POSE_CHANGE_CHOICES,
                        "correct_idx": pose_idx,
                        "answer_metadata": {
                            "yaw_delta_deg": yaw_delta_deg,
                            "no_change_threshold_deg": cfg.camera_turn_small_deg,
                            "rotated_back_threshold_deg": cfg.camera_turn_back_deg,
                            "reference_time_sec": last_visible.sampled_time_sec,
                        },
                    }
                    running_idx += 1
                except Exception:
                    pass

    return results


def save_benchmark_json(items: dict[str, dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate staged OOS benchmark questions on top of the existing repo")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "staged_oos_benchmark_config.yaml",
        help="Path to the staged benchmark config YAML",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output JSON override",
    )
    parser.add_argument(
        "--pre_context_sec",
        type=float,
        default=None,
        help="Seconds before last visible time to include in clip"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config.resolve())

    output_json = args.output_json.resolve() if args.output_json is not None else cfg.output_json
    pre_context_sec = args.pre_context_sec if args.pre_context_sec is not None else cfg.pre_context_sec

    cfg = BenchmarkConfig(
        **{
            **asdict(cfg),
            "output_json": output_json,
            "pre_context_sec": pre_context_sec,
        }
    )

    benchmark = generate_staged_benchmark(cfg)
    save_benchmark_json(benchmark, output_json)
    print(f"Generated {len(benchmark)} staged OOS questions")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
