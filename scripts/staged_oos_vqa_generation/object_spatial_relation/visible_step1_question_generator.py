from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

import key_frame_generator as kfg
from staged_oos_question_generator import (
    BenchmarkConfig,
    RuntimeCaches,
    _build_common_fields,
    _format_horizon_token,
    _load_config,
    _load_visibility_store,
    _state_attr,
    _time_token,
)

def _finalize_choices(
    choices: list[str],
    correct_answer: str,
    rng: random.Random,
    *,
    shuffle: bool = True,
) -> tuple[list[str], int]:
    final_choices = list(choices)
    if shuffle:
        rng.shuffle(final_choices)
    correct_idx = final_choices.index(correct_answer)
    return final_choices, correct_idx

def _build_step1_visible_yes(
    candidate: kfg.KeyFrameCandidate,
    object_state: Any,
    time_tok: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Build a step-1 visibility question whose correct answer is always index 0.

    We keep choices unshuffled so this script can generate binary-visible examples
    with correct answer being "Yes", while the original OOS generator remains unchanged.
    """
    correct_answer = "Yes"
    choices, correct_idx = _finalize_choices(
        choices=["Yes", "No"],
        correct_answer=correct_answer,
        rng=rng,
        shuffle=True,
    )
    return {
        "step": 1,
        "question_class": "oos_step1_visibility",
        "question": (
            f"At the current time {time_tok}, is the "
            f"{candidate.object_name} visible in the current frame?"
        ),
        "choices": choices,
        "correct_idx": correct_idx,
        "answer_metadata": {
            "status": _state_attr(object_state, "status"),
            "is_visible": _state_attr(object_state, "is_visible"),
            "is_stably_visible": _state_attr(object_state, "is_stably_visible"),
            "projected_pixel": _state_attr(object_state, "projected_pixel"),
            "camera_coordinates": _state_attr(object_state, "camera_coordinates"),
            "frame_index": _state_attr(object_state, "frame_number"),        
            },
    }


def _finalize_visible_trajectory(
    trajectory_id: str,
    common: dict[str, Any],
    step1: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    return trajectory_id, {
        **common,
        "question_class": "visible_staged_trajectory",
        "trajectory_id": trajectory_id,
        "num_incremental_steps": 1,
        "num_branch_steps": 0,
        "terminated_at_step": 1,
        "stop_reason": "object_visible_at_query_time",
        "incremental_steps": [step1],
        "branch_groups": {"post_step3": []},
    }


def _is_visible_track_state(object_state: Any) -> bool:
    if object_state is None:
        return False
    return bool(
        _state_attr(object_state, "is_visible")
        and _state_attr(object_state, "is_stably_visible")
        and _state_attr(object_state, "projected_pixel") is not None
    )


def generate_visible_benchmark(
    cfg: BenchmarkConfig,
    *,
    min_visible_context_sec: float = 0.5,
) -> dict[str, dict[str, Any]]:
    if not cfg.video_ids:
        raise ValueError("No input videos were provided in the config.")

    # The RNG is kept for deterministic trajectory ids/order if you later add
    # sampling, but this script does not shuffle step-1 choices.
    rng = random.Random(cfg.random_seed)
    horizon_token = _format_horizon_token(cfg.out_of_sight_horizon_sec)
    candidate_pool_per_video = max(cfg.max_questions_per_video * 5, cfg.max_questions_per_video)

    visibility_store = _load_visibility_store(cfg)
    precomputed_tracks_by_video = {
        video_id: visibility_store.get_object_tracks(video_id)
        for video_id in cfg.video_ids
        if visibility_store.has_video(video_id)
    }
    if not precomputed_tracks_by_video:
        raise ValueError(
            "Visible generation requires precomputed visibility tracks. "
            "Set visibility_tracks_json in the config or pass --visibility_tracks_json."
        )

    keyframes_by_video = kfg.generate_visible_key_frames_for_videos(
        video_ids=cfg.video_ids,
        annotations_root=cfg.annotations_root,
        max_questions_per_video=candidate_pool_per_video,
        sampling_fps=cfg.sampling_fps,
        fps_for_frame_lookup=cfg.fps_for_frame_lookup,
        intermediate_root=cfg.intermediate_root,
        random_seed=cfg.random_seed,
        min_visible_context_sec=min_visible_context_sec,
        precomputed_tracks_by_video=precomputed_tracks_by_video,
    )

    caches = RuntimeCaches(cfg, visibility_store)
    results: dict[str, dict[str, Any]] = {}
    running_idx = 0

    for video_id in cfg.video_ids:
        candidates = sorted(keyframes_by_video.get(video_id, []), key=lambda c: c.query_time_sec)
        emitted_for_video = 0

        for candidate in candidates:
            if emitted_for_video >= cfg.max_questions_per_video:
                break

            states = caches.states_by_assoc_id(candidate.video_id, candidate.query_time_sec)
            object_state = states.get(candidate.assoc_id)
            if not _is_visible_track_state(object_state):
                continue

            common = _build_common_fields(candidate)
            common["generation_info"]["visible_only_generation"] = True
            common["generation_info"]["selector"] = "generate_visible_key_frames_for_videos"

            time_tok = _time_token(candidate.query_time_sec, input_key="video 1")
            step1 = _build_step1_visible_yes(candidate, object_state, time_tok, rng=rng)

            trajectory_id = f"visible_staged_{horizon_token}_{running_idx}"
            key, payload = _finalize_visible_trajectory(trajectory_id, common, step1)
            results[key] = payload
            running_idx += 1
            emitted_for_video += 1

        if emitted_for_video < cfg.max_questions_per_video:
            print(
                f"[WARN] video {video_id}: emitted {emitted_for_video}/{cfg.max_questions_per_video} "
                "visible step-1 trajectories. Candidate pool may be too small."
            )

    return results


def save_benchmark_json(items: dict[str, dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fast visible-only step-1 visibility trajectories"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "staged_oos_question_generator_config.yaml",
        help="Path to the staged benchmark config YAML",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output JSON override",
    )
    parser.add_argument(
        "--visibility_tracks_json",
        type=Path,
        default=None,
        help="Optional precomputed visibility track JSON override for single-video configs",
    )
    parser.add_argument(
        "--min_visible_context_sec",
        type=float,
        default=0.5,
        help="Minimum seconds after a visible span starts before sampling the query frame",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config.resolve())

    if args.output_json is not None:
        output_json = args.output_json.resolve()
    elif cfg.output_json is not None:
        output_json = cfg.output_json.with_name(cfg.output_json.stem + "_visible_step1.json")
    else:
        output_json = Path("visible_step1_questions.json").resolve()

    visibility_tracks_json_by_video = dict(cfg.visibility_tracks_json_by_video or {})
    if args.visibility_tracks_json is not None:
        if len(cfg.video_ids) != 1:
            raise ValueError(
                "--visibility_tracks_json can only be used when exactly one input video is provided."
            )
        visibility_tracks_json_by_video[cfg.video_ids[0]] = args.visibility_tracks_json.resolve()


    cfg = BenchmarkConfig(
        **{
            **asdict(cfg),
            "output_json": output_json,
            "visibility_tracks_json_by_video": visibility_tracks_json_by_video,
        }
    )

    benchmark = generate_visible_benchmark(
        cfg,
        min_visible_context_sec=args.min_visible_context_sec,
    )
    save_benchmark_json(benchmark, output_json)

    print(f"Generated {len(benchmark)} visible step-1 trajectories")
    if visibility_tracks_json_by_video:
        for video_id, path in visibility_tracks_json_by_video.items():
            print(f"[{video_id}] Using visibility tracks: {path}")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()


# python scripts/staged_oos_vqa_generation/object_spatial_relation/visible_step1_question_generator.py \
#     --config scripts/staged_oos_vqa_generation/object_spatial_relation/staged_oos_question_generator_config.yaml \

