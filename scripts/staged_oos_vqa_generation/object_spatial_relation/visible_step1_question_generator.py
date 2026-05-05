from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
import re

sys.path.append(str(Path(__file__).resolve().parent))

import key_frame_generator as kfg
from candidate_detection_filter import CandidateDetectionFilter
from scripts.visibility_track.detection_refinement import VideoFrameReader
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
    """Build a step-1 visibility question 
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
            f"At time {time_tok}, can the {candidate.object_name} that was moved earlier be seen in the current frame?"
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

def _normalize_detic_class_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\d+$", "", name).strip()  # bowl2 -> bowl
    name = name.replace(",", " ")             # Detic uses comma as separator
    name = re.sub(r"\s+", " ", name).strip()
    return name

def _is_visible_track_state(object_state: Any) -> bool:
    if object_state is None:
        return False
    return bool(
        _state_attr(object_state, "is_visible")
        and _state_attr(object_state, "is_stably_visible")
        and _state_attr(object_state, "projected_pixel") is not None
    )



def _candidate_key(candidate: kfg.KeyFrameCandidate) -> tuple[str, str, float]:
    return (
        str(candidate.video_id),
        str(candidate.assoc_id),
        round(float(candidate.query_time_sec), 6),
    )


def _collect_track_visible_candidates(
    *,
    candidates: list[kfg.KeyFrameCandidate],
    caches: RuntimeCaches,
) -> tuple[list[kfg.KeyFrameCandidate], dict[tuple[str, str, float], Any]]:
    """Keep only candidates that the original visibility track says are visible.

    This is only the cheap first-stage filter. The detector is still the final
    authority before a Yes question is emitted.
    """
    visible_candidates: list[kfg.KeyFrameCandidate] = []
    object_state_by_candidate: dict[tuple[str, str, float], Any] = {}

    for candidate in candidates:
        states = caches.states_by_assoc_id(candidate.video_id, candidate.query_time_sec)
        object_state = states.get(candidate.assoc_id)

        if not _is_visible_track_state(object_state):
            continue

        key = _candidate_key(candidate)
        object_state_by_candidate[key] = object_state
        visible_candidates.append(candidate)

    return visible_candidates, object_state_by_candidate

def _limit_candidates_per_object_name(
    candidates: list[kfg.KeyFrameCandidate],
    max_per_name: int = 2,
) -> list[kfg.KeyFrameCandidate]:
    """Limit how many detector checks/questions come from the same object category."""
    counts: dict[str, int] = {}
    out: list[kfg.KeyFrameCandidate] = []

    for candidate in candidates:
        name = _normalize_detic_class_name(candidate.object_name)
        if not name:
            name = str(candidate.object_name).strip().lower()

        if counts.get(name, 0) >= max_per_name:
            continue

        counts[name] = counts.get(name, 0) + 1
        out.append(candidate)

    return out

def generate_visible_benchmark(
    cfg: BenchmarkConfig,
    *,
    min_visible_context_sec: float = 0.5,
) -> dict[str, dict[str, Any]]:
    if not cfg.video_ids:
        raise ValueError("No input videos were provided in the config.")

    rng = random.Random(cfg.random_seed)
    horizon_token = _format_horizon_token(cfg.out_of_sight_horizon_sec)

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

    caches = RuntimeCaches(cfg, visibility_store)
    results: dict[str, dict[str, Any]] = {}
    running_idx = 0

    # Start with the configured pool, then expand it if detector rejection makes
    # the accepted count too small. You can also set this optional field under
    # object_detection_filter in YAML: max_candidate_pool_multiplier: 50
    start_multiplier = max(1, int(cfg.candidate_pool_multiplier))
    max_multiplier = max(
        start_multiplier,
        int(getattr(cfg.detection, "max_candidate_pool_multiplier", start_multiplier * 8)),
    )

    for video_id in cfg.video_ids:
        emitted_for_video = 0
        rejected: list[dict[str, Any]] = []
        checked_keys: set[tuple[str, str, float]] = set()

        participant_id = video_id.split("-")[0]
        video_path = Path(cfg.video_root) / participant_id / f"{video_id}.mp4"

        # Build a broad vocabulary once from all track-visible candidates at the
        # largest pool size. This prevents the detector vocabulary from changing
        # between expansion rounds.
        vocab_candidates = kfg.generate_visible_key_frames_for_video(
            video_id=video_id,
            annotations_root=cfg.annotations_root,
            max_questions_per_video=cfg.max_questions_per_video,
            candidate_pool_multiplier=max_multiplier,
            sampling_fps=cfg.sampling_fps,
            fps_for_frame_lookup=cfg.fps_for_frame_lookup,
            intermediate_root=cfg.intermediate_root,
            random_seed=cfg.random_seed,
            min_visible_context_sec=min_visible_context_sec,
            precomputed_tracks=precomputed_tracks_by_video.get(video_id),
        )

        vocab_visible_candidates, _ = _collect_track_visible_candidates(
            candidates=sorted(vocab_candidates, key=lambda c: c.query_time_sec),
            caches=caches,
        )
        custom_vocabulary_list = sorted({
            _normalize_detic_class_name(c.object_name)
            for c in vocab_visible_candidates
            if _normalize_detic_class_name(c.object_name)
        })

        custom_vocabulary = ",".join(custom_vocabulary_list)

        detection_filter = CandidateDetectionFilter(
            video_id=video_id,
            video_path=video_path,
            annotations_root=cfg.annotations_root,
            intermediate_root=cfg.intermediate_root,
            video_fps=cfg.fps_for_frame_lookup,
            det_cfg=cfg.detection,
            accept_labels=set(cfg.detection.accept_labels),
            custom_vocabulary=custom_vocabulary,
        )

        multiplier = start_multiplier
        with VideoFrameReader(video_path, cfg.fps_for_frame_lookup) as frame_reader:
            while emitted_for_video < cfg.max_questions_per_video and multiplier <= max_multiplier:
                candidates = kfg.generate_visible_key_frames_for_video(
                    video_id=video_id,
                    annotations_root=cfg.annotations_root,
                    max_questions_per_video=cfg.max_questions_per_video,
                    candidate_pool_multiplier=multiplier,
                    sampling_fps=cfg.sampling_fps,
                    fps_for_frame_lookup=cfg.fps_for_frame_lookup,
                    intermediate_root=cfg.intermediate_root,
                    random_seed=cfg.random_seed,
                    min_visible_context_sec=min_visible_context_sec,
                    precomputed_tracks=precomputed_tracks_by_video.get(video_id),
                )
                visible_candidates, object_state_by_candidate = _collect_track_visible_candidates(
                    candidates=sorted(candidates, key=lambda c: c.query_time_sec),
                    caches=caches,
                )

                made_progress_this_round = False
                for candidate in visible_candidates:
                    if emitted_for_video >= cfg.max_questions_per_video:
                        break

                    candidate_key = _candidate_key(candidate)
                    if candidate_key in checked_keys:
                        continue
                    checked_keys.add(candidate_key)

                    decision = detection_filter.check_one(candidate, frame_reader)

                    if not decision.accepted:
                        rejected.append({
                            "video_id": candidate.video_id,
                            "assoc_id": candidate.assoc_id,
                            "object_name": candidate.object_name,
                            "query_time_sec": candidate.query_time_sec,
                            "detection": decision.metadata,
                        })
                        continue

                    object_state = object_state_by_candidate[candidate_key]

                    common = _build_common_fields(candidate)
                    common["generation_info"]["visible_only_generation"] = True
                    common["generation_info"]["selector"] = "generate_visible_key_frames_for_video"
                    common["generation_info"]["detector_verified"] = True
                    common["generation_info"]["detector_metadata"] = decision.metadata

                    time_tok = _time_token(candidate.query_time_sec, input_key="video 1")
                    step1 = _build_step1_visible_yes(candidate, object_state, time_tok, rng=rng)

                    trajectory_id = f"visible_staged_{horizon_token}_{running_idx}"
                    result_key, payload = _finalize_visible_trajectory(trajectory_id, common, step1)

                    results[result_key] = payload
                    print(f"[ACCEPTED] {result_key} from candidate {candidate_key} (pool x{multiplier})")
                    running_idx += 1
                    emitted_for_video += 1
                    made_progress_this_round = True

                if emitted_for_video >= cfg.max_questions_per_video:
                    break

                next_multiplier = multiplier * 2
                if next_multiplier > max_multiplier:
                    break
                if not made_progress_this_round and len(checked_keys) >= len(visible_candidates):
                    # The current pool is exhausted; expand the pool and try new
                    # candidates rather than stopping early.
                    pass
                multiplier = next_multiplier
                print(
                    f"[INFO] video {video_id}: detector accepted {emitted_for_video}/"
                    f"{cfg.max_questions_per_video}; expanding candidate_pool_multiplier to {multiplier}"
                )

        rejected_path = (
            cfg.output_json.parent
            / "detector_debug"
            / f"{video_id}_visible_step1_rejected_by_detector.json"
        )

        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        with rejected_path.open("w", encoding="utf-8") as f:
            json.dump(rejected, f, indent=2, ensure_ascii=False)

        if emitted_for_video < cfg.max_questions_per_video:
            print(
                f"[WARN] video {video_id}: emitted {emitted_for_video}/{cfg.max_questions_per_video} "
                f"visible step-1 trajectories after detector filtering. Tried "
                f"{len(checked_keys)} candidates up to candidate_pool_multiplier={max_multiplier}. "
                "There may not be enough detector-verified visible samples."
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


    cfg = replace(
        cfg,
        output_json=output_json,
        visibility_tracks_json_by_video=visibility_tracks_json_by_video,
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

