#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".webm"]

# Use string-normalized step IDs because the new data mixes ints and strings:
# 1, 2, "3", 4, "5a", "5b", "5c"
IGNORED_STEP_IDS = {"0"}   # change to set() if you want to keep fixture questions


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_step_id(step_value: Any) -> str:
    return str(step_value).strip()

def normalize_question_text(question: Any) -> str:
    if question is None:
        return ""
    if isinstance(question, str):
        return question.strip()
    if isinstance(question, list):
        return " ".join(str(q).strip() for q in question if q is not None and str(q).strip())
    return str(question).strip()

def normalize_answer_metadata(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}

    out = dict(meta)

    # Arrow-safe normalization for mixed nested types
    if "reference_source" in out:
        value = out["reference_source"]
        if value is not None and not isinstance(value, str):
            out["reference_source"] = json.dumps(value, ensure_ascii=False, sort_keys=True)

    return out


def iter_all_steps(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten the new trajectory structure into one ordered list.

    Order:
      incremental_steps first
      then branch groups in deterministic order
    """
    steps: List[Dict[str, Any]] = []

    for step in traj.get("incremental_steps", []) or []:
        steps.append(step)

    branch_groups = traj.get("branch_groups", {}) or {}

    # Keep deterministic ordering
    for group_name in sorted(branch_groups.keys()):
        group_steps = branch_groups.get(group_name, []) or []
        group_steps = sorted(group_steps, key=lambda s: normalize_step_id(s.get("step")))
        steps.extend(group_steps)

    return steps


def safe_correct_answer(step: Dict[str, Any]) -> Optional[str]:
    choices = step.get("choices", []) or []
    correct_idx = step.get("correct_idx", None)
    if correct_idx is None:
        return None
    if not isinstance(correct_idx, int):
        return None
    if correct_idx < 0 or correct_idx >= len(choices):
        return None
    return choices[correct_idx]


def should_exclude_trajectory(traj: Dict[str, Any]) -> bool:
    for step in iter_all_steps(traj):
        step_id = normalize_step_id(step.get("step"))
        if step_id in IGNORED_STEP_IDS:
            continue

        if bool(step.get("skipped", False)):
            return True

        answer_metadata = step.get("answer_metadata", {}) or {}
        status = str(answer_metadata.get("status", "")).strip().lower()
        if status in {"error", "failed", "invalid", "exception"}:
            return True

    return False


def find_video_path(video_root: Optional[Path], video_id: str) -> Optional[str]:
    """
    Always construct a path without checking existence.
    """
    if video_root is None or video_id is None:
        return None

    # Choose a default extension (mp4 is safest)
    return str((video_root / f"{video_id}.mp4").resolve())

def build_common_fields(
    trajectory_id: str,
    traj: Dict[str, Any],
    video_path: Optional[str],
) -> Dict[str, Any]:
    return {
        "trajectory_id": trajectory_id,
        "question_class": traj.get("question_class"),
        "video_id": traj.get("video_id"),
        "video_path": video_path,
        "query_time_sec": traj.get("query_time_sec"),
        "query_time_in_clip_sec": traj.get("query_time_in_clip_sec"),
        "clip_start_time_sec": traj.get("clip_start_time_sec"),
        "clip_end_time_sec": traj.get("clip_end_time_sec"),
        "clip_duration_sec": traj.get("clip_duration_sec"),
        "horizon_sec": traj.get("horizon_sec"),
        "object_a_assoc_id": traj.get("object_a_assoc_id"),
        "object_a_name": traj.get("object_a_name"),
        "num_incremental_steps": traj.get("num_incremental_steps"),
        "num_branch_steps": traj.get("num_branch_steps"),
        "terminated_at_step": traj.get("terminated_at_step"),
        "stop_reason": traj.get("stop_reason"),
        "generation_info": traj.get("generation_info"),
    }


def get_step_target_text(step: Dict[str, Any]) -> Optional[str]:
    mcq_answer = safe_correct_answer(step)
    if mcq_answer is not None:
        return mcq_answer

    explicit = step.get("target_text")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    meta = step.get("answer_metadata", {}) or {}
    qclass = step.get("question_class") or step.get("step_question_class") or ""

    if qclass == "oos_step2_last_visible":
        t_token = meta.get("sampled_last_visible_time_token")
        pt = meta.get("normalized_projected_pixel")
        if isinstance(t_token, str) and isinstance(pt, (list, tuple)) and len(pt) == 2:
            return f"{t_token}; Point=({pt[0]:.4f}, {pt[1]:.4f})"

    if qclass == "oos_step3_last_placement":
        t_token = meta.get("last_placement_time_token")
        pt = meta.get("normalized_projected_pixel")
        fixture = meta.get("fixture")
        if isinstance(t_token, str) and isinstance(pt, (list, tuple)) and len(pt) == 2:
            return f"{t_token}; Point=({pt[0]:.4f}, {pt[1]:.4f})"

    if qclass == "oos_step4_fixture":
        fixture = meta.get("correct_fixture")
        if fixture:
            return str(fixture)

    if qclass == "oos_branch_object_camera_relative_position":
        label = meta.get("correct_label")
        if label:
            return str(label)

    if qclass == "oos_branch_object_object_distance":
        bucket = meta.get("distance_bucket")
        if bucket:
            return str(bucket)

    # For relation branch, the correct_idx + choices path above usually already works.
    # acceptable_idxs can still be preserved in metadata for evaluation.
    return None

def compute_dependencies_for_trajectory(traj: Dict[str, Any]) -> Dict[str, List[str]]:
    incremental_steps = traj.get("incremental_steps", []) or []
    incremental_ids = [normalize_step_id(s.get("step")) for s in incremental_steps]

    dep_map: Dict[str, List[str]] = {}

    # Incremental chain: each step depends on earlier incremental steps
    for i, sid in enumerate(incremental_ids):
        dep_map[sid] = incremental_ids[:i]

    # Branch steps: each branch step depends on all incremental steps
    branch_groups = traj.get("branch_groups", {}) or {}
    for _, group_steps in sorted(branch_groups.items()):
        for step in sorted(group_steps or [], key=lambda s: normalize_step_id(s.get("step"))):
            sid = normalize_step_id(step.get("step"))
            dep_map[sid] = incremental_ids[:]

    return dep_map

def trajectory_to_single_turn_examples(
    trajectory_id: str,
    traj: Dict[str, Any],
    video_path: Optional[str],
) -> List[Dict[str, Any]]:
    common = build_common_fields(trajectory_id, traj, video_path)
    examples: List[Dict[str, Any]] = []
    all_steps = iter_all_steps(traj)
    dep_map = compute_dependencies_for_trajectory(traj)
    for step in iter_all_steps(traj):


        step_id = normalize_step_id(step.get("step"))
        if step_id in IGNORED_STEP_IDS:
            continue

        ex = {
            **common,
            "doc_id": f"{trajectory_id}__step_{step_id}",
            "mode": "single_turn",
            "step": step_id,
            "branch_group": step.get("branch_group"),
            "depends_on_steps": dep_map[step_id],
            "step_question_class": step.get("question_class"),
            "question": normalize_question_text(step.get("question")),
            "choices": step.get("choices", []),
            "correct_idx": step.get("correct_idx"),
            "acceptable_idxs": step.get("acceptable_idxs"),
            "target_text": get_step_target_text(step),
            "answer_metadata": normalize_answer_metadata(step.get("answer_metadata", {})),
            "skipped": bool(step.get("skipped", False)),
        }
        examples.append(ex)

    return examples


def make_multiturn_messages_from_gold(
    steps: List[Dict[str, Any]],
    include_system_prompt: bool = True,
    system_prompt: str = "Answer using only the video evidence available up to the queried time.",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    if include_system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })

    for step in steps:
        step_id = normalize_step_id(step.get("step"))
        qclass = step.get("question_class", "")

        if step_id in IGNORED_STEP_IDS:
            continue

        # Skip parallel branch questions from gold history
        if qclass in {
            "oos_branch_object_camera_relative_position",
            "oos_branch_object_object_relation",
            "oos_branch_object_object_distance",
        }:
            continue

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": normalize_question_text(step.get("question"))}],
        })

        gold_answer = get_step_target_text(step)
        if gold_answer is not None:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": gold_answer}],
            })

    return messages


def trajectory_to_multiturn_example(
    trajectory_id: str,
    traj: Dict[str, Any],
    video_path: Optional[str],
    include_gold_history: bool,
) -> Dict[str, Any]:
    common = build_common_fields(trajectory_id, traj, video_path)
    steps = iter_all_steps(traj)

    dep_map = compute_dependencies_for_trajectory(traj)

    step_records = []
    for step in steps:
        step_id = normalize_step_id(step.get("step"))

        if step_id in IGNORED_STEP_IDS:
            continue

        step_records.append({
            "step": step_id,
            "branch_group": step.get("branch_group"),
            "depends_on_steps": dep_map[step_id],
            "step_question_class": step.get("question_class"),
            "question": normalize_question_text(step.get("question")),
            "choices": step.get("choices", []),
            "correct_idx": step.get("correct_idx"),
            "acceptable_idxs": step.get("acceptable_idxs"),
            "target_text": get_step_target_text(step),
            "answer_metadata": normalize_answer_metadata(step.get("answer_metadata", {})),
            "skipped": bool(step.get("skipped", False)),
        })

    ex = {
        **common,
        "doc_id": trajectory_id,
        "mode": "multi_turn",
        "steps": step_records,
        "include_gold_history": include_gold_history,
    }

    if include_gold_history:
        ex["gold_history_messages"] = make_multiturn_messages_from_gold(steps)

    return ex


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_stats(
    single_turn_records: List[Dict[str, Any]],
    multi_turn_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    per_step_class: Dict[str, int] = {}
    skipped_count = 0
    mcq_count = 0
    open_count = 0

    for rec in single_turn_records:
        key = rec.get("step_question_class") or "unknown"
        per_step_class[key] = per_step_class.get(key, 0) + 1
        skipped_count += int(bool(rec.get("skipped", False)))
        if rec.get("choices"):
            mcq_count += 1
        else:
            open_count += 1

    return {
        "num_single_turn_examples": len(single_turn_records),
        "num_multi_turn_examples": len(multi_turn_records),
        "num_mcq_examples": mcq_count,
        "num_open_ended_examples": open_count,
        "num_skipped_examples": skipped_count,
        "per_step_question_class": per_step_class,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess staged OOS QA JSON for lmms-eval.")
    parser.add_argument(
    "--input",
    type=Path,
    nargs="+",
    required=True,
    help="Path(s) to one or more VQA JSON files",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write processed files")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Root directory containing the original videos",
    )
    parser.add_argument(
        "--include-gold-history",
        action="store_true",
        help="Include gold assistant replies in multi_turn.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    for input_path in args.input:
        raw = load_json(input_path)

    single_turn_records: List[Dict[str, Any]] = []
    multi_turn_records: List[Dict[str, Any]] = []

    for input_path in args.input:
        raw = load_json(input_path)

        for trajectory_id, traj in raw.items():
            if should_exclude_trajectory(traj):
                continue

            video_id = traj.get("video_id")
            video_path = find_video_path(args.video_root, video_id) if video_id else None

            if video_id:
                merged_trajectory_id = f"{video_id}__{trajectory_id}"
            else:
                merged_trajectory_id = trajectory_id

            single_turn_records.extend(
                trajectory_to_single_turn_examples(
                    trajectory_id=merged_trajectory_id,
                    traj=traj,
                    video_path=video_path,
                )
            )

            multi_turn_records.append(
                trajectory_to_multiturn_example(
                    trajectory_id=merged_trajectory_id,
                    traj=traj,
                    video_path=video_path,
                    include_gold_history=args.include_gold_history,
                )
            )

    single_path = args.output_dir / "single_turn.jsonl"
    multi_path = args.output_dir / "multi_turn.jsonl"
    stats_path = args.output_dir / "stats.json"

    write_jsonl(single_path, single_turn_records)
    write_jsonl(multi_path, multi_turn_records)

    stats = build_stats(single_turn_records, multi_turn_records)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote: {single_path}")
    print(f"Wrote: {multi_path}")
    print(f"Wrote: {stats_path}")


if __name__ == "__main__":
    main()

# python scripts/preprocessing/preprocess_oos_for_lmms_eval_v2.py \
#     --input \
#         scripts/staged_oos_vqa_generation/object_spatial_relation/outputs/generated_vqa/P01-20240202-110250_vqa_5_anchor_fixed_normalized.json \
#         scripts/staged_oos_vqa_generation/object_spatial_relation/outputs/generated_vqa/P01-20240202-110250_vqa_30_anchor_fixed_normalized.json \
#         scripts/staged_oos_vqa_generation/object_spatial_relation/outputs/generated_vqa/P01-20240203-132119_vqa_5_anchor_fixed_normalized.json \
#         scripts/staged_oos_vqa_generation/object_spatial_relation/outputs/generated_vqa/P01-20240203-132119_vqa_30_anchor_fixed_normalized.json \
#     --output-dir outputs/jsonl_v2 \
#     --video-root /work/courses/3dv/team1/data/HD-EPIC/Videos/P01_preprocessed_with_watermark \
#     --include-gold-history