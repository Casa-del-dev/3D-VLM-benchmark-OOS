#!/usr/bin/env python3
"""
Preprocess staged out-of-sight QA JSON into lmms-eval-friendly JSONL files.

Input format:
    The current trajectory JSON where each top-level key is a trajectory_id and
    each value contains:
      - video_id
      - query_time_sec
      - steps: list[step dict]

Outputs:
    1) single_turn.jsonl
       One example per step. Good for clean per-question evaluation.
    2) multi_turn.jsonl
       One example per trajectory. Contains all steps in order and can be used
       with a custom doc_to_messages for multi-turn evaluation.

Design goals:
    - Keep ONE source video path per example
    - Store query_time_sec so the model/task code can enforce causal loading
      (0 -> query_time_sec) at runtime without duplicating videos
    - Preserve metadata needed for analysis

Typical usage:
    python preprocess_oos_for_lmms_eval.py ^
        --input staged_oos_trajectories.json ^
        --video-root D:\videos ^
        --output-dir processed_oos

Windows notes:
    - This script prints and stores Windows paths correctly.
    - It does not cut or rewrite video files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".webm"]
IGNORED_STEP_NUMBERS = {4}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    """
    Exclude the entire trajectory if any non-ignored step is explicitly skipped
    or has an error-like answer status.
    """
    for step in traj.get("steps", []):
        if step.get("step") in IGNORED_STEP_NUMBERS:
            continue

        if bool(step.get("skipped", False)):
            return True

        answer_metadata = step.get("answer_metadata", {}) or {}
        status = str(answer_metadata.get("status", "")).strip().lower()

        if status in {"error", "failed", "invalid", "exception"}:
            return True

    return False


def find_video_path(video_root: Optional[Path], video_id: str) -> Optional[str]:
    if video_root is None:
        return None

    candidates = []
    for ext in VIDEO_EXTENSIONS:
        candidates.append(video_root / f"{video_id}{ext}")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # Recursive fallback
    for ext in VIDEO_EXTENSIONS:
        matches = list(video_root.rglob(f"{video_id}{ext}"))
        if matches:
            return str(matches[0].resolve())

    return None


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
        "num_steps": traj.get("num_steps"),
        "terminated_at_step": traj.get("terminated_at_step"),
        "stop_reason": traj.get("stop_reason"),
        "generation_info": traj.get("generation_info"),
    }


def trajectory_to_single_turn_examples(
    trajectory_id: str,
    traj: Dict[str, Any],
    video_path: Optional[str],
) -> List[Dict[str, Any]]:
    common = build_common_fields(trajectory_id, traj, video_path)
    examples: List[Dict[str, Any]] = []

    for step in traj.get("steps", []):
        if step.get("step") in IGNORED_STEP_NUMBERS:
            continue

        target_text = get_step_target_text(step)

        # Keep a plain text target when possible.
        # For open-ended steps the official free-form answer is not present as a
        # single string, so target_text remains None and answer_metadata is kept.
        ex = {
            **common,
            "doc_id": f"{trajectory_id}__step_{step.get('step')}",
            "mode": "single_turn",
            "step": step.get("step"),
            "step_question_class": step.get("question_class"),
            "question": step.get("question"),
            "choices": step.get("choices", []),
            "correct_idx": step.get("correct_idx"),
            "target_text": target_text,
            "answer_metadata": step.get("answer_metadata", {}),
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
        if step.get("step") in IGNORED_STEP_NUMBERS:
            continue

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": step.get("question", "").strip()}],
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
    steps = traj.get("steps", [])

    step_records = []
    for step in steps:
        if step.get("step") in IGNORED_STEP_NUMBERS:
            continue

        step_records.append({
            "step": step.get("step"),
            "step_question_class": step.get("question_class"),
            "question": step.get("question"),
            "choices": step.get("choices", []),
            "correct_idx": step.get("correct_idx"),
            "target_text": get_step_target_text(step),
            "answer_metadata": step.get("answer_metadata", {}),
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


def build_stats(single_turn_records: List[Dict[str, Any]], multi_turn_records: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    if qclass == "oos_step3_fixture":
        fixture = meta.get("correct_fixture")
        if fixture:
            return str(fixture)

    if qclass == "oos_step5_camera_quadrant":
        label = meta.get("label")
        if label:
            return str(label)

    return None


def write_helper_readme(path: Path) -> None:
    text = """# LMMS-eval preprocessing output

Files:
- single_turn.jsonl
  One row per question step.
- multi_turn.jsonl
  One row per trajectory with ordered steps and optional gold-history messages.
- stats.json
  Summary counts.

Recommended lmms-eval usage

Current preprocessing behavior:
- step 4 is ignored by default
- an entire trajectory is excluded if any remaining step is skipped or has an error-like status

1) Single-turn evaluation
Use `single_turn.jsonl` and define:
- `doc_to_visual(doc)` -> return the original `video_path`
- `doc_to_text(doc)` -> return `question`
- task/model code should enforce causal loading using `query_time_sec`

2) Multi-turn evaluation
Use `multi_turn.jsonl` and define:
- `doc_to_messages(doc)` -> attach `video_path` once in the first user turn
- later turns should be text-only follow-up questions
- optionally use `gold_history_messages` for teacher-forced history

Important
Do NOT pass the full raw video naively at inference time if you want causal evaluation.
Use `video_path` + `query_time_sec` and enforce loading frames only from 0 to query_time_sec.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess staged OOS QA JSON for lmms-eval.")
    parser.add_argument("--input", type=Path, required=True, help="Path to staged_oos_trajectories.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write processed files")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Root directory containing the original videos. If provided, the script will try to resolve video_id to a file path.",
    )
    parser.add_argument(
        "--include-gold-history",
        action="store_true",
        help="For multi_turn.jsonl, include gold assistant replies for MCQ steps as chat history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    raw = load_json(args.input)

    single_turn_records: List[Dict[str, Any]] = []
    multi_turn_records: List[Dict[str, Any]] = []
    missing_videos: List[str] = []

    excluded_trajectory_ids: List[str] = []

    for trajectory_id, traj in raw.items():
        if should_exclude_trajectory(traj):
            excluded_trajectory_ids.append(trajectory_id)
            continue

        video_id = traj.get("video_id")
        video_path = find_video_path(args.video_root, video_id) if video_id else None
        if args.video_root is not None and video_path is None and video_id is not None:
            missing_videos.append(video_id)

        single_turn_records.extend(
            trajectory_to_single_turn_examples(
                trajectory_id=trajectory_id,
                traj=traj,
                video_path=video_path,
            )
        )

        multi_turn_records.append(
            trajectory_to_multiturn_example(
                trajectory_id=trajectory_id,
                traj=traj,
                video_path=video_path,
                include_gold_history=args.include_gold_history,
            )
        )

    single_path = args.output_dir / "single_turn.jsonl"
    multi_path = args.output_dir / "multi_turn.jsonl"
    stats_path = args.output_dir / "stats.json"
    readme_path = args.output_dir / "README_lmms_eval.txt"

    write_jsonl(single_path, single_turn_records)
    write_jsonl(multi_path, multi_turn_records)

    stats = build_stats(single_turn_records, multi_turn_records)
    stats["missing_video_ids"] = sorted(set(missing_videos))
    stats["num_excluded_trajectories"] = len(excluded_trajectory_ids)
    stats["excluded_trajectory_ids"] = excluded_trajectory_ids
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_helper_readme(readme_path)

    print(f"Wrote: {single_path}")
    print(f"Wrote: {multi_path}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {readme_path}")

    if missing_videos:
        print(f"Warning: could not resolve {len(set(missing_videos))} video IDs under --video-root")


if __name__ == "__main__":
    main()
