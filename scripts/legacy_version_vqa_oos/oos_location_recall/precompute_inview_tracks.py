from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from in_view_determination import DEFAULT_INTERMEDIATE_ROOT
from in_view_track_generator_copy import (
    apply_external_visibility_spans,
    generate_in_view_tracks,
    load_visibility_span_rows,
    tracks_to_dict,
)

FORMAT_VERSION = 5


@dataclass(frozen=True)
class PrecomputeConfig:
    annotations_root: Path
    video_ids: list[str]
    sampling_fps: float = 1.0
    fps_for_frame_lookup: float = 30.0
    intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT
    output_json_by_video: dict[str, Path] | None = None
    visibility_status_jsonl_by_video: dict[str, Path] | None = None
    start_time_sec: float | None = None
    end_time_sec: float | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load the benchmark config. Install it with: pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_path_mapping(raw_value: Any, root: Path, video_ids: list[str], field_name: str, default_pattern: str | None = None) -> dict[str, Path]:
    if raw_value is None:
        if default_pattern is None:
            return {}
        return {video_id: (root / default_pattern.format(video_id=video_id)).resolve() for video_id in video_ids}

    if isinstance(raw_value, str):
        if "{video_id}" in raw_value:
            return {video_id: (root / raw_value.format(video_id=video_id)).resolve() for video_id in video_ids}
        if len(video_ids) != 1:
            raise ValueError(
                f"{field_name} is a single path string, but multiple videos were provided. "
                f"Use a mapping or a template with {{video_id}}."
            )
        return {video_ids[0]: (root / raw_value).resolve()}

    if isinstance(raw_value, dict):
        if "template" in raw_value:
            template = str(raw_value["template"])
            return {video_id: (root / template.format(video_id=video_id)).resolve() for video_id in video_ids}
        output: dict[str, Path] = {}
        for video_id in video_ids:
            if video_id not in raw_value:
                raise ValueError(f"Missing {field_name} entry for video '{video_id}'.")
            output[video_id] = (root / str(raw_value[video_id])).resolve()
        return output

    raise TypeError(f"{field_name} must be a string, a template mapping, or a mapping from video_id to path.")


def _load_config(path: Path) -> PrecomputeConfig:
    raw = _load_yaml(path)
    root = path.parent
    inputs = raw.get("inputs", {})
    video_ids = [str(v) for v in inputs.get("videos", [])]

    return PrecomputeConfig(
        annotations_root=(root / raw["annotations_root"]).resolve(),
        video_ids=video_ids,
        sampling_fps=float(raw.get("sampling_fps", 2.0)),
        fps_for_frame_lookup=float(raw.get("fps_for_frame_lookup", 30.0)),
        intermediate_root=str(raw.get("intermediate_root", DEFAULT_INTERMEDIATE_ROOT)),
        output_json_by_video=_parse_path_mapping(
            raw.get("visibility_tracks_json"),
            root=root,
            video_ids=video_ids,
            field_name="visibility_tracks_json",
            default_pattern="inview_track_{video_id}.json",
        ),
        visibility_status_jsonl_by_video=_parse_path_mapping(
            raw.get("visibility_status_jsonl"),
            root=root,
            video_ids=video_ids,
            field_name="visibility_status_jsonl",
        ) if raw.get("visibility_status_jsonl") is not None else {},
        start_time_sec=float(raw["start_time_sec"]) if raw.get("start_time_sec") is not None else None,
        end_time_sec=float(raw["end_time_sec"]) if raw.get("end_time_sec") is not None else None,
    )


def _build_payload_for_video(*, video_id: str, tracks: dict[str, Any], cfg: PrecomputeConfig) -> dict[str, Any]:
    if not tracks:
        return {
            "format_version": FORMAT_VERSION,
            "video_id": video_id,
            "sampling_fps": cfg.sampling_fps,
            "fps_for_frame_lookup": cfg.fps_for_frame_lookup,
            "intermediate_root": cfg.intermediate_root,
            "object_tracks": {},
            "video_window": None,
        }

    any_track = next(iter(tracks.values()))
    sampled_times = list(any_track.sampled_times_sec)
    video_window = {
        "start_sec": min(sampled_times),
        "end_sec": max(sampled_times),
        "num_samples": len(sampled_times),
    }

    return {
        "format_version": FORMAT_VERSION,
        "video_id": video_id,
        "sampling_fps": cfg.sampling_fps,
        "fps_for_frame_lookup": cfg.fps_for_frame_lookup,
        "intermediate_root": cfg.intermediate_root,
        "object_tracks": tracks_to_dict(tracks),
        "video_window": video_window,
    }


def precompute_visibility_tracks(cfg: PrecomputeConfig) -> dict[str, dict[str, Any]]:
    if not cfg.video_ids:
        raise ValueError("No input videos were provided in the config.")
    if cfg.output_json_by_video is None:
        raise ValueError("No output paths were resolved for visibility tracks.")

    payloads_by_video: dict[str, dict[str, Any]] = {}

    for video_id in cfg.video_ids:
        tracks = generate_in_view_tracks(
            video_id=video_id,
            annotations_root=cfg.annotations_root,
            sampling_fps=cfg.sampling_fps,
            fps_for_frame_lookup=cfg.fps_for_frame_lookup,
            intermediate_root=cfg.intermediate_root,
            start_time_sec=cfg.start_time_sec,
            end_time_sec=cfg.end_time_sec,
        )

        visibility_status_path = (cfg.visibility_status_jsonl_by_video or {}).get(video_id)
        if visibility_status_path is not None and visibility_status_path.exists():
            span_rows = load_visibility_span_rows(visibility_status_path)
            tracks = apply_external_visibility_spans(tracks, span_rows=span_rows, video_id=video_id)
        elif visibility_status_path is not None:
            raise FileNotFoundError(f"visibility_status_jsonl not found for {video_id}: {visibility_status_path}")

        payloads_by_video[video_id] = _build_payload_for_video(video_id=video_id, tracks=tracks, cfg=cfg)

    return payloads_by_video


def save_tracks_json(items: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute visibility tracks for staged OOS generation")
    parser.add_argument("--config", type=Path, required=True, help="Path to the benchmark config YAML")
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help=(
            "Optional output JSON override. Only valid when exactly one video is provided; "
            "otherwise use visibility_tracks_json mapping in the config."
        ),
    )
    parser.add_argument("--start_time_sec", type=float, default=None, help="Optional override for the track generation start time")
    parser.add_argument("--end_time_sec", type=float, default=None, help="Optional override for the track generation end time")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config.resolve())

    start_time_sec = args.start_time_sec if args.start_time_sec is not None else cfg.start_time_sec
    end_time_sec = args.end_time_sec if args.end_time_sec is not None else cfg.end_time_sec

    output_json_by_video = dict(cfg.output_json_by_video or {})

    if args.output_json is not None:
        if len(cfg.video_ids) != 1:
            raise ValueError("--output_json can only be used when exactly one input video is provided.")
        output_json_by_video[cfg.video_ids[0]] = args.output_json.resolve()

    cfg = PrecomputeConfig(
        **{
            **asdict(cfg),
            "output_json_by_video": output_json_by_video,
            "start_time_sec": start_time_sec,
            "end_time_sec": end_time_sec,
        }
    )

    payloads_by_video = precompute_visibility_tracks(cfg)

    for video_id in cfg.video_ids:
        output_path = output_json_by_video[video_id]
        save_tracks_json(payloads_by_video[video_id], output_path)
        print(f"[{video_id}] Saved JSON: {output_path}")

    print(f"Precomputed visibility tracks for {len(cfg.video_ids)} video(s)")


if __name__ == "__main__":
    main()
