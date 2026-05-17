"""Stage 2: sample per-object in-view state over time."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.visibility_track.common import PipelineConfig, load_config, write_jsonl  # noqa: E402
    from scripts.visibility_track.in_view_track.in_view_determination import (  # noqa: E402
        DEFAULT_INTERMEDIATE_ROOT,
        VideoCache,
        determine_in_view_objects,
        load_jsonl,
    )
else:
    from ..common import PipelineConfig, load_config, write_jsonl
    from .in_view_determination import (
        DEFAULT_INTERMEDIATE_ROOT,
        VideoCache,
        determine_in_view_objects,
        load_jsonl,
    )


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "visibility_track_config.yaml"


@dataclass(frozen=True)
class VisibilitySpan:
    start_sec: float
    end_sec: float
    in_view: bool


@dataclass
class ObjectSample:
    time_sec: float
    status: str
    in_view: bool | None
    projected_uv: list[float] | None
    mask_bbox: list[float] | None
    fixture: str | None
    frame_index: int | None
    world_coordinates: list[float] | None = None
    camera_coordinates: list[float] | None = None
    geometrically_occluded: bool | None = None
    occlusion_fraction: float | None = None


@dataclass
class ObjectInViewTrack:
    assoc_id: str
    name: str
    sampled_times_sec: list[float]
    samples: list[ObjectSample]
    in_view_spans: list[VisibilitySpan] = field(default_factory=list)


def _infer_time_window_sec(
    video_id: str,
    annotations_root: str | Path,
    fps: float,
    intermediate_root: str,
) -> tuple[float, float]:
    annotations_root = Path(annotations_root)
    participant_id = video_id.split("-")[0]
    framewise_path = annotations_root / intermediate_root / participant_id / video_id / "framewise_info.jsonl"
    rows = load_jsonl(framewise_path)
    frame_indices = [int(row["frame_index"]) for row in rows if row.get("frame_index") is not None]
    if not frame_indices:
        raise ValueError(f"No frame indices in {framewise_path}")
    return min(frame_indices) / fps, max(frame_indices) / fps


def build_sample_times(start_sec: float, end_sec: float, sampling_fps: float) -> list[float]:
    if sampling_fps <= 0:
        raise ValueError("sampling_fps must be > 0")
    if end_sec < start_sec:
        raise ValueError("end_sec must be >= start_sec")

    step = 1.0 / sampling_fps
    times: list[float] = []
    t = start_sec
    while t <= end_sec + 1e-9:
        times.append(round(t, 6))
        t += step
    return times


def _collapse(flags: list[bool], times: list[float]) -> list[VisibilitySpan]:
    if not flags:
        return []

    spans: list[VisibilitySpan] = []
    run_start = 0
    run_value = flags[0]

    for i in range(1, len(flags)):
        if flags[i] != run_value:
            spans.append(VisibilitySpan(times[run_start], times[i - 1], run_value))
            run_start = i
            run_value = flags[i]

    spans.append(VisibilitySpan(times[run_start], times[-1], run_value))
    return spans


def generate_in_view_tracks(
    video_id: str,
    annotations_root: str | Path,
    sampling_fps: float = 1.0,
    fps_for_frame_lookup: float = 30.0,
    intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT,
    start_time_sec: float | None = None,
    end_time_sec: float | None = None,
) -> dict[str, ObjectInViewTrack]:
    if start_time_sec is None or end_time_sec is None:
        inferred_start, inferred_end = _infer_time_window_sec(
            video_id=video_id,
            annotations_root=annotations_root,
            fps=fps_for_frame_lookup,
            intermediate_root=intermediate_root,
        )
        if start_time_sec is None:
            start_time_sec = inferred_start
        if end_time_sec is None:
            end_time_sec = inferred_end

    times = build_sample_times(start_time_sec, end_time_sec, sampling_fps)
    if not times:
        return {}

    # Build cache once per video — avoids re-reading large JSON files each step.
    video_cache = VideoCache.build(video_id, annotations_root, intermediate_root)

    names: dict[str, str] = {}
    samples_by_object: dict[str, list[ObjectSample]] = {}

    for time_sec in tqdm(times, desc=video_id, unit="s", leave=False):
        states = determine_in_view_objects(
            video_id=video_id,
            time_sec=time_sec,
            annotations_root=annotations_root,
            fps=fps_for_frame_lookup,
            intermediate_root=intermediate_root,
            cache=video_cache,
        )
        for state in states:
            names.setdefault(state.assoc_id, state.name)
            samples_by_object.setdefault(state.assoc_id, [])
            samples_by_object[state.assoc_id].append(
                ObjectSample(
                    time_sec=time_sec,
                    status=state.status,
                    in_view=state.in_view,
                    projected_uv=list(state.projected_pixel) if state.projected_pixel is not None else None,
                    mask_bbox=list(state.mask_bbox) if state.mask_bbox is not None else None,
                    fixture=state.fixture,
                    frame_index=state.frame_number,
                    world_coordinates=list(state.world_coordinates) if state.world_coordinates is not None else None,
                    camera_coordinates=list(state.camera_coordinates) if state.camera_coordinates is not None else None,
                )
            )

    tracks: dict[str, ObjectInViewTrack] = {}
    for assoc_id, samples in samples_by_object.items():
        flags = [bool(sample.status == "ok" and sample.in_view) for sample in samples]
        sample_times = [sample.time_sec for sample in samples]
        tracks[assoc_id] = ObjectInViewTrack(
            assoc_id=assoc_id,
            name=names[assoc_id],
            sampled_times_sec=times,
            samples=samples,
            in_view_spans=_collapse(flags, sample_times),
        )

    return tracks


def track_to_dict(track: ObjectInViewTrack) -> dict[str, Any]:
    return {
        "assoc_id": track.assoc_id,
        "name": track.name,
        "sampled_times_sec": track.sampled_times_sec,
        "in_view_spans": [
            {"start_sec": span.start_sec, "end_sec": span.end_sec, "in_view": span.in_view}
            for span in track.in_view_spans
        ],
        "samples": [
            {
                "time_sec": sample.time_sec,
                "status": sample.status,
                "in_view": sample.in_view,
                "projected_uv": sample.projected_uv,
                "mask_bbox": sample.mask_bbox,
                "fixture": sample.fixture,
                "frame_index": sample.frame_index,
                "world_coordinates": sample.world_coordinates,
                "camera_coordinates": sample.camera_coordinates,
            }
            for sample in track.samples
        ],
    }


def track_from_dict(row: dict[str, Any]) -> ObjectInViewTrack:
    samples = [
        ObjectSample(
            time_sec=float(sample["time_sec"]),
            status=sample["status"],
            in_view=sample.get("in_view"),
            projected_uv=sample.get("projected_uv"),
            mask_bbox=sample.get("mask_bbox"),
            fixture=sample.get("fixture"),
            frame_index=sample.get("frame_index"),
            world_coordinates=sample.get("world_coordinates"),
            camera_coordinates=sample.get("camera_coordinates"),
            geometrically_occluded=sample.get("geometrically_occluded"),
            occlusion_fraction=sample.get("occlusion_fraction"),
        )
        for sample in row.get("samples", [])
    ]

    spans = [
        VisibilitySpan(
            start_sec=float(span["start_sec"]),
            end_sec=float(span["end_sec"]),
            in_view=bool(span["in_view"]),
        )
        for span in row.get("in_view_spans", [])
    ]

    return ObjectInViewTrack(
        assoc_id=row["assoc_id"],
        name=row["name"],
        sampled_times_sec=list(row.get("sampled_times_sec", [])),
        samples=samples,
        in_view_spans=spans,
    )


def run_stage(cfg: PipelineConfig, video_ids: List[str]) -> None:
    print(f"[stage 2] building in-view tracks for {len(video_ids)} video(s)")
    for video_id in video_ids:
        tracks = generate_in_view_tracks(
            video_id=video_id,
            annotations_root=cfg.annotations_root,
            sampling_fps=cfg.in_view_sampling_fps,
            fps_for_frame_lookup=cfg.video_fps,
            intermediate_root=str(cfg.intermediate_data_root),
        )
        out_dir = cfg.video_output_dir(video_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "in_view_tracks.jsonl"
        write_jsonl(out_path, (track_to_dict(track) for track in tracks.values()))
        print(f"[stage 2] {video_id} -> {out_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    video_ids = args.video or cfg.videos
    if not video_ids:
        raise ValueError("No videos configured. Populate inputs.videos or pass --video.")
    run_stage(cfg, video_ids)


if __name__ == "__main__":
    main()
