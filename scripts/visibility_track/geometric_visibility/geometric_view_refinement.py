"""Stage 2.5: geometric visibility refinement via ray-casting against digital-twin meshes.

For every object sample marked as *in view* (within camera frustum) by
stage 2, this module casts a ray from the camera position to the object's 3D
world location and checks whether any fixed scene geometry (counters,
cupboards, shelves, …) blocks the line of sight.

Samples that are occluded are flagged with ``geometrically_occluded = True``
so that the downstream combine stage (stage 3) can classify them as
``not_visible_geometric_occlusion``.

The output file ``geometric_refined_in_view_tracks.jsonl`` has the same
schema as ``in_view_tracks.jsonl`` with one additional per-sample field:

    ``geometrically_occluded``: bool | None

This stage is optional. If skipped, combine.py falls back to the raw
``in_view_tracks.jsonl`` transparently (the field simply won't be present).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.visibility_track.common import PipelineConfig, load_config, read_jsonl, write_jsonl  # noqa: E402
    from scripts.visibility_track.geometric_visibility.mesh_scene import RayOcclusionChecker  # noqa: E402
    from scripts.visibility_track.in_view_track.in_view_determination import (  # noqa: E402
        load_frame_context,
        invert_rigid_4x4,
    )
    from scripts.visibility_track.in_view_track.in_view_track_generator import (  # noqa: E402
        ObjectInViewTrack,
        track_from_dict,
        track_to_dict,
    )
else:
    from ..common import PipelineConfig, load_config, read_jsonl, write_jsonl
    from .mesh_scene import RayOcclusionChecker
    from ..in_view_track.in_view_determination import load_frame_context, invert_rigid_4x4
    from ..in_view_track.in_view_track_generator import (
        ObjectInViewTrack,
        track_from_dict,
        track_to_dict,
    )


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "visibility_track_config.yaml"


def _camera_world_position(T_camera_world: list[list[float]]) -> np.ndarray:
    """Extract camera position in world coordinates from T_camera_world.

    T_camera_world maps world → camera.  The camera's world position is the
    translation component of the *inverse* transform (camera → world).
    """
    T_world_camera = invert_rigid_4x4(T_camera_world)
    return np.array([T_world_camera[0][3], T_world_camera[1][3], T_world_camera[2][3]])


def refine_tracks_geometric(
    video_id: str,
    in_view_tracks: dict[str, ObjectInViewTrack],
    checker: RayOcclusionChecker,
    annotations_root: str | Path,
    video_fps: float,
    intermediate_root: str,
    occlusion_tolerance: float = 0.05,
) -> dict[str, ObjectInViewTrack]:
    """Add ``geometrically_occluded`` flags to every in-view sample.

    For samples where the object is projected within the camera frustum
    (status == "ok" and in_view == True), a ray is cast from camera to
    object.  If fixed scene geometry blocks the ray, the sample is marked
    ``geometrically_occluded = True``.

    Camera positions are loaded per unique frame index to avoid redundant I/O.
    """
    # Collect unique sample times across all tracks so each query-time camera
    # pose is loaded once. Do not key by sample.frame_index: that index comes
    # from the selected object mask frame and may be stale relative to query
    # time when tracks are propagated from past/future segments.
    sample_times: set[float] = set()
    for track in in_view_tracks.values():
        for sample in track.samples:
            if sample.time_sec is not None:
                sample_times.add(float(sample.time_sec))

    # Pre-load camera world positions per unique sample time.
    camera_positions: dict[float, np.ndarray] = {}
    for time_sec in sorted(sample_times):
        try:
            ctx = load_frame_context(
                video_id=video_id,
                time_sec=time_sec,
                annotations_root=annotations_root,
                fps=video_fps,
                intermediate_root=intermediate_root,
            )
            camera_positions[time_sec] = _camera_world_position(ctx.T_camera_world)
        except Exception:
            # If we can't load frame context at a query time, skip it — samples
            # at that time will get geometrically_occluded = None.
            pass

    total_checks = sum(
        1
        for track in in_view_tracks.values()
        for s in track.samples
        if s.status == "ok" and s.in_view and s.world_coordinates is not None
    )

    with tqdm(total=total_checks, desc=f"{video_id} geometric", unit="ray") as pbar:
        for track in in_view_tracks.values():
            for sample in track.samples:
                if not (sample.status == "ok" and sample.in_view and sample.world_coordinates is not None):
                    sample.geometrically_occluded = None
                    continue

                cam_pos = camera_positions.get(float(sample.time_sec))
                if cam_pos is None:
                    sample.geometrically_occluded = None
                    pbar.update(1)
                    continue

                sample.geometrically_occluded = checker.is_occluded(
                    camera_world=cam_pos,
                    object_world=sample.world_coordinates,
                    tolerance=occlusion_tolerance,
                )
                pbar.update(1)

    return in_view_tracks


def _track_to_dict_with_occlusion(track: ObjectInViewTrack) -> dict:
    """Serialise track, including the ``geometrically_occluded`` field."""
    d = track_to_dict(track)
    for sample_dict, sample_obj in zip(d["samples"], track.samples):
        sample_dict["geometrically_occluded"] = getattr(sample_obj, "geometrically_occluded", None)
    return d


def run_stage(cfg: PipelineConfig, video_ids: List[str]) -> None:
    print(f"[stage 2.5] geometric visibility refinement for {len(video_ids)} video(s)")

    # Group videos by participant so we load each participant's mesh once.
    by_participant: dict[str, list[str]] = {}
    for vid in video_ids:
        pid = vid.split("-")[0]
        by_participant.setdefault(pid, []).append(vid)

    for participant_id, p_videos in by_participant.items():
        print(f"[stage 2.5] loading meshes for {participant_id} ...")
        checker = RayOcclusionChecker.from_data_root(cfg.data_root, participant_id)

        for video_id in p_videos:
            out_dir = cfg.video_output_dir(video_id)
            in_view_path = out_dir / "in_view_tracks.jsonl"
            if not in_view_path.exists():
                raise FileNotFoundError(f"Missing {in_view_path}. Run stage 2 first.")

            tracks = {row["assoc_id"]: track_from_dict(row) for row in read_jsonl(in_view_path)}
            tracks = refine_tracks_geometric(
                video_id=video_id,
                in_view_tracks=tracks,
                checker=checker,
                annotations_root=cfg.annotations_root,
                video_fps=cfg.video_fps,
                intermediate_root=str(cfg.intermediate_data_root),
                occlusion_tolerance=cfg.geometric_occlusion_tolerance,
            )

            out_path = out_dir / "geometric_refined_in_view_tracks.jsonl"
            write_jsonl(out_path, (_track_to_dict_with_occlusion(t) for t in tracks.values()))
            print(f"[stage 2.5] {video_id} -> {out_path}")


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
