"""Stage 2.5: geometric visibility refinement via ray-casting against digital-twin meshes.

For every object sample marked as *in view* by stage 1, this casts one
or more rays from the camera to world-space target points around the
object and reports the fraction blocked by fixed scene geometry
(counters, cupboards, shelves, ...).

When ``multi_ray`` is true, target points come from back-projecting the
mask's 2D bbox corners + edge midpoints + center pixel through the mask
reference frame's FISHEYE624 camera into world space at the centroid's
depth. The un-projection uses
``projectaria_tools.core.calibration.CameraCalibration`` (the same model
that produced the 2D annotations), so the back-projected rays follow
the real fisheye geometry rather than a pinhole approximation. Bbox
pixels that fall outside the fisheye's valid region are skipped, so the
test simply uses fewer rays for that sample.

When ``multi_ray`` is false, or when the per-sample 2D bbox / mask frame
is missing, we fall back to a single ray from the camera to the
centroid.

The fraction of blocked rays is stored as ``occlusion_fraction``; the
boolean ``geometrically_occluded`` is then derived from
``occlusion_fraction >= occlusion_threshold``.

Per-sample output fields written to ``geometric_refined_in_view_tracks.jsonl``:

    ``geometrically_occluded`` : bool | None
    ``occlusion_fraction``     : float | None

This stage is optional. If skipped, combine.py falls back to the raw
``in_view_tracks.jsonl`` transparently.
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
        FrameContext,
        VideoCache,
        load_frame_context,
        invert_rigid_4x4,
        transform_point,
    )
    from scripts.visibility_track.in_view_track.in_view_track_generator import (  # noqa: E402
        ObjectInViewTrack,
        track_from_dict,
        track_to_dict,
    )
else:
    from ..common import PipelineConfig, load_config, read_jsonl, write_jsonl
    from .mesh_scene import RayOcclusionChecker
    from ..in_view_track.in_view_determination import (
        FrameContext,
        VideoCache,
        load_frame_context,
        invert_rigid_4x4,
        transform_point,
    )
    from ..in_view_track.in_view_track_generator import (
        ObjectInViewTrack,
        track_from_dict,
        track_to_dict,
    )


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "visibility_track_config.yaml"


# --- Aria camera un-projection ----------------------------------------------

def _build_aria_calibration(ctx: FrameContext):
    """Wrap a FrameContext's FISHEYE624 parameters in a projectaria_tools
    CameraCalibration. Cached by the caller via (image_w, image_h,
    projection_params) since calibration objects are stateless per-frame.
    """
    from projectaria_tools.core.calibration import CameraCalibration, CameraModelType  # noqa: E402
    from projectaria_tools.core.sophus import SE3  # noqa: E402

    params = np.asarray(ctx.projection_params, dtype=np.float64)
    return CameraCalibration(
        "fisheye624",
        CameraModelType.FISHEYE624,
        params,
        SE3(),                # T_Device_Camera; not used here
        ctx.image_width,
        ctx.image_height,
        None,                 # valid_radius
        180.0,                # max_solid_angle (degrees)
        "",                   # serial_number
        0.0,                  # time_offset_sec_device_camera
        0.0,                  # readout_time_sec
    )


def _unit_ray_camera(calib, pixel_uv: tuple[float, float]) -> np.ndarray | None:
    """Camera-frame ray direction for pixel (u, v), z-normalised (z=1) as
    returned by projectaria_tools' ``unproject``. Returns ``None`` when
    the pixel falls outside the camera's valid fisheye region so callers
    can skip that sample instead of biasing it onto the optical axis."""
    ray = calib.unproject(np.asarray(pixel_uv, dtype=np.float64))
    if ray is None or ray.size != 3:
        return None
    return ray.astype(np.float64)


def _bbox_pixel_samples(bbox_xyxy: list[float]) -> list[tuple[float, float]]:
    """Nine sample pixels for back-projection: 4 corners + 4 edge midpoints + center."""
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return [
        (cx, cy),         # center
        (x1, y1),         # TL
        (x2, y1),         # TR
        (x1, y2),         # BL
        (x2, y2),         # BR
        (cx, y1),         # top mid
        (cx, y2),         # bottom mid
        (x1, cy),         # left mid
        (x2, cy),         # right mid
    ]


def _world_targets_from_bbox(
    mask_ctx: FrameContext,
    world_centroid: np.ndarray,
    bbox_xyxy: list[float],
) -> np.ndarray:
    """Build (N, 3) world-space target points by back-projecting bbox pixel
    samples through *mask_ctx*'s FISHEYE624 camera to the centroid's depth,
    then transforming camera→world.
    """
    calib = _build_aria_calibration(mask_ctx)

    # Centroid in the mask frame's camera coordinates: we need its z (depth
    # along the camera axis) to anchor all targets at the same depth.
    cam_centroid = np.asarray(
        transform_point(mask_ctx.T_camera_world, world_centroid.tolist()),
        dtype=np.float64,
    )
    z_depth = float(cam_centroid[2])
    if z_depth <= 1e-6:
        # Centroid is at or behind the camera in the mask frame — fall back
        # to a single world-space point at the centroid.
        return world_centroid.reshape(1, 3)

    T_world_camera = np.asarray(
        invert_rigid_4x4(mask_ctx.T_camera_world), dtype=np.float64
    )
    R_world_camera = T_world_camera[:3, :3]
    t_world_camera = T_world_camera[:3, 3]

    targets_world: list[np.ndarray] = []
    for u, v in _bbox_pixel_samples(bbox_xyxy):
        ray_cam = _unit_ray_camera(calib, (u, v))
        if ray_cam is None:
            continue  # pixel outside fisheye valid region
        # Scale the ray so its camera-frame z matches the centroid depth.
        cam_pt = ray_cam * (z_depth / ray_cam[2])
        world_pt = R_world_camera @ cam_pt + t_world_camera
        targets_world.append(world_pt)

    if not targets_world:
        return world_centroid.reshape(1, 3)
    return np.stack(targets_world, axis=0)


# --- Camera world position helper --------------------------------------------

def _camera_world_position(T_camera_world: list[list[float]]) -> np.ndarray:
    """Camera position in world coordinates from T_camera_world (world→camera)."""
    T_world_camera = invert_rigid_4x4(T_camera_world)
    return np.array([T_world_camera[0][3], T_world_camera[1][3], T_world_camera[2][3]])


# --- Stage 2 ----------------------------------------------------------------

def refine_tracks_geometric(
    video_id: str,
    in_view_tracks: dict[str, ObjectInViewTrack],
    checker: RayOcclusionChecker,
    annotations_root: str | Path,
    video_fps: float,
    intermediate_root: str,
    occlusion_tolerance: float = 0.05,
    multi_ray: bool = True,
    occlusion_threshold: float = 0.5,
    cache: VideoCache | None = None,
) -> dict[str, ObjectInViewTrack]:
    """Annotate every in-view sample with ``geometrically_occluded`` and
    ``occlusion_fraction``.

    For each sample with valid in-view projection:
      - if ``multi_ray`` is true and a per-sample 2D bbox + mask frame
        are available, back-projects the 2D bbox corners + midpoints +
        center through the mask reference frame's FISHEYE624 camera
        into world space at the centroid's depth;
      - otherwise casts a single ray from the camera to the centroid.
    """
    if cache is None:
        cache = VideoCache.build(video_id, annotations_root, intermediate_root)

    # Pre-load camera world positions per unique *sample* time so each
    # query-time pose is fetched once.
    sample_times: set[float] = set()
    mask_frame_indices: set[int] = set()
    for track in in_view_tracks.values():
        for sample in track.samples:
            if sample.time_sec is not None:
                sample_times.add(float(sample.time_sec))
            if (
                multi_ray
                and sample.frame_index is not None
                and sample.mask_bbox is not None
                and sample.status == "ok"
                and sample.in_view
                and sample.world_coordinates is not None
            ):
                mask_frame_indices.add(int(sample.frame_index))

    camera_positions: dict[float, np.ndarray] = {}
    for time_sec in sorted(sample_times):
        try:
            ctx = load_frame_context(
                video_id=video_id,
                time_sec=time_sec,
                annotations_root=annotations_root,
                fps=video_fps,
                intermediate_root=intermediate_root,
                cache=cache,
            )
            camera_positions[time_sec] = _camera_world_position(ctx.T_camera_world)
        except Exception:
            pass

    # Pre-load mask-frame camera contexts so the projectaria_tools
    # CameraCalibration can be reused across samples that share the same
    # mask reference frame.
    mask_frame_ctx: dict[int, FrameContext] = {}
    for frame_index in sorted(mask_frame_indices):
        try:
            mask_frame_ctx[frame_index] = load_frame_context(
                video_id=video_id,
                time_sec=float(frame_index) / float(video_fps),
                annotations_root=annotations_root,
                fps=video_fps,
                intermediate_root=intermediate_root,
                cache=cache,
            )
        except Exception:
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
                    sample.occlusion_fraction = None
                    continue

                cam_pos = camera_positions.get(float(sample.time_sec))
                if cam_pos is None:
                    sample.geometrically_occluded = None
                    sample.occlusion_fraction = None
                    pbar.update(1)
                    continue

                world_centroid = np.asarray(sample.world_coordinates, dtype=np.float64)

                targets: np.ndarray
                if (
                    multi_ray
                    and sample.mask_bbox is not None
                    and sample.frame_index is not None
                    and int(sample.frame_index) in mask_frame_ctx
                ):
                    targets = _world_targets_from_bbox(
                        mask_ctx=mask_frame_ctx[int(sample.frame_index)],
                        world_centroid=world_centroid,
                        bbox_xyxy=list(sample.mask_bbox),
                    )
                else:
                    targets = world_centroid.reshape(1, 3)

                frac = checker.occlusion_fraction(
                    camera_world=cam_pos,
                    target_points_world=targets,
                    tolerance=occlusion_tolerance,
                )
                sample.occlusion_fraction = float(frac)
                sample.geometrically_occluded = bool(frac >= occlusion_threshold)
                pbar.update(1)

    return in_view_tracks


def _track_to_dict_with_occlusion(track: ObjectInViewTrack) -> dict:
    """Serialise track, including the per-sample occlusion fields."""
    d = track_to_dict(track)
    for sample_dict, sample_obj in zip(d["samples"], track.samples):
        sample_dict["geometrically_occluded"] = getattr(sample_obj, "geometrically_occluded", None)
        sample_dict["occlusion_fraction"] = getattr(sample_obj, "occlusion_fraction", None)
    return d


def run_stage(cfg: PipelineConfig, video_ids: List[str]) -> None:
    print(f"[stage 2.5] geometric visibility refinement for {len(video_ids)} video(s)")

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

            cache = VideoCache.build(video_id, cfg.annotations_root, str(cfg.intermediate_data_root))
            tracks = {row["assoc_id"]: track_from_dict(row) for row in read_jsonl(in_view_path)}
            tracks = refine_tracks_geometric(
                video_id=video_id,
                in_view_tracks=tracks,
                checker=checker,
                annotations_root=cfg.annotations_root,
                video_fps=cfg.video_fps,
                intermediate_root=str(cfg.intermediate_data_root),
                occlusion_tolerance=cfg.geometric_occlusion_tolerance,
                multi_ray=cfg.geometric_occlusion_multi_ray,
                occlusion_threshold=cfg.geometric_occlusion_threshold,
                cache=cache,
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
