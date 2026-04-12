"""Driver: run all four visibility-track stages in order.

  1.  in_view_track.in_view_track_generator
        -> in_view_tracks.jsonl
        States: in_view, out_of_view, in_motion

  2.  geometric_visibility.geometric_view_refinement  (optional)
        -> geometric_refined_in_view_tracks.jsonl
        States: in_view, out_of_view, in_motion, geometrically_occluded

  3.  combine  (builds fixture intervals, then overlays onto in-view tracks)
        -> fixture_intervals.jsonl + coarse_visibility_track.jsonl
        States: in_view, out_of_view, in_motion, geometrically_occluded,
                occluded_inside_closed_fixture,
                potentially_visible_inside_open_fixture

  4.  detection_refinement  (optional)
        -> visibility_track.jsonl + visibility_track_summary.json
        States: in_view, out_of_view, in_motion, geometrically_occluded,
                occluded_inside_closed_fixture,
                observed_visible_in_open_fixture,
                observed_not_visible_in_open_fixture

Stages are individually importable; this script just chains them with a
shared ``PipelineConfig``.  Use ``--video`` to override the video list,
``--no-detection`` to skip stage 4, or ``--no-geometric`` to skip stage 2.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.visibility_track.common import PipelineConfig, load_config  # noqa: E402
    from scripts.visibility_track.in_view_track.in_view_track_generator import run_stage as stage1_run  # noqa: E402
    from scripts.visibility_track.geometric_visibility.geometric_view_refinement import run_stage as stage2_run  # noqa: E402
    from scripts.visibility_track.combine import run_stage as stage3_run  # noqa: E402
    from scripts.visibility_track.detection_refinement import run_stage as stage4_run  # noqa: E402
else:
    from .common import PipelineConfig, load_config
    from .in_view_track.in_view_track_generator import run_stage as stage1_run
    from .geometric_visibility.geometric_view_refinement import run_stage as stage2_run
    from .combine import run_stage as stage3_run
    from .detection_refinement import run_stage as stage4_run


DEFAULT_CONFIG = Path(__file__).resolve().parent / "visibility_track_config.yaml"


def run_pipeline(cfg: PipelineConfig, video_ids: list[str]) -> None:
    random.seed(cfg.random_seed)

    stage1_run(cfg, video_ids)                          # 1. in-view track

    if cfg.geometric_occlusion_enabled:
        stage2_run(cfg, video_ids)                      # 2. geometric refinement

    stage3_run(cfg, video_ids)                          # 3. fixture overlay

    if cfg.detection.enabled:
        stage4_run(cfg, video_ids)                      # 4. detection refinement


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", action="append", default=None,
                        help="Process specific video ID(s). May be repeated.")
    parser.add_argument("--participant", action="append", default=None,
                        help="Process all videos for a participant. May be repeated.")
    parser.add_argument("--no-geometric", action="store_true")
    parser.add_argument("--no-detection", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    if args.no_geometric:
        cfg.geometric_occlusion_enabled = False
    if args.no_detection:
        cfg.detection.enabled = False

    video_ids: list[str] = list(args.video or [])

    if args.participant:
        for participant in args.participant:
            for v in cfg.videos_for_participant(participant):
                if v not in video_ids:
                    video_ids.append(v)

    if not video_ids:
        # No explicit flags — fall back to config
        for participant in cfg.participants:
            for v in cfg.videos_for_participant(participant):
                if v not in video_ids:
                    video_ids.append(v)

    if not video_ids:
        video_ids = cfg.videos

    if not video_ids:
        raise ValueError(
            "No videos to process. Pass --video, --participant, or populate inputs in the config."
        )

    print(f"[pipeline] videos: {video_ids}")
    print(f"[pipeline] output: {cfg.output_root}")
    run_pipeline(cfg, video_ids)
    print("[pipeline] done")


if __name__ == "__main__":
    main()
