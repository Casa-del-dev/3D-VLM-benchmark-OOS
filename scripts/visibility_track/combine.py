"""Stage 3: overlay fixture open/closed state onto in-view tracks.

First generates per-fixture open/closed intervals (same logic as
open_close_track.build_tracks). Then walks each object's in-view track and
refines the status of samples that are currently ``in_view`` or
``geometrically_occluded``:

  * If the object sits inside a **closed** fixture (medium+ confidence)
    -> ``occluded_inside_closed_fixture``
  * If the object sits inside an **open** fixture
    -> ``potentially_visible_inside_open_fixture``
  * Otherwise the sample keeps its current state (``in_view`` or
    ``geometrically_occluded``).

States ``in_motion``, ``unobservable_no_data``, and ``out_of_view`` pass
through unchanged.

Status precedence (the single status emitted for a sample is the FIRST
match in this ordered list — stages 2, 3, and 4 cooperate to produce it):

  1. in_motion                              (stage 1; person manipulating)
  2. unobservable_no_data                   (stage 1; no_track or no_valid_mask)
  3. out_of_view                            (stage 1; projection outside image,
                                             behind camera, or fisheye black border)
  4. occluded_inside_closed_fixture         (stage 3; inside closed fixture,
                                             confidence >= medium)
  5a. observed_visible_in_open_fixture       (stage 4; open fixture, detector
                                              ran with positives >= min_required)
  5b. observed_not_visible_in_open_fixture   (stage 4; open fixture, detector
                                              ran without enough positives)
  5c. assumed_not_visible_in_open_fixture    (stage 4; open fixture, detector
                                              never ran -- n_tested == 0)
  6. geometrically_occluded                 (stage 2; ray blocked, not inside
                                             any fixture)
  7. in_view                                (default; passes through)

Note: geometric occlusion is intentionally *discarded* when a sample sits
inside an open fixture — stage 4 takes over with detection.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.visibility_track.common import PipelineConfig, load_config, read_jsonl, write_jsonl  # noqa: E402
    from scripts.visibility_track.in_view_track.in_view_track_generator import (  # noqa: E402
        ObjectInViewTrack,
        ObjectSample,
        track_from_dict,
    )
    from scripts.visibility_track.open_close_track.build_tracks import (  # noqa: E402
        run_stage as build_fixture_intervals,
        interval_from_dict,
    )
    from scripts.visibility_track.open_close_track.state_machine import Interval as FixtureInterval  # noqa: E402
else:
    from .common import PipelineConfig, load_config, read_jsonl, write_jsonl
    from .in_view_track.in_view_track_generator import ObjectInViewTrack, ObjectSample, track_from_dict
    from .open_close_track.build_tracks import run_stage as build_fixture_intervals, interval_from_dict
    from .open_close_track.state_machine import Interval as FixtureInterval


DEFAULT_CONFIG = Path(__file__).resolve().parent / "visibility_track_config.yaml"

# --- Status constants --------------------------------------------------------

IN_VIEW = "in_view"
OUT_OF_VIEW = "out_of_view"
UNOBSERVABLE_NO_DATA = "unobservable_no_data"
IN_MOTION = "in_motion"
GEOMETRICALLY_OCCLUDED = "geometrically_occluded"
OCCLUDED_INSIDE_CLOSED_FIXTURE = "occluded_inside_closed_fixture"
POTENTIALLY_VISIBLE_INSIDE_OPEN_FIXTURE = "potentially_visible_inside_open_fixture"

# Statuses that stage 3 may emit. Stage 4 may upgrade
# POTENTIALLY_VISIBLE_INSIDE_OPEN_FIXTURE to one of three observed/assumed
# variants (see detection_refinement.py).
STAGE3_STATUSES: frozenset[str] = frozenset({
    IN_MOTION,
    UNOBSERVABLE_NO_DATA,
    OUT_OF_VIEW,
    OCCLUDED_INSIDE_CLOSED_FIXTURE,
    POTENTIALLY_VISIBLE_INSIDE_OPEN_FIXTURE,
    GEOMETRICALLY_OCCLUDED,
    IN_VIEW,
})

# Fixture confidence tiers. "assumed_closed" intentionally shares rank 4
# with "very_high": a fixture with zero open/close events is treated as
# confidently closed by default (see build_tracks.py zero-events placeholder).
CONFIDENCE_RANK: dict[str, int] = {
    "none": -1,
    "very_low": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "very_high": 4,
    "assumed_closed": 4,
}
CLOSED_CONFIDENCE_MIN_RANK: int = CONFIDENCE_RANK["medium"]


def confidence_rank(confidence: str | None) -> int:
    return CONFIDENCE_RANK.get(confidence, -1) if confidence is not None else -1


# --- Data --------------------------------------------------------------------

@dataclass
class CoarseInterval:
    video_id: str
    assoc_id: str
    object_name: str
    start_sec: float
    end_sec: float
    status: str
    reason: str
    fixture: str | None = None
    fixture_confidence: str | None = None


# --- Helpers -----------------------------------------------------------------

def _fixture_interval_at(
    intervals: list[FixtureInterval],
    fixture_id: str | None,
    time_sec: float,
) -> FixtureInterval | None:
    """Return the fixture interval active at *time_sec* for *fixture_id*."""
    if fixture_id is None:
        return None
    for interval in intervals:
        if interval.fixture_id != fixture_id:
            continue
        end_time = interval.end_time if interval.end_time is not None else float("inf")
        if interval.start_time <= time_sec <= end_time:
            return interval
    return None


# --- Per-sample classification -----------------------------------------------

def classify_sample(
    sample: ObjectSample,
    fixture_intervals: list[FixtureInterval],
) -> tuple[str, str, str | None]:
    """Classify a sample into a visibility state. Third value is the
    fixture confidence label active at this sample, or None if not in a fixture."""

    if sample.status == "in_motion":
        return IN_MOTION, "object in motion", None

    if sample.status in {"no_track_available", "no_valid_mask"}:
        return UNOBSERVABLE_NO_DATA, "no stable track or mask available", None

    if not sample.in_view:
        return OUT_OF_VIEW, "projection outside image", None

    if getattr(sample, "geometrically_occluded", None) is True:
        base_state = GEOMETRICALLY_OCCLUDED
    else:
        base_state = IN_VIEW

    fi = _fixture_interval_at(fixture_intervals, sample.fixture, sample.time_sec)

    if fi is not None:
        if fi.state == "closed" and confidence_rank(fi.confidence) >= CLOSED_CONFIDENCE_MIN_RANK:
            return (
                OCCLUDED_INSIDE_CLOSED_FIXTURE,
                f"inside closed {fi.fixture_type} ({fi.confidence} confidence)",
                fi.confidence,
            )
        if fi.state == "open":
            return (
                POTENTIALLY_VISIBLE_INSIDE_OPEN_FIXTURE,
                f"inside open {fi.fixture_type}",
                fi.confidence,
            )

    return base_state, (
        "in view, not inside a fixture"
        if base_state == IN_VIEW
        else "line of sight blocked by scene geometry"
    ), None


# --- Collapse consecutive same-status samples into intervals -----------------

def _collapse_track(
    video_id: str,
    track: ObjectInViewTrack,
    fixture_intervals: list[FixtureInterval],
) -> list[CoarseInterval]:
    if not track.samples:
        return []

    labels: list[tuple[str, str, str | None, str | None]] = []
    for sample in track.samples:
        status, reason, fixture_conf = classify_sample(sample, fixture_intervals)
        labels.append((status, reason, sample.fixture, fixture_conf))

    output: list[CoarseInterval] = []
    run_start = 0
    run_status, run_reason, run_fixture, run_fixture_conf = labels[0]

    for i in range(1, len(labels)):
        status, reason, fixture, fixture_conf = labels[i]
        if status == run_status:
            continue

        output.append(
            CoarseInterval(
                video_id=video_id,
                assoc_id=track.assoc_id,
                object_name=track.name,
                start_sec=track.samples[run_start].time_sec,
                end_sec=track.samples[i - 1].time_sec,
                status=run_status,
                reason=run_reason,
                fixture=run_fixture,
                fixture_confidence=run_fixture_conf,
            )
        )
        run_start = i
        run_status, run_reason, run_fixture, run_fixture_conf = status, reason, fixture, fixture_conf

    output.append(
        CoarseInterval(
            video_id=video_id,
            assoc_id=track.assoc_id,
            object_name=track.name,
            start_sec=track.samples[run_start].time_sec,
            end_sec=track.samples[-1].time_sec,
            status=run_status,
            reason=run_reason,
            fixture=run_fixture,
            fixture_confidence=run_fixture_conf,
        )
    )
    return output


def combine_tracks(
    video_id: str,
    in_view_tracks: Dict[str, ObjectInViewTrack],
    fixture_intervals: Iterable[FixtureInterval],
) -> list[CoarseInterval]:
    fixture_rows = list(fixture_intervals)
    output: list[CoarseInterval] = []
    for track in in_view_tracks.values():
        output.extend(_collapse_track(video_id, track, fixture_rows))
    output.sort(key=lambda row: (row.assoc_id, row.start_sec))
    _assert_coarse_invariants(output)
    return output


def _assert_coarse_invariants(intervals: list[CoarseInterval]) -> None:
    by_assoc: dict[str, list[CoarseInterval]] = {}
    for iv in intervals:
        if iv.status not in STAGE3_STATUSES:
            raise AssertionError(
                f"stage 3 emitted unknown status {iv.status!r} for assoc_id={iv.assoc_id}"
            )
        if iv.end_sec < iv.start_sec:
            raise AssertionError(
                f"stage 3 interval end_sec<start_sec for assoc_id={iv.assoc_id} "
                f"({iv.start_sec}..{iv.end_sec})"
            )
        by_assoc.setdefault(iv.assoc_id, []).append(iv)

    for assoc_id, rows in by_assoc.items():
        for prev, curr in zip(rows, rows[1:]):
            if curr.start_sec < prev.end_sec:
                raise AssertionError(
                    f"stage 3 overlapping intervals for assoc_id={assoc_id}: "
                    f"[{prev.start_sec}..{prev.end_sec}] then [{curr.start_sec}..{curr.end_sec}]"
                )


# --- Serialisation -----------------------------------------------------------

def coarse_to_dict(interval: CoarseInterval) -> dict:
    return {
        "video_id": interval.video_id,
        "assoc_id": interval.assoc_id,
        "object_name": interval.object_name,
        "start_sec": interval.start_sec,
        "end_sec": interval.end_sec,
        "status": interval.status,
        "reason": interval.reason,
        "fixture": interval.fixture,
        "fixture_confidence": interval.fixture_confidence,
    }


def coarse_from_dict(row: dict) -> CoarseInterval:
    return CoarseInterval(
        video_id=row["video_id"],
        assoc_id=row["assoc_id"],
        object_name=row["object_name"],
        start_sec=float(row["start_sec"]),
        end_sec=float(row["end_sec"]),
        status=row["status"],
        reason=row["reason"],
        fixture=row.get("fixture"),
        fixture_confidence=row.get("fixture_confidence"),
    )


# --- I/O helpers -------------------------------------------------------------

def _load_in_view_tracks(path: Path) -> Dict[str, ObjectInViewTrack]:
    return {row["assoc_id"]: track_from_dict(row) for row in read_jsonl(path)}


def _load_fixture_intervals(path: Path) -> List[FixtureInterval]:
    return [interval_from_dict(row) for row in read_jsonl(path)]


# --- Stage entry point -------------------------------------------------------

def run_stage(cfg: PipelineConfig, video_ids: List[str]) -> None:
    """Stage 3: build fixture intervals, then overlay onto in-view tracks."""

    # 3a) Generate per-fixture open/closed intervals.
    print("[stage 3] building fixture intervals ...")
    build_fixture_intervals(cfg, video_ids)

    # 3b) Overlay fixture state onto in-view tracks.
    print(f"[stage 3] overlaying fixture state for {len(video_ids)} video(s)")
    for video_id in video_ids:
        out_dir = cfg.video_output_dir(video_id)

        # Read the geometric-refined tracks if geometric occlusion was run,
        # otherwise read the raw in-view tracks.
        geo_path = out_dir / "geometric_refined_in_view_tracks.jsonl"
        raw_path = out_dir / "in_view_tracks.jsonl"
        in_view_path = geo_path if (cfg.geometric_occlusion_enabled and geo_path.exists()) else raw_path
        if not in_view_path.exists():
            raise FileNotFoundError(f"Missing in-view tracks in {out_dir}. Run stage 1 first.")

        fixture_path = out_dir / "fixture_intervals.jsonl"
        if not fixture_path.exists():
            raise FileNotFoundError(f"Missing {fixture_path}. Fixture interval generation failed.")

        in_view_tracks = _load_in_view_tracks(in_view_path)
        fixture_intervals = _load_fixture_intervals(fixture_path)
        coarse = combine_tracks(video_id, in_view_tracks, fixture_intervals)

        out_path = out_dir / "coarse_visibility_track.jsonl"
        write_jsonl(out_path, (coarse_to_dict(interval) for interval in coarse))
        print(f"[stage 3] {video_id} -> {out_path}")


# --- CLI ---------------------------------------------------------------------

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
