"""Stage 1: build per-fixture open/closed intervals.

Wires together the open/close-track sub-modules into a single call:

  narrations  -> NarrationEvent list (open/close verbs only, sound-snapped)
  fixtures    -> per-kitchen catalog of openable fixture instances
  resolver    -> map each event to a concrete fixture_id (+ confidence)
  state_machine -> collapse resolved events into per-fixture intervals

Emits `fixture_intervals.jsonl` per video, consumed by stage 3.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.visibility_track.common import PipelineConfig, load_config, write_jsonl  # noqa: E402
    from scripts.visibility_track.open_close_track import config as oc_config, fixtures, framewise, narrations  # noqa: E402
    from scripts.visibility_track.open_close_track.resolver import ResolvedEvent, resolve_event  # noqa: E402
    from scripts.visibility_track.open_close_track.state_machine import Interval, build_intervals  # noqa: E402
else:
    from ..common import PipelineConfig, load_config, write_jsonl
    from . import config as oc_config, fixtures, framewise, narrations
    from .resolver import ResolvedEvent, resolve_event
    from .state_machine import Interval, build_intervals


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "visibility_track_config.yaml"


def interval_to_dict(interval: Interval) -> dict:
    return asdict(interval)


def interval_from_dict(row: dict) -> Interval:
    return Interval(
        video_id=row["video_id"],
        fixture_id=row["fixture_id"],
        fixture_type=row["fixture_type"],
        state=row["state"],
        start_time=float(row["start_time"]),
        end_time=None if row.get("end_time") is None else float(row["end_time"]),
        confidence=str(row.get("confidence", "none")),
        source_events=list(row.get("source_events", [])),
        notes=list(row.get("notes", [])),
        backfilled=bool(row.get("backfilled", False)),
    )


def _participant_of(video_id: str) -> str:
    return video_id.split("-", 1)[0]


def build_fixture_intervals(
    cfg: PipelineConfig,
    video_ids: Iterable[str] | None = None,
) -> tuple[Dict[str, List[Interval]], List[ResolvedEvent]]:
    selected = set(video_ids) if video_ids else None

    sounds_path = cfg.sounds_csv if cfg.sounds_csv.exists() else None
    events = narrations.extract_events(cfg.narrations_pkl, sounds_path=sounds_path)
    if selected is not None:
        events = [event for event in events if event.video_id in selected]

    catalog = fixtures.build_kitchen_catalog(cfg.mask_info_json)
    by_video: Dict[str, List[narrations.NarrationEvent]] = defaultdict(list)
    for event in events:
        by_video[event.video_id].append(event)

    # The set of videos we must emit intervals for: whatever the caller asked
    # for, plus any video that happens to have events.
    target_videos: set[str] = set(by_video.keys())
    if selected is not None:
        target_videos |= selected

    resolved: list[ResolvedEvent] = []
    video_end: Dict[str, float] = {}

    for video_id in target_videos:
        frame_seq = framewise.load_framewise(cfg.framewise_path(video_id))
        if frame_seq:
            video_end[video_id] = oc_config.frame_to_time(frame_seq[-1].frame_index)
        elif by_video[video_id]:
            video_end[video_id] = max(event.end_time for event in by_video[video_id])
        else:
            # No events and no frame data — we have nothing to anchor on.
            video_end[video_id] = 0.0

        participant = _participant_of(video_id)
        kitchen = catalog.get(participant, {})
        for event in by_video[video_id]:
            resolved.append(resolve_event(event=event, kitchen=kitchen, framewise=frame_seq))

    intervals = build_intervals(resolved, video_end)
    intervals_by_video: Dict[str, List[Interval]] = defaultdict(list)
    for interval in intervals:
        intervals_by_video[interval.video_id].append(interval)

    # Fixtures with zero events are assumed closed for the whole video. Emit a
    # single closed-interval placeholder for every fixture in the kitchen that
    # the state-machine didn't already cover, so downstream code can look up
    # *any* fixture_id without special-casing "no intervals".
    for video_id in target_videos:
        participant = _participant_of(video_id)
        kitchen = catalog.get(participant, {})
        seen = {interval.fixture_id for interval in intervals_by_video[video_id]}
        end = video_end.get(video_id)
        for fixture_id, fixture in kitchen.items():
            if fixture_id in seen:
                continue
            intervals_by_video[video_id].append(
                Interval(
                    video_id=video_id,
                    fixture_id=fixture_id,
                    fixture_type=fixture.fixture_type,
                    state="closed",
                    start_time=0.0,
                    end_time=end,
                    confidence="assumed_closed",
                    source_events=["__no_events__"],
                    notes=["no open/close events observed for this fixture"],
                )
            )
        intervals_by_video[video_id].sort(key=lambda row: (row.fixture_id, row.start_time))

    return intervals_by_video, resolved


def run_stage(cfg: PipelineConfig, video_ids: List[str]) -> None:
    print(f"[stage 1] building fixture intervals for {len(video_ids)} video(s)")
    intervals_by_video, _ = build_fixture_intervals(cfg, video_ids=video_ids)

    for video_id in video_ids:
        out_dir = cfg.video_output_dir(video_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "fixture_intervals.jsonl"
        rows = (interval_to_dict(interval) for interval in intervals_by_video.get(video_id, []))
        write_jsonl(out_path, rows)
        print(f"[stage 1] {video_id} -> {out_path}")


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
