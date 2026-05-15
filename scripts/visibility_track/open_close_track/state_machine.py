"""Turn resolved open/close events into per-fixture intervals.

State-fill policy (fixtures with at least one event):
  * Between events: forward-fill from each event's target state until the
    next transition.
  * Before the first event: backward-fill with the *opposite* of the first
    event (e.g. if the first action is `close`, the fixture must have been
    `open` beforehand).
  * After the last event: forward-fill the last target state until video end.

Fixtures with no events are emitted as a single `closed` interval spanning
the whole video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .resolver import ResolvedEvent


@dataclass
class Interval:
    video_id: str
    fixture_id: str
    fixture_type: str
    state: str
    start_time: float
    end_time: float | None
    confidence: str
    source_events: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    backfilled: bool = False


def _group_by_fixture(events: Iterable[ResolvedEvent]) -> Dict[tuple[str, str], list[ResolvedEvent]]:
    grouped: Dict[tuple[str, str], list[ResolvedEvent]] = {}
    for event in events:
        if event.fixture_id is None:
            continue
        key = (event.event.video_id, event.fixture_id)
        grouped.setdefault(key, []).append(event)
    for rows in grouped.values():
        rows.sort(key=lambda row: row.event.start_time)
    return grouped


def build_intervals(
    resolved_events: Iterable[ResolvedEvent],
    video_end_times: Dict[str, float] | None = None,
) -> List[Interval]:
    video_end_times = video_end_times or {}
    output: list[Interval] = []

    for (video_id, fixture_id), rows in _group_by_fixture(resolved_events).items():
        fixture_type = rows[0].fixture_type or ""

        # Backward-fill: state before the first event is the opposite of that
        # event's target. If first action is `close`, it must have been open.
        first_target = "open" if rows[0].event.verb == "open" else "closed"
        current_state = "open" if first_target == "closed" else "closed"
        current_start = 0.0
        current_conf = rows[0].confidence
        current_sources = ["__backfill__"]
        current_notes: list[str] = [f"backfilled from first {rows[0].event.verb}"]
        current_backfilled = True

        for row in rows:
            target_state = "open" if row.event.verb == "open" else "closed"
            if target_state == current_state:
                current_sources.append(row.event.unique_narration_id)
                current_notes.append(f"duplicate {row.event.verb}")
                continue

            output.append(
                Interval(
                    video_id=video_id,
                    fixture_id=fixture_id,
                    fixture_type=fixture_type,
                    state=current_state,
                    start_time=current_start,
                    end_time=row.event.end_time,
                    confidence=current_conf,
                    source_events=list(current_sources),
                    notes=list(current_notes),
                    backfilled=current_backfilled,
                )
            )

            current_state = target_state
            current_start = row.event.end_time
            current_conf = row.confidence
            current_sources = [row.event.unique_narration_id]
            current_notes = []
            current_backfilled = False

        output.append(
            Interval(
                video_id=video_id,
                fixture_id=fixture_id,
                fixture_type=fixture_type,
                state=current_state,
                start_time=current_start,
                end_time=video_end_times.get(video_id),
                confidence=current_conf,
                source_events=list(current_sources),
                notes=list(current_notes),
                backfilled=current_backfilled,
            )
        )

    output.sort(key=lambda row: (row.video_id, row.fixture_id, row.start_time))
    return output
