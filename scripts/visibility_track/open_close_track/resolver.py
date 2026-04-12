"""Resolve open/close narration events to concrete fixture instances.

Given a `NarrationEvent` ("open the cupboard") and the kitchen catalog, this
picks the specific fixture_id the narrator interacted with and attaches a
confidence tier:

* very_high — only one fixture of this type exists in the kitchen.
* high / medium / low / very_low — multiple candidates, scored by combining
  camera-gaze alignment with physical proximity (camera position near the
  fixture centroid), averaged over the event window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from . import config
from .fixtures import Fixture
from .framewise import FrameInfo, Vec3
from .narrations import NarrationEvent


@dataclass
class ResolvedEvent:
    event: NarrationEvent
    fixture_id: str | None
    fixture_type: str | None
    confidence: str
    reason: str
    rule: str
    best_gaze: float | None = None
    runner_up_gaze: float | None = None
    candidate_scores: List[Tuple[str, float]] = field(default_factory=list)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _alignment(camera: Vec3, gaze: Vec3, target: Vec3) -> float:
    cam_to_target = _sub(target, camera)
    d = _norm(cam_to_target)
    g = _norm(gaze)
    if d < 1e-6 or g < 1e-6:
        return -1.0
    return _dot(cam_to_target, gaze) / (d * g)


def _distance(a: Vec3, b: Vec3) -> float:
    return _norm(_sub(a, b))


# Distance (in metres) at which proximity score drops to e**-1 ~= 0.37.
# Kitchens are small, interactions happen well within 1.5 m.
_PROXIMITY_DECAY_M: float = 1.5


def _proximity_score(camera: Vec3, target: Vec3) -> float:
    d = _distance(camera, target)
    return math.exp(-d / _PROXIMITY_DECAY_M)


def _mean_interaction_score(
    frames: Sequence[FrameInfo],
    start_t: float,
    end_t: float,
    fixture: Fixture,
) -> float | None:
    """Mean of (gaze alignment * proximity) over the event's frames.

    Combining gaze and proximity prevents false matches where the person only
    glances at a distant fixture. Both terms live in [0, 1]; the product makes
    a high score require *both* signals.
    """
    if fixture.centroid is None:
        return None

    start_f = config.time_to_frame(start_t)
    end_f = config.time_to_frame(end_t)
    total = 0.0
    count = 0

    for frame_i in range(start_f, end_f + 1):
        if frame_i < 0 or frame_i >= len(frames):
            continue
        frame = frames[frame_i]
        if frame.camera_position is None or frame.gaze_direction is None:
            continue
        gaze = _alignment(frame.camera_position, frame.gaze_direction, fixture.centroid)
        # Gaze alignment is in [-1, 1]; clamp to [0, 1] so it multiplies cleanly.
        gaze = max(0.0, gaze)
        prox = _proximity_score(frame.camera_position, fixture.centroid)
        total += gaze * prox
        count += 1

    if count == 0:
        return None
    return total / count


def _confidence_from_scores(best: float, runner_up: float) -> str:
    # Score is in [0, 1] (gaze * proximity, averaged over frames).
    if best >= 0.55 and (best - runner_up) >= 0.12:
        return "high"
    if best >= 0.30:
        return "medium"
    if best >= 0.10:
        return "low"
    return "very_low"


def resolve_event(
    event: NarrationEvent,
    kitchen: Dict[str, Fixture],
    framewise: Sequence[FrameInfo],
) -> ResolvedEvent:
    candidates: Dict[str, Fixture] = {}

    for fixture_type in event.candidate_types:
        matches = {
            fixture_id: fixture
            for fixture_id, fixture in kitchen.items()
            if fixture.fixture_type == fixture_type
        }
        if matches:
            candidates = matches
            break

    if not candidates:
        return ResolvedEvent(
            event=event,
            fixture_id=None,
            fixture_type=None,
            confidence="none",
            reason="no fixture of candidate type in kitchen",
            rule="unresolved",
        )

    if len(candidates) == 1:
        fixture_id, fixture = next(iter(candidates.items()))
        return ResolvedEvent(
            event=event,
            fixture_id=fixture_id,
            fixture_type=fixture.fixture_type,
            confidence="very_high",
            reason="single fixture of this type",
            rule="rule1_unique",
        )

    scores: list[tuple[str, float]] = []
    for fixture_id, fixture in candidates.items():
        score = _mean_interaction_score(framewise, event.start_time, event.end_time, fixture)
        if score is not None:
            scores.append((fixture_id, score))

    if not scores:
        fallback_id = sorted(candidates.keys())[0]
        return ResolvedEvent(
            event=event,
            fixture_id=fallback_id,
            fixture_type=candidates[fallback_id].fixture_type,
            confidence="very_low",
            reason="multiple fixtures and no usable gaze data; picked deterministic fallback",
            rule="rule2_gaze",
            candidate_scores=[(fixture_id, float("nan")) for fixture_id in sorted(candidates.keys())],
        )

    scores.sort(key=lambda item: item[1], reverse=True)
    best_id, best_score = scores[0]
    runner_up = scores[1][1] if len(scores) > 1 else -1.0

    return ResolvedEvent(
        event=event,
        fixture_id=best_id,
        fixture_type=candidates[best_id].fixture_type,
        confidence=_confidence_from_scores(best_score, runner_up),
        reason=f"best gaze score={best_score:.2f}, runner-up={runner_up:.2f}",
        rule="rule2_gaze_and_proximity",
        best_gaze=best_score,
        runner_up_gaze=runner_up,
        candidate_scores=scores,
    )
