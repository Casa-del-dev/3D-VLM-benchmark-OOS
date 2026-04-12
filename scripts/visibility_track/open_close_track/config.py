"""Domain constants and frame-time helpers for the open/close track stage.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Fixture types (suffixless) that can be meaningfully "opened" or "closed".
OPENABLE_FIXTURE_TYPES: frozenset[str] = frozenset({
    "cupboard",
    "top_cupboard",
    "drawer",
    "top_drawer",
    "fridge",
    "top_fridge",
    "fridgefreezer",
    "freezer",
    "oven",
    "dishwasher",
    "microwave",
    "top_microwave",
    "washingmachine",
    "bin",
    "storage",
    "top_storage",
})

# Video framerate for time <-> frame conversion. HD-EPIC is 30 fps.
VIDEO_FPS: float = 30.0


def time_to_frame(t_seconds: float) -> int:
    return int(round(t_seconds * VIDEO_FPS))


def frame_to_time(frame_index: int) -> float:
    return frame_index / VIDEO_FPS
