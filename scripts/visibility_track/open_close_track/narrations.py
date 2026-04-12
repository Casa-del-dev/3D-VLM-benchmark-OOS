"""Extract open/close narrations from HD_EPIC_Narrations.pkl.

Each extracted event is a single (verb, fixture-type-candidate, time) sample.
For a narration like "Open the upper cupboard by holding the handle", only the
nouns paired with an `open`/`close` verb class (ids 3 and 4) are kept, so
`hold/handle` is ignored even though both are on the same row.

Optionally, event timing can be snapped to the nearest "open / close" sound in
HD_EPIC_Sounds, which gives a much tighter fixture-state timestamp than the
narration's start/end window.

The output is downstream fodder for the resolver, which attaches a concrete
fixture instance (e.g. `P01_cupboard.009`).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Verb class ids (see HD_EPIC_verb_classes.csv)
# ---------------------------------------------------------------------------

OPEN_VERB_CLASS_ID: int = 3
CLOSE_VERB_CLASS_ID: int = 4

# Sounds CSV class name that covers both fixture open and close events.
OPEN_CLOSE_SOUND_CLASS: str = "open / close"
# Max gap between a narration window and a matching open/close sound (seconds).
SOUND_SNAP_TOLERANCE_SEC: float = 1.5


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NarrationEvent:
    """One open/close action mentioned in a narration."""
    unique_narration_id: str
    video_id: str
    participant_id: str
    verb: str                     # "open" or "close" (canonicalized)
    start_time: float             # seconds
    end_time: float               # seconds
    narration: str
    nouns: tuple[str, ...]        # raw nouns from the annotation
    candidate_types: tuple[str, ...]  # fixture types inferred from nouns


# ---------------------------------------------------------------------------
# Noun -> fixture type mapping
# ---------------------------------------------------------------------------

# Map of substrings (lowercase, on noun phrase) -> tuple of candidate fixture
# types in priority order. The first existing type in the video catalog wins,
# but the full list is kept so downstream code can fall back gracefully.
#
# Order matters: earlier, more specific needles are matched first.
_NOUN_TO_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("top microwave", ("top_microwave", "microwave")),
    ("top cupboard", ("top_cupboard", "cupboard")),
    ("top drawer", ("top_drawer", "drawer")),
    ("top fridge", ("top_fridge", "fridge", "fridgefreezer")),
    ("top storage", ("top_storage", "storage")),
    ("fridge freezer", ("fridgefreezer", "fridge", "freezer")),
    ("fridge-freezer", ("fridgefreezer", "fridge", "freezer")),
    ("washing machine", ("washingmachine",)),
    ("dishwasher", ("dishwasher",)),
    ("microwave", ("microwave", "top_microwave")),
    ("freezer", ("freezer", "fridgefreezer")),
    ("fridge", ("fridge", "fridgefreezer", "top_fridge")),
    ("cupboard", ("cupboard", "top_cupboard")),
    ("cabinet", ("cupboard", "top_cupboard")),
    ("drawer", ("drawer", "top_drawer")),
    ("oven", ("oven",)),
    ("bin", ("bin",)),
    ("storage", ("storage", "top_storage")),
)


def infer_fixture_types(nouns: Sequence[str]) -> tuple[str, ...]:
    """Return unique fixture types hinted at by any noun phrase, priority-ordered."""
    seen: list[str] = []
    for raw in nouns:
        phrase = raw.lower()
        for needle, ftypes in _NOUN_TO_TYPES:
            if needle in phrase:
                for ft in ftypes:
                    if ft in config.OPENABLE_FIXTURE_TYPES and ft not in seen:
                        seen.append(ft)
                break
    return tuple(seen)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def load_narrations(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f)


def _pick_open_close_nouns(row) -> tuple[str | None, tuple[str, ...]]:
    """Return (verb, nouns) where verb is 'open'/'close' from verb-class ids.

    Uses `pair_classes` so only nouns actually paired with an open/close verb
    are returned — e.g. for "Open the cupboard by holding the handle", the
    `hold`/`handle` pair is dropped. Open wins over close if both are present
    (rare, but possible in multi-action rows).
    """
    pair_classes = getattr(row, "pair_classes", None) or []
    pairs = getattr(row, "pairs", None) or []
    if len(pair_classes) != len(pairs):
        return None, ()

    open_nouns: list[str] = []
    close_nouns: list[str] = []
    for (verb_cls, _noun_cls), (_verb, noun) in zip(pair_classes, pairs):
        if verb_cls == OPEN_VERB_CLASS_ID:
            open_nouns.append(str(noun))
        elif verb_cls == CLOSE_VERB_CLASS_ID:
            close_nouns.append(str(noun))

    if open_nouns:
        return "open", tuple(open_nouns)
    if close_nouns:
        return "close", tuple(close_nouns)
    return None, ()


def load_sounds(path: Path) -> pd.DataFrame:
    """Load HD_EPIC_Sounds.csv as a DataFrame indexed by video_id."""
    return pd.read_csv(path)


def _build_sound_index(sounds: pd.DataFrame) -> dict[str, list[tuple[float, float]]]:
    """Return {video_id: [(start, stop), ...]} for open/close sound events."""
    mask = sounds["class"].str.strip().str.lower() == OPEN_CLOSE_SOUND_CLASS
    filtered = sounds.loc[mask, ["video_id", "start_timestamp", "stop_timestamp"]]

    def _to_sec(ts: str) -> float:
        h, m, s = str(ts).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    index: dict[str, list[tuple[float, float]]] = {}
    for video_id, group in filtered.groupby("video_id"):
        events = sorted(
            (_to_sec(start), _to_sec(stop))
            for start, stop in zip(group["start_timestamp"], group["stop_timestamp"])
        )
        index[str(video_id)] = events
    return index


def _snap_to_sound(
    video_id: str,
    start_sec: float,
    end_sec: float,
    sound_index: dict[str, list[tuple[float, float]]],
    tolerance: float,
) -> tuple[float, float] | None:
    """Pick the closest open/close sound whose midpoint is within `tolerance`."""
    events = sound_index.get(video_id)
    if not events:
        return None

    window_mid = 0.5 * (start_sec + end_sec)
    best = min(events, key=lambda ev: abs(0.5 * (ev[0] + ev[1]) - window_mid))
    best_mid = 0.5 * (best[0] + best[1])
    if abs(best_mid - window_mid) > tolerance:
        return None
    return best


def extract_events(
    narrations_path: Path,
    df: pd.DataFrame | None = None,
    sounds_path: Path | None = None,
) -> List[NarrationEvent]:
    """Return all narration-level open/close events, chronologically per video.

    If `sounds_path` is provided, each event is snapped to the nearest
    `open / close` sound in HD_EPIC_Sounds.csv when one is within
    `SOUND_SNAP_TOLERANCE_SEC` of the narration window.
    """
    if df is None:
        df = load_narrations(narrations_path)

    sound_index: dict[str, list[tuple[float, float]]] = {}
    if sounds_path is not None and Path(sounds_path).exists():
        sound_index = _build_sound_index(load_sounds(Path(sounds_path)))

    events: list[NarrationEvent] = []
    for row in df.itertuples(index=False):
        verb, open_close_nouns = _pick_open_close_nouns(row)
        if verb is None:
            continue
        candidate_types = infer_fixture_types(open_close_nouns)
        if not candidate_types:
            # No noun hints towards a known openable fixture — skip, since we
            # cannot resolve it to a fixture instance downstream.
            continue

        start_time = float(row.start_timestamp)
        end_time = float(row.end_timestamp)
        if sound_index:
            snapped = _snap_to_sound(
                video_id=row.video_id,
                start_sec=start_time,
                end_sec=end_time,
                sound_index=sound_index,
                tolerance=SOUND_SNAP_TOLERANCE_SEC,
            )
            if snapped is not None:
                start_time, end_time = snapped

        events.append(
            NarrationEvent(
                unique_narration_id=row.unique_narration_id,
                video_id=row.video_id,
                participant_id=row.participant_id,
                verb=verb,
                start_time=start_time,
                end_time=end_time,
                narration=row.narration,
                nouns=tuple(open_close_nouns),
                candidate_types=candidate_types,
            )
        )
    events.sort(key=lambda e: (e.video_id, e.start_time))
    return events
