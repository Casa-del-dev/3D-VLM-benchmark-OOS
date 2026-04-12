"""Build the global catalog of openable fixtures from mask_info.json.

Each fixture id in mask_info (e.g. `P01_cupboard.009`) is aggregated across
*all* masks that reference it, producing:
  * the fixture type (cupboard, drawer, ...)
  * a mean 3D location — used later for gaze/proximity matching
  * n_instances — how many sibling fixtures of the same type live in the
    same kitchen (drives the resolver's "unique fixture -> very_high" rule).

Fixtures are participant-scoped because fixture ids are already prefixed by
participant, and the resolver reasons at kitchen (participant) granularity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from . import config


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Fixture:
    fixture_id: str                                     # e.g. P01_cupboard.009
    fixture_type: str                                   # cupboard, drawer, ...
    xyz: Tuple[float, float, float] | None = None       # mean 3D location
    n_instances: int = 0                                # fixtures of this type in this kitchen

    # Compatibility alias — `centroid` is used by the resolver's gaze scoring.
    @property
    def centroid(self) -> Tuple[float, float, float] | None:
        return self.xyz


# Per-kitchen catalog: participant_id -> fixture_id -> Fixture.
KitchenCatalog = Dict[str, Dict[str, Fixture]]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_mask_info(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _fixture_type(fixture_id: str) -> str:
    """`P01_cupboard.009` -> `cupboard`. Unknown schemes fall through unchanged."""
    base = fixture_id.split("_", 1)[1] if "_" in fixture_id else fixture_id
    return base.split(".", 1)[0]


def _participant_of(video_id: str) -> str:
    return video_id.split("-", 1)[0]


# ---------------------------------------------------------------------------
# Catalog construction
# ---------------------------------------------------------------------------

def build_kitchen_catalog(
    mask_info_path: Path,
    mask_info: dict | None = None,
) -> KitchenCatalog:
    """Aggregate mask_info into a per-participant catalog of openable fixtures.

    Each fixture stores its mean 3D location (averaged across all observing
    masks) and `n_instances` — the number of sibling fixtures of the same type
    in the same kitchen (e.g. every cupboard in a kitchen with 5 cupboards has
    `n_instances=5`). The resolver uses `n_instances == 1` as its strongest
    confidence signal.
    """
    if mask_info is None:
        mask_info = load_mask_info(mask_info_path)

    # Pass 1: collect xyz observations per fixture_id.
    sums: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    catalog: KitchenCatalog = {}

    for video_id, masks in mask_info.items():
        participant = _participant_of(video_id)
        kitchen = catalog.setdefault(participant, {})
        p_sums = sums.setdefault(participant, {})
        p_counts = counts.setdefault(participant, {})

        for mask in masks.values():
            fixture_id = mask.get("fixture")
            if not fixture_id:
                continue
            ftype = _fixture_type(fixture_id)
            if ftype not in config.OPENABLE_FIXTURE_TYPES:
                continue

            if fixture_id not in kitchen:
                kitchen[fixture_id] = Fixture(fixture_id=fixture_id, fixture_type=ftype)
                p_sums[fixture_id] = (0.0, 0.0, 0.0)
                p_counts[fixture_id] = 0

            xyz = mask.get("3d_location")
            if xyz is not None:
                sx, sy, sz = p_sums[fixture_id]
                x, y, z = xyz
                p_sums[fixture_id] = (sx + x, sy + y, sz + z)
                p_counts[fixture_id] += 1

    # Pass 2: finalize mean xyz and fill in per-type n_instances.
    for participant, kitchen in catalog.items():
        type_counts: Dict[str, int] = {}
        for fx in kitchen.values():
            type_counts[fx.fixture_type] = type_counts.get(fx.fixture_type, 0) + 1

        for fixture_id, fx in kitchen.items():
            n = counts[participant][fixture_id]
            if n > 0:
                sx, sy, sz = sums[participant][fixture_id]
                fx.xyz = (sx / n, sy / n, sz / n)
            fx.n_instances = type_counts[fx.fixture_type]

    return catalog
