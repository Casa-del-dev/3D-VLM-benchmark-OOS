"""Config loading, path resolution, and lightweight JSONL I/O helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    enabled: bool = True
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    roi_scale: float = 2.0
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    visible_threshold: float = 0.5
    partial_threshold: float = 0.3
    uncertainty_px: int = 40
    default_expected_size_px: int = 120

    # Interval verdict (used by detection_refinement._refine_candidate).
    min_positive_samples: int = 1
    count_partial_as_positive: bool = False

    backend: str = "groundingdino"  # "groundingdino" or "detic"

    # Detic-specific
    detic_root: str | None = None
    config_file: str | None = None
    weights: str | None = None
    vocabulary: str = "custom"
    device: str = "cpu"


@dataclass
class PipelineConfig:
    annotations_root: Path
    data_root: Path
    intermediate_data_root: Path
    videos: list[str]
    participants: list[str]
    in_view_sampling_fps: float
    object_detection_sampling_fps: float
    video_fps: float
    detection: DetectionConfig
    output_root: Path
    geometric_occlusion_enabled: bool = True
    geometric_occlusion_tolerance: float = 0.05
    # When true, cast 9 rays per sample (centroid + 4 bbox corners + 4
    # bbox edge midpoints, back-projected through the mask frame's
    # FISHEYE624 camera). When false (or when the 2D bbox / mask frame
    # is missing), fall back to a single centroid ray.
    geometric_occlusion_multi_ray: bool = True
    # Fraction-of-blocked-rays threshold above which the sample is reported
    # as geometrically_occluded.
    geometric_occlusion_threshold: float = 0.5
    random_seed: int = 42
    config_path: Path | None = None

    # Convenient derived paths ------------------------------------------------
    @property
    def narrations_pkl(self) -> Path:
        return self.annotations_root / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl"

    @property
    def mask_info_json(self) -> Path:
        return self.annotations_root / "scene-and-object-movements" / "mask_info.json"

    @property
    def assoc_info_json(self) -> Path:
        return self.annotations_root / "scene-and-object-movements" / "assoc_info.json"

    @property
    def sounds_csv(self) -> Path:
        return self.annotations_root / "audio-annotations" / "HD_EPIC_Sounds.csv"

    def framewise_path(self, video_id: str) -> Path:
        participant = video_id.split("-", 1)[0]
        return self.intermediate_data_root / participant / video_id / "framewise_info.jsonl"

    def video_file(self, video_id: str) -> Path:
        participant = video_id.split("-", 1)[0]
        return self.data_root / "HD-EPIC" / "Videos" / participant / f"{video_id}.mp4"

    def video_output_dir(self, video_id: str) -> Path:
        return self.output_root / video_id

    def videos_for_participant(self, participant: str) -> list[str]:
        """Return all video IDs for *participant* by scanning intermediate_data_root."""
        participant_dir = self.intermediate_data_root / participant
        if not participant_dir.is_dir():
            raise FileNotFoundError(f"No intermediate data directory for participant: {participant_dir}")
        return sorted(
            p.name
            for p in participant_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _resolve(base: Path, p: str | os.PathLike) -> Path:
    """Resolve a path relative to `base` if it's relative, otherwise as-is."""
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def load_config(config_path: str | os.PathLike) -> PipelineConfig:
    """Parse the visibility-track YAML into a PipelineConfig.

    Relative paths are resolved against the config file's parent directory,
    so the file can be moved without updating the pipeline code.
    """
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    base = config_path.parent
    annotations_root = _resolve(base, raw["annotations_root"])
    data_root = _resolve(base, raw["data_root"])
    intermediate_data_root = _resolve(base, raw["intermediate_data_root"])
    output_root = _resolve(base, raw.get("output_root", "./outputs"))

    inputs = raw.get("inputs", {}) or {}
    det_raw = raw.get("object_detection", {}) or {}
    detection = DetectionConfig(
        enabled=bool(det_raw.get("enabled", True)),
        model_id=str(det_raw.get("model_id", "IDEA-Research/grounding-dino-tiny")),
        roi_scale=float(det_raw.get("roi_scale", 2.0)),
        box_threshold=float(det_raw.get("box_threshold", 0.25)),
        text_threshold=float(det_raw.get("text_threshold", 0.25)),
        visible_threshold=float(det_raw.get("visible_threshold", 0.5)),
        partial_threshold=float(det_raw.get("partial_threshold", 0.3)),
        uncertainty_px=int(det_raw.get("uncertainty_px", 40)),
        default_expected_size_px=int(det_raw.get("default_expected_size_px", 120)),

        min_positive_samples=int(det_raw.get("min_positive_samples", 1)),
        count_partial_as_positive=bool(det_raw.get("count_partial_as_positive", False)),

        backend=det_raw.get("backend", "groundingdino"),
        detic_root=det_raw.get("detic_root"),
        config_file=det_raw.get("config_file"),
        weights=det_raw.get("weights"),
        vocabulary=det_raw.get("vocabulary", "custom"),
        device=det_raw.get("device", "cpu"),
    )

    geo_raw = raw.get("geometric_occlusion", {}) or {}
    cfg = PipelineConfig(
        annotations_root=annotations_root,
        data_root=data_root,
        intermediate_data_root=intermediate_data_root,
        videos=list(inputs.get("videos", []) or []),
        participants=list(inputs.get("participants", []) or []),
        in_view_sampling_fps=float(raw.get("in_view_sampling_fps", 1.0)),
        object_detection_sampling_fps=float(raw.get("object_detection_sampling_fps", 0.2)),
        video_fps=float(raw.get("video_fps", 30.0)),
        detection=detection,
        output_root=output_root,
        geometric_occlusion_enabled=bool(geo_raw.get("enabled", True)),
        geometric_occlusion_tolerance=float(geo_raw.get("tolerance_m", 0.05)),
        geometric_occlusion_multi_ray=bool(geo_raw.get("multi_ray", True)),
        geometric_occlusion_threshold=float(geo_raw.get("occlusion_threshold", 0.5)),
        random_seed=int(raw.get("random_seed", 42)),
        config_path=config_path,
    )
    return cfg


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
