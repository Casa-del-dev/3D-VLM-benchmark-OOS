"""Stage 4: verify ``potentially_visible_inside_open_fixture`` intervals
with ROI object detection.

For every coarse interval that stage 3 labelled
``potentially_visible_inside_open_fixture``, this samples a handful of frames
(dense at short intervals, quartile-spaced at long ones), projects the object
into each frame, runs Grounding DINO inside a tight ROI around that
projection, and upgrades the interval to:

  * ``observed_visible_in_open_fixture``     -- detected in at least one sample
  * ``observed_not_visible_in_open_fixture`` -- never detected across all samples

All other statuses from stage 3 (in_view, out_of_view, in_motion,
geometrically_occluded, occluded_inside_closed_fixture) pass through
unchanged.

Output statuses
---------------
  in_view, out_of_view, in_motion, geometrically_occluded,
  occluded_inside_closed_fixture, observed_visible_in_open_fixture,
  observed_not_visible_in_open_fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.visibility_track.combine import (  # noqa: E402
        CoarseInterval,
        POTENTIALLY_VISIBLE_INSIDE_OPEN_FIXTURE,
        coarse_from_dict,
    )
    from scripts.visibility_track.common import (  # noqa: E402
        DetectionConfig,
        PipelineConfig,
        load_config,
        read_jsonl,
        write_jsonl,
    )
    from scripts.visibility_track.in_view_track.in_view_determination import VideoCache, determine_in_view_objects  # noqa: E402
else:
    from .combine import CoarseInterval, POTENTIALLY_VISIBLE_INSIDE_OPEN_FIXTURE, coarse_from_dict
    from .common import DetectionConfig, PipelineConfig, load_config, read_jsonl, write_jsonl
    from .in_view_track.in_view_determination import VideoCache, determine_in_view_objects


DEFAULT_CONFIG = Path(__file__).resolve().parent / "visibility_track_config.yaml"

# --- Status constants --------------------------------------------------------

OBSERVED_VISIBLE = "observed_visible_in_open_fixture"
OBSERVED_NOT_VISIBLE = "observed_not_visible_in_open_fixture"


# --- Data --------------------------------------------------------------------

@dataclass
class RefinedInterval:
    video_id: str
    assoc_id: str
    object_name: str
    start_sec: float
    end_sec: float
    status: str
    reason: str
    fixture: str | None = None
    fixture_confidence: str | None = None
    n_visible: int = 0
    n_partial: int = 0
    n_tested: int = 0
    max_score: float | None = None
    detection_samples: List[dict] = field(default_factory=list)


# --- Video frame reader ------------------------------------------------------

class VideoFrameReader:
    def __init__(self, video_path: Path, fps: float) -> None:
        import cv2  # type: ignore[import-not-found]

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cv2 = cv2
        self._fps = fps
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

    def read_at(self, time_sec: float):
        frame_idx = int(round(time_sec * self._fps))
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoFrameReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# --- Grounding DINO estimator or Detic estimator ---------------------------------------------

def _build_estimator(det_cfg: DetectionConfig):
    backend = getattr(det_cfg, "backend", "groundingdino").lower()

    if backend in {"groundingdino", "grounding_dino", "gdino"}:
        if __package__ in (None, ""):
            from scripts.visibility_track.object_detection.groundingdino_roi_visibility import (
                GroundingDINODetector,
                ROIGroundingDINOVisibilityEstimator,
            )
        else:
            from .object_detection.groundingdino_roi_visibility import (
                GroundingDINODetector,
                ROIGroundingDINOVisibilityEstimator,
            )

        detector = GroundingDINODetector(model_id=det_cfg.model_id)
        return ROIGroundingDINOVisibilityEstimator(
            detector=detector,
            roi_scale=det_cfg.roi_scale,
            box_threshold=det_cfg.box_threshold,
            text_threshold=det_cfg.text_threshold,
            visible_threshold=det_cfg.visible_threshold,
            partial_threshold=det_cfg.partial_threshold,
        )

    if backend == "detic":
        if __package__ in (None, ""):
            from scripts.visibility_track.object_detection.detic_detector import DeticDetector
            from scripts.visibility_track.object_detection.roi_visibility import ROIVisibilityEstimator
        else:
            from .object_detection.detic_detector import DeticDetector
            from .object_detection.roi_visibility import ROIVisibilityEstimator

        if not det_cfg.detic_root:
            raise ValueError("object_detection.detic_root must be set when backend='detic'")
        if not det_cfg.config_file:
            raise ValueError("object_detection.config_file must be set when backend='detic'")
        if not det_cfg.weights:
            raise ValueError("object_detection.weights must be set when backend='detic'")

        detector = DeticDetector(
            detic_root=det_cfg.detic_root,
            config_file=det_cfg.config_file,
            weights=det_cfg.weights,
            vocabulary=getattr(det_cfg, "vocabulary", "custom"),
            custom_vocabulary=["object"],
            confidence_threshold=det_cfg.box_threshold,
            device=getattr(det_cfg, "device", "cpu"),
        )

        return ROIVisibilityEstimator(
            detector=detector,
            detector_name="Detic",
            roi_scale=det_cfg.roi_scale,
            box_threshold=det_cfg.box_threshold,
            visible_threshold=det_cfg.visible_threshold,
            partial_threshold=det_cfg.partial_threshold,
        )

    raise ValueError(f"Unsupported object detection backend: {backend}")


# --- Detection sampling strategy ---------------------------------------------

def _detection_times(start_sec: float, end_sec: float, sampling_fps: float) -> list[float]:
    if sampling_fps <= 0 or end_sec < start_sec:
        return []

    step = 1.0 / sampling_fps
    duration = end_sec - start_sec
    mid = round((start_sec + end_sec) / 2.0, 6)

    if duration <= step:
        return [mid]

    if duration <= 4.0 * step:
        return sorted({round(start_sec, 6), mid, round(end_sec, 6)})

    q1 = round(start_sec + 0.25 * duration, 6)
    q3 = round(start_sec + 0.75 * duration, 6)
    return sorted({round(start_sec, 6), q1, mid, q3, round(end_sec, 6)})


# --- Single-sample detection -------------------------------------------------

def _sample_detection(
    estimator,
    frame_reader: VideoFrameReader,
    time_sec: float,
    video_id: str,
    annotations_root: Path,
    intermediate_root_name: str,
    video_fps: float,
    assoc_id: str,
    object_name: str,
    det_cfg: DetectionConfig,
    cache: "VideoCache | None" = None,
) -> dict:
    states = determine_in_view_objects(
        video_id=video_id,
        time_sec=time_sec,
        annotations_root=annotations_root,
        fps=video_fps,
        intermediate_root=intermediate_root_name,
        cache=cache,
    )
    state = next((row for row in states if row.assoc_id == assoc_id), None)
    if (
        state is None
        or state.status != "ok"
        or state.projected_pixel is None
        or not state.in_view
    ):
        return {
            "time_sec": time_sec,
            "label": "skipped",
            "score": None,
            "reason": "not a valid in-view sample",
            "mask_bbox": None,
            "expected_box_size_px": None,
            "roi_bbox_xyxy": None,
        }

    frame = frame_reader.read_at(time_sec)
    if frame is None:
        return {
            "time_sec": time_sec,
            "label": "skipped",
            "score": None,
            "reason": "frame read failed",
            "mask_bbox": None,
            "expected_box_size_px": None,
            "roi_bbox_xyxy": None,
        }

    if state.mask_bbox is not None:
        x1, y1, x2, y2 = state.mask_bbox
        expected_size = (max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1)))
        mask_bbox = [float(x1), float(y1), float(x2), float(y2)]
    else:
        d = float(det_cfg.default_expected_size_px)
        expected_size = (d, d)
        mask_bbox = None

    result, _ = estimator.estimate(
        image_bgr=frame,
        projected_uv=tuple(state.projected_pixel),
        text_prompt=object_name,
        expected_box_size_px=expected_size,
        uncertainty_px=det_cfg.uncertainty_px,
    )

    return {
        "time_sec": time_sec,
        "label": result.label,
        "score": float(result.visibility_score),
        "reason": result.reason,
        "projected_uv": list(state.projected_pixel),
        "mask_bbox": mask_bbox,
        "expected_box_size_px": [float(expected_size[0]), float(expected_size[1])],
        "roi_bbox_xyxy": [result.roi.x1, result.roi.y1, result.roi.x2, result.roi.y2],
        "detections": [
            {
                "bbox_xyxy": list(det.bbox_xyxy),
                "confidence": float(det.confidence),
                "phrase": det.phrase,
            }
            for det in result.detections
        ],
    }


# --- Interval refinement -----------------------------------------------------

def _passthrough(interval: CoarseInterval) -> RefinedInterval:
    return RefinedInterval(
        video_id=interval.video_id,
        assoc_id=interval.assoc_id,
        object_name=interval.object_name,
        start_sec=interval.start_sec,
        end_sec=interval.end_sec,
        status=interval.status,
        reason=interval.reason,
        fixture=interval.fixture,
        fixture_confidence=interval.fixture_confidence,
    )


def _refine_candidate(
    interval: CoarseInterval,
    estimator,
    frame_reader: VideoFrameReader,
    video_id: str,
    annotations_root: Path,
    intermediate_root_name: str,
    video_fps: float,
    det_sampling_fps: float,
    det_cfg: DetectionConfig,
    cache: "VideoCache | None" = None,
) -> RefinedInterval:
    times = _detection_times(interval.start_sec, interval.end_sec, det_sampling_fps)
    samples = [
        _sample_detection(
            estimator=estimator,
            frame_reader=frame_reader,
            time_sec=time_sec,
            video_id=video_id,
            annotations_root=annotations_root,
            intermediate_root_name=intermediate_root_name,
            video_fps=video_fps,
            assoc_id=interval.assoc_id,
            object_name=interval.object_name,
            det_cfg=det_cfg,
            cache=cache,
        )
        for time_sec in times
    ]

    tested = [row for row in samples if row["label"] != "skipped"]
    n_visible = sum(1 for row in tested if row["label"] == "visible")
    n_partial = sum(1 for row in tested if row["label"] == "partially_visible")
    n_tested = len(tested)
    scores = [row["score"] for row in tested if row.get("score") is not None]
    max_score = max(scores) if scores else None

    n_positive = n_visible + (n_partial if det_cfg.count_partial_as_positive else 0)
    min_required = max(1, int(det_cfg.min_positive_samples))

    if n_tested == 0:
        status = OBSERVED_NOT_VISIBLE
        reason = "no valid detection samples"
    elif n_positive >= min_required:
        status = OBSERVED_VISIBLE
        reason = f"detected in {n_positive}/{n_tested} samples (min {min_required})"
    else:
        status = OBSERVED_NOT_VISIBLE
        reason = f"only {n_positive}/{n_tested} positive samples (min {min_required})"

    return RefinedInterval(
        video_id=interval.video_id,
        assoc_id=interval.assoc_id,
        object_name=interval.object_name,
        start_sec=interval.start_sec,
        end_sec=interval.end_sec,
        status=status,
        reason=reason,
        fixture=interval.fixture,
        fixture_confidence=interval.fixture_confidence,
        n_visible=n_visible,
        n_partial=n_partial,
        n_tested=n_tested,
        max_score=max_score,
        detection_samples=samples,
    )


# --- Full video refinement ---------------------------------------------------

def refine_with_detection(
    coarse: list[CoarseInterval],
    video_id: str,
    video_path: Path,
    annotations_root: Path,
    intermediate_root_name: str,
    video_fps: float,
    det_sampling_fps: float,
    det_cfg: DetectionConfig,
) -> list[RefinedInterval]:
    target_status = POTENTIALLY_VISIBLE_INSIDE_OPEN_FIXTURE

    estimator = _build_estimator(det_cfg)
    cache = VideoCache.build(video_id, annotations_root, intermediate_root_name)
    output: list[RefinedInterval] = []

    detection_intervals = [iv for iv in coarse if iv.status == target_status]
    total_samples = sum(
        len(_detection_times(iv.start_sec, iv.end_sec, det_sampling_fps))
        for iv in detection_intervals
    )

    with VideoFrameReader(video_path, video_fps) as frame_reader:
        with tqdm(total=total_samples, desc=f"{video_id} detection", unit="sample") as pbar:
            for interval in coarse:
                if interval.status != target_status:
                    output.append(_passthrough(interval))
                    continue

                pbar.set_postfix_str(interval.object_name)
                output.append(
                    _refine_candidate(
                        interval=interval,
                        estimator=estimator,
                        frame_reader=frame_reader,
                        video_id=video_id,
                        annotations_root=annotations_root,
                        intermediate_root_name=intermediate_root_name,
                        video_fps=video_fps,
                        det_sampling_fps=det_sampling_fps,
                        det_cfg=det_cfg,
                        cache=cache,
                    )
                )
                pbar.update(len(_detection_times(interval.start_sec, interval.end_sec, det_sampling_fps)))

    output.sort(key=lambda row: (row.assoc_id, row.start_sec))
    return output


# --- Serialisation -----------------------------------------------------------

def refined_to_dict(interval: RefinedInterval) -> dict:
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
        "n_visible": interval.n_visible,
        "n_partial": interval.n_partial,
        "n_tested": interval.n_tested,
        "max_score": interval.max_score,
        "detection_samples": interval.detection_samples,
    }


def _load_coarse(path: Path) -> list[CoarseInterval]:
    return [coarse_from_dict(row) for row in read_jsonl(path)]


def _write_summary(path: Path, refined: list[RefinedInterval]) -> None:
    per_object: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    names: Dict[str, str] = {}

    for row in refined:
        per_object[row.assoc_id][row.status] += max(0.0, row.end_sec - row.start_sec)
        names[row.assoc_id] = row.object_name

    summary = {
        "per_object": [
            {
                "assoc_id": assoc_id,
                "object_name": names[assoc_id],
                "duration_by_status": {status: round(duration, 3) for status, duration in durations.items()},
            }
            for assoc_id, durations in sorted(per_object.items(), key=lambda item: item[0])
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


# --- Stage entry point -------------------------------------------------------

def run_stage(cfg: PipelineConfig, video_ids: list[str]) -> None:
    print(f"[stage 4] refining intervals for {len(video_ids)} video(s)")
    for video_id in tqdm(video_ids, desc="Stage 4 videos", unit="video"):
        out_dir = cfg.video_output_dir(video_id)
        coarse_path = out_dir / "coarse_visibility_track.jsonl"
        if not coarse_path.exists():
            raise FileNotFoundError(f"Missing {coarse_path}. Run stage 3 first.")

        coarse = _load_coarse(coarse_path)
        refined = refine_with_detection(
            coarse=coarse,
            video_id=video_id,
            video_path=cfg.video_file(video_id),
            annotations_root=cfg.annotations_root,
            intermediate_root_name=str(cfg.intermediate_data_root),
            video_fps=cfg.video_fps,
            det_sampling_fps=cfg.object_detection_sampling_fps,
            det_cfg=cfg.detection,
        )

        out_path = out_dir / "visibility_track.jsonl"
        write_jsonl(out_path, (refined_to_dict(interval) for interval in refined))
        _write_summary(out_dir / "visibility_track_summary.json", refined)
        print(f"[stage 4] {video_id} -> {out_path}")


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
