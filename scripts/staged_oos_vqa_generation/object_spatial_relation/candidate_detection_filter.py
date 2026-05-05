from __future__ import annotations

import sys
from dataclasses import dataclass, replace, is_dataclass, asdict
from pathlib import Path
from typing import Any
from types import SimpleNamespace

# candidate_detection_filter.py is in:
# scripts/staged_oos_vqa_generation/object_spatial_relation/
# repo root is 3 parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visibility_track.detection_refinement import (
    VideoFrameReader,
    _build_estimator,
)
from scripts.visibility_track.in_view_track.in_view_determination import (
    VideoCache,
    determine_in_view_objects,
)


@dataclass
class CandidateDetectionDecision:
    accepted: bool
    metadata: dict[str, Any]


class CandidateDetectionFilter:
    def __init__(
        self,
        *,
        video_id: str,
        video_path: Path,
        annotations_root: Path,
        intermediate_root: str,
        video_fps: float,
        det_cfg,
        accept_labels: set[str] | None = None,
        custom_vocabulary: list[str] | None = None,
    ) -> None:
        if isinstance(det_cfg, dict):
            det_cfg = SimpleNamespace(**det_cfg)
        elif is_dataclass(det_cfg):
            # Convert frozen DetectionFilterConfig into a mutable namespace.
            # This also allows adding FIction-Detic-specific runtime fields
            # such as custom_vocabulary even if they are not declared in the dataclass.
            det_cfg = SimpleNamespace(**asdict(det_cfg))

        if custom_vocabulary is not None:
            det_cfg.custom_vocabulary = custom_vocabulary

        self.video_id = video_id
        self.video_path = video_path
        self.annotations_root = annotations_root
        self.intermediate_root = intermediate_root
        self.video_fps = video_fps
        self.det_cfg = det_cfg
        self.accept_labels = accept_labels or {"visible"}

        print("[DETECTION_BACKEND]", getattr(det_cfg, "backend", None))

        self.estimator = _build_estimator(det_cfg)
        self.cache = VideoCache.build(video_id, annotations_root, intermediate_root)

    def filter_candidates(self, candidates: list[Any], target_n: int) -> tuple[list[Any], list[dict[str, Any]]]:
        accepted = []
        rejected = []

        with VideoFrameReader(self.video_path, self.video_fps) as frame_reader:
            for cand in candidates:
                if len(accepted) >= target_n:
                    break

                decision = self.check_one(cand, frame_reader)

                if decision.accepted:
                    # Attach detector metadata for debugging / visualization.
                    if hasattr(cand, "__dict__"):
                        # dataclass is frozen in your current file, so this may not work directly.
                        # Better: convert to dict later and attach metadata there.
                        pass
                    accepted.append((cand, decision.metadata))
                else:
                    rejected.append({
                        "video_id": cand.video_id,
                        "assoc_id": cand.assoc_id,
                        "object_name": cand.object_name,
                        "query_time_sec": cand.query_time_sec,
                        "detection": decision.metadata,
                    })

        return accepted, rejected

    def check_one(self, cand: Any, frame_reader: VideoFrameReader) -> CandidateDetectionDecision:
        time_sec = float(cand.query_time_sec)

        states = determine_in_view_objects(
            video_id=self.video_id,
            time_sec=time_sec,
            annotations_root=self.annotations_root,
            fps=self.video_fps,
            intermediate_root=self.intermediate_root,
            cache=self.cache,
        )

        state = next((row for row in states if str(row.assoc_id) == str(cand.assoc_id)), None)

        if (
            state is None
            or state.status != "ok"
            or state.projected_pixel is None
            or not state.in_view
        ):
            return CandidateDetectionDecision(
                accepted=False,
                metadata={
                    "label": "skipped",
                    "reason": "not a valid in-view sample",
                    "projected_uv": None,
                },
            )

        frame = frame_reader.read_at(time_sec)
        if frame is None:
            return CandidateDetectionDecision(
                accepted=False,
                metadata={
                    "label": "skipped",
                    "reason": "frame read failed",
                    "projected_uv": list(state.projected_pixel),
                },
            )

        if state.mask_bbox is not None:
            x1, y1, x2, y2 = state.mask_bbox
            expected_size = (
                max(1.0, float(x2 - x1)),
                max(1.0, float(y2 - y1)),
            )
            mask_bbox = [float(x1), float(y1), float(x2), float(y2)]
        else:
            d = float(self.det_cfg.default_expected_size_px)
            expected_size = (d, d)
            mask_bbox = None

        result, _ = self.estimator.estimate(
            image_bgr=frame,
            projected_uv=tuple(state.projected_pixel),
            text_prompt=str(cand.object_name),
            expected_box_size_px=expected_size,
            uncertainty_px=self.det_cfg.uncertainty_px,
        )

        metadata = {
            "label": result.label,
            "score": float(result.visibility_score),
            "reason": result.reason,
            "projected_uv": list(state.projected_pixel),
            "mask_bbox": mask_bbox,
            "expected_box_size_px": [float(expected_size[0]), float(expected_size[1])],
            "roi_bbox_xyxy": [
                result.roi.x1,
                result.roi.y1,
                result.roi.x2,
                result.roi.y2,
            ],
            "detections": [
                {
                    "bbox_xyxy": list(det.bbox_xyxy),
                    "confidence": float(det.confidence),
                    "phrase": det.phrase,
                }
                for det in result.detections
            ],
        }

        return CandidateDetectionDecision(
            accepted=result.label in self.accept_labels,
            metadata=metadata,
        )