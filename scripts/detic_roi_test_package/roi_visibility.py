from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from common import DetectionResult, ROIBox, VisibilityResult


class TemporalSmoother:
    def __init__(self, alpha: float = 0.7):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)
        self._score: Optional[float] = None

    def update(self, score: float) -> float:
        if self._score is None:
            self._score = float(score)
        else:
            self._score = self.alpha * float(score) + (1.0 - self.alpha) * self._score
        return self._score


class ROIVisibilityEstimator:
    """
    Detector-agnostic ROI visibility estimator.

    The detector object only needs a method:

        detect(image_bgr, text_prompt, box_threshold, text_threshold=None)
            -> list[DetectionResult]

    This means you can plug in Detic, Grounding DINO, YOLO, etc.
    """

    def __init__(
        self,
        detector,
        detector_name: str = "detector",
        roi_scale: float = 1.8,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        visible_threshold: float = 0.35,
        partial_threshold: float = 0.18,
        smoother: Optional[TemporalSmoother] = None,
        score_conf_weight: float = 0.50,
        score_location_weight: float = 0.35,
        score_size_weight: float = 0.15,
    ) -> None:
        self.detector = detector
        self.detector_name = detector_name
        self.roi_scale = float(roi_scale)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.visible_threshold = float(visible_threshold)
        self.partial_threshold = float(partial_threshold)
        self.smoother = smoother

        total = score_conf_weight + score_location_weight + score_size_weight
        if total <= 0:
            raise ValueError("At least one score weight must be positive.")
        self.score_conf_weight = score_conf_weight / total
        self.score_location_weight = score_location_weight / total
        self.score_size_weight = score_size_weight / total

    @staticmethod
    def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance_score(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        norm: float,
    ) -> float:
        d = math.dist(p1, p2)
        return max(0.0, 1.0 - d / max(norm, 1.0))

    @staticmethod
    def _size_score(box: Tuple[int, int, int, int], expected_roi: ROIBox) -> float:
        x1, y1, x2, y2 = box
        det_area = max(1, x2 - x1) * max(1, y2 - y1)
        exp_area = max(1, expected_roi.area())
        ratio = det_area / exp_area
        if ratio <= 0:
            return 0.0
        return float(np.exp(-abs(np.log(max(ratio, 1e-6))) / 1.2))

    @staticmethod
    def _local_to_global(det: DetectionResult, roi: ROIBox) -> DetectionResult:
        x1, y1, x2, y2 = det.bbox_xyxy
        return DetectionResult(
            bbox_xyxy=(x1 + roi.x1, y1 + roi.y1, x2 + roi.x1, y2 + roi.y1),
            confidence=float(det.confidence),
            phrase=str(det.phrase),
        )

    def build_roi(
        self,
        image_shape: Tuple[int, int, int],
        projected_uv: Tuple[float, float],
        expected_box_size_px: Optional[Tuple[float, float]] = None,
        uncertainty_px: int = 40,
        last_seen_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> ROIBox:
        image_h, image_w = image_shape[:2]
        u, v = projected_uv
        cx, cy = float(u), float(v)

        if last_seen_bbox is not None:
            bx1, by1, bx2, by2 = last_seen_bbox
            box_w = max(1.0, float(bx2 - bx1))
            box_h = max(1.0, float(by2 - by1))
            half_w = int(0.5 * self.roi_scale * box_w + uncertainty_px)
            half_h = int(0.5 * self.roi_scale * box_h + uncertainty_px)
        elif expected_box_size_px is not None:
            exp_w, exp_h = expected_box_size_px
            half_w = int(0.5 * self.roi_scale * float(exp_w) + uncertainty_px)
            half_h = int(0.5 * self.roi_scale * float(exp_h) + uncertainty_px)
        else:
            half_w = int(uncertainty_px)
            half_h = int(uncertainty_px)

        roi = ROIBox(
            x1=int(round(cx)) - half_w,
            y1=int(round(cy)) - half_h,
            x2=int(round(cx)) + half_w,
            y2=int(round(cy)) + half_h,
        )
        return roi.clip(image_w, image_h)

    def _score_detection(
        self,
        det: DetectionResult,
        roi: ROIBox,
        projected_uv: Tuple[float, float],
    ) -> float:
        det_center = self._box_center(det.bbox_xyxy)
        roi_diag = math.hypot(max(1, roi.width()), max(1, roi.height()))
        loc_score = self._distance_score(det_center, projected_uv, norm=0.6 * roi_diag)
        size_score = self._size_score(det.bbox_xyxy, roi)
        conf_score = float(det.confidence)

        return (
            self.score_conf_weight * conf_score
            + self.score_location_weight * loc_score
            + self.score_size_weight * size_score
        )

    def estimate(
        self,
        image_bgr: np.ndarray,
        projected_uv: Tuple[float, float],
        text_prompt: Union[str, Sequence[str]],
        expected_box_size_px: Optional[Tuple[float, float]] = None,
        uncertainty_px: int = 40,
        draw_debug: bool = False,
        last_seen_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> tuple[VisibilityResult, Optional[np.ndarray]]:
        image_h, image_w = image_bgr.shape[:2]
        u, v = projected_uv

        if not (0 <= u < image_w and 0 <= v < image_h):
            result = VisibilityResult(
                projected_uv=projected_uv,
                roi=ROIBox(0, 0, 0, 0),
                detections=[],
                visibility_score=0.0,
                label="out_of_view",
                reason="Projected point is outside image bounds.",
            )
            debug = image_bgr.copy() if draw_debug else None
            if debug is not None:
                cv2.circle(debug, (int(round(u)), int(round(v))), 5, (0, 0, 255), -1)
            return result, debug

        roi = self.build_roi(
            image_shape=image_bgr.shape,
            projected_uv=projected_uv,
            expected_box_size_px=expected_box_size_px,
            uncertainty_px=uncertainty_px,
            last_seen_bbox=last_seen_bbox,
        )
        roi_img = image_bgr[roi.y1:roi.y2, roi.x1:roi.x2]

        if roi_img.size == 0:
            result = VisibilityResult(
                projected_uv=projected_uv,
                roi=roi,
                detections=[],
                visibility_score=0.0,
                label="uncertain",
                reason="ROI is empty after clipping.",
            )
            return result, image_bgr.copy() if draw_debug else None

        local_dets = self.detector.detect(
            roi_img,
            text_prompt=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        detections = [self._local_to_global(d, roi) for d in local_dets]

        if not detections:
            raw_score = 0.0
            label = "not_visible"
            reason = f"No {self.detector_name} detection found inside the projected ROI."
        else:
            best_det = max(detections, key=lambda d: self._score_detection(d, roi, projected_uv))
            raw_score = self._score_detection(best_det, roi, projected_uv)

            if raw_score >= self.visible_threshold:
                label = "visible"
                reason = (
                    f"{self.detector_name} matched the ROI. "
                    f"Best class='{best_det.phrase}', conf={best_det.confidence:.3f}"
                )
            elif raw_score >= self.partial_threshold:
                label = "partially_visible"
                reason = (
                    f"Weak or offset {self.detector_name} match near the ROI. "
                    f"Best class='{best_det.phrase}', conf={best_det.confidence:.3f}"
                )
            else:
                label = "not_visible"
                reason = (
                    f"{self.detector_name} detection was too weak or mismatched. "
                    f"Best class='{best_det.phrase}', conf={best_det.confidence:.3f}"
                )

        score = self.smoother.update(raw_score) if self.smoother is not None else raw_score
        if self.smoother is not None and detections:
            if score >= self.visible_threshold:
                label = "visible"
            elif score >= self.partial_threshold:
                label = "partially_visible"
            else:
                label = "not_visible"

        result = VisibilityResult(
            projected_uv=projected_uv,
            roi=roi,
            detections=detections,
            visibility_score=float(score),
            label=label,
            reason=reason,
        )

        debug = None
        if draw_debug:
            debug = image_bgr.copy()
            self.draw_debug(debug, result)
        return result, debug

    @staticmethod
    def draw_debug(image: np.ndarray, result: VisibilityResult) -> None:
        u, v = result.projected_uv
        cv2.circle(image, (int(round(u)), int(round(v))), 5, (0, 255, 255), -1)
        cv2.rectangle(
            image,
            (result.roi.x1, result.roi.y1),
            (result.roi.x2, result.roi.y2),
            (255, 255, 0),
            2,
        )
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.phrase}:{det.confidence:.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        summary = f"label={result.label}, score={result.visibility_score:.2f}"
        cv2.putText(
            image,
            summary,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
