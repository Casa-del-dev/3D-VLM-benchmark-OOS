from __future__ import annotations

"""
ROI-based visibility estimation using Grounding DINO.

This version assumes you ALREADY have the projected object coordinates in the image,
so it skips camera intrinsics / pose projection and starts from:
    1. projected pixel location (u, v)
    2. expected object size or a manually chosen ROI size
    3. text prompt(s) for Grounding DINO

Pipeline:
1. Build an ROI around the provided image coordinates.
2. Run Grounding DINO only inside that ROI.
3. Map detections back to full-image coordinates.
4. Score detections based on confidence, location agreement, and size plausibility.
5. Return visible / partially_visible / not_visible / out_of_view.

Dependencies:
    pip install transformers torch pillow opencv-python numpy

Notes:
- Uses Hugging Face Transformers Grounding DINO interface.
- Prompts should be lowercase and typically end with a dot, e.g.:
      "a red bowl. a ceramic bowl. a mixing bowl."
- If you want multiple candidate categories, separate them with periods.

NOTE: Kept in sync with `scripts/vqa_oos/oos_location_recall/groundingdino_roi_visibility.py`.
Any logic change here must be mirrored there (and vice versa).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
import math

import cv2
import numpy as np
from PIL import Image

_IMPORT_ERROR: Optional[BaseException] = None

try:
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except Exception as e:  # pragma: no cover
    torch = None
    AutoProcessor = None
    AutoModelForZeroShotObjectDetection = None
    _IMPORT_ERROR = e


@dataclass
class ROIBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def area(self) -> int:
        return self.width() * self.height()

    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clip(self, image_w: int, image_h: int) -> "ROIBox":
        return ROIBox(
            x1=max(0, min(self.x1, image_w - 1)),
            y1=max(0, min(self.y1, image_h - 1)),
            x2=max(0, min(self.x2, image_w)),
            y2=max(0, min(self.y2, image_h)),
        )


@dataclass
class DetectionResult:
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    phrase: str


@dataclass
class VisibilityResult:
    projected_uv: Tuple[float, float]
    roi: ROIBox
    detections: List[DetectionResult]
    visibility_score: float
    label: str
    reason: str


class TemporalSmoother:
    def __init__(self, alpha: float = 0.7):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self._score: Optional[float] = None

    def update(self, score: float) -> float:
        if self._score is None:
            self._score = float(score)
        else:
            self._score = self.alpha * float(score) + (1.0 - self.alpha) * self._score
        return self._score


def get_device() -> str:
    if torch is None:
        raise ImportError(f"torch failed to import: {_IMPORT_ERROR}") from _IMPORT_ERROR

    if torch.cuda.is_available():
        return "cuda"

    # MPS is Apple/macOS only; guard it safely.
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return "mps"

    return "cpu"


class GroundingDINODetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
    ) -> None:
        if AutoProcessor is None or AutoModelForZeroShotObjectDetection is None or torch is None:
            if _IMPORT_ERROR is not None:
                raise ImportError(
                    f"Dependency import failed. Original error: {_IMPORT_ERROR!r}"
                ) from _IMPORT_ERROR
            raise ImportError(
                "Dependency import failed: torch/transformers did not load correctly."
            )

        if device is None:
            device = get_device()

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    @staticmethod
    def _prepare_prompt(text_prompt: Union[str, Sequence[str]]) -> str:
        if isinstance(text_prompt, str):
            text = text_prompt.strip().lower()
            return text if text.endswith(".") else text + "."

        cleaned = []
        for item in text_prompt:
            x = str(item).strip().lower().rstrip(".")
            if x:
                cleaned.append(x)
        if not cleaned:
            raise ValueError("text_prompt is empty")
        return ". ".join(cleaned) + "."

    def detect(
        self,
        image_bgr: np.ndarray,
        text_prompt: Union[str, Sequence[str]],
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> List[DetectionResult]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        prompt = self._prepare_prompt(text_prompt)

        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [pil_image.size[::-1]]  # (h, w)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )

        if not results:
            return []

        result = results[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])

        preds: List[DetectionResult] = []
        for box, score, label in zip(boxes, scores, labels):
            if hasattr(box, "tolist"):
                box = box.tolist()
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            preds.append(
                DetectionResult(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(score),
                    phrase=str(label),
                )
            )
        preds.sort(key=lambda d: d.confidence, reverse=True)
        return preds


class ROIGroundingDINOVisibilityEstimator:
    def __init__(
        self,
        detector: GroundingDINODetector,
        roi_scale: float = 2.0,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        visible_threshold: float = 0.62,
        partial_threshold: float = 0.28,
        smoother: Optional[TemporalSmoother] = None,
    ) -> None:
        self.detector = detector
        self.roi_scale = float(roi_scale)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.visible_threshold = float(visible_threshold)
        self.partial_threshold = float(partial_threshold)
        self.smoother = smoother

    @staticmethod
    def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance_score(p1: Tuple[float, float], p2: Tuple[float, float], norm: float) -> float:
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
            confidence=det.confidence,
            phrase=det.phrase,
        )

    def build_roi(
        self,
        image_shape: Tuple[int, int, int],
        projected_uv: Tuple[float, float],
        expected_box_size_px: Optional[Tuple[float, float]] = None,
        uncertainty_px: int = 40,
        last_seen_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> ROIBox:
        image_h, image_w = image_shape[:2]
        u, v = projected_uv

        # ROI center
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
            half_w = uncertainty_px
            half_h = uncertainty_px

        roi = ROIBox(
            x1=int(round(cx)) - half_w,
            y1=int(round(cy)) - half_h,
            x2=int(round(cx)) + half_w,
            y2=int(round(cy)) + half_h,
        )

        return roi.clip(image_w, image_h)

    def _score_detection(self, det: DetectionResult, roi: ROIBox, projected_uv: Tuple[float, float]) -> float:
        det_center = self._box_center(det.bbox_xyxy)
        roi_diag = math.hypot(max(1, roi.width()), max(1, roi.height()))
        loc_score = self._distance_score(det_center, projected_uv, norm=0.6 * roi_diag)
        size_score = self._size_score(det.bbox_xyxy, roi)
        conf_score = float(det.confidence)
        return 0.50 * conf_score + 0.35 * loc_score + 0.15 * size_score

    def estimate(
        self,
        image_bgr: np.ndarray,
        projected_uv: Tuple[float, float],
        text_prompt: Union[str, Sequence[str]],
        expected_box_size_px: Optional[Tuple[float, float]] = None,
        uncertainty_px: int = 40,
        draw_debug: bool = False,
        last_seen_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[VisibilityResult, Optional[np.ndarray]]:
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
            reason = "No Grounding DINO detection found inside the projected ROI."
        else:
            best_det = max(detections, key=lambda d: self._score_detection(d, roi, projected_uv))
            raw_score = self._score_detection(best_det, roi, projected_uv)
            if raw_score >= self.visible_threshold:
                label = "visible"
                reason = (
                    f"Grounding DINO matched the ROI. Best phrase='{best_det.phrase}', "
                    f"conf={best_det.confidence:.3f}"
                )
            elif raw_score >= self.partial_threshold:
                label = "partially_visible"
                reason = (
                    f"Weak or offset match near the ROI. Best phrase='{best_det.phrase}', "
                    f"conf={best_det.confidence:.3f}"
                )
            else:
                label = "not_visible"
                reason = (
                    f"Detection was too weak or mismatched. Best phrase='{best_det.phrase}', "
                    f"conf={best_det.confidence:.3f}"
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
            cv2.putText(image, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        summary = f"label={result.label}, score={result.visibility_score:.2f}"
        cv2.putText(image, summary, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def example_usage() -> None:
    image = cv2.imread("frame.jpg")
    if image is None:
        print("Put a test image named 'frame.jpg' next to this script.")
        return

    detector = GroundingDINODetector(model_id="IDEA-Research/grounding-dino-tiny")
    estimator = ROIGroundingDINOVisibilityEstimator(
        detector=detector,
        roi_scale=1.2,
        box_threshold=0.30,
        text_threshold=0.25,
        visible_threshold=0.62,
        partial_threshold=0.28,
        smoother=TemporalSmoother(alpha=0.6),
    )

    projected_uv = (640.0, 360.0)
    text_prompt = [
        "a bowl",
        "a stack of bowls",
        "a mixing bowl",
    ]

    result, debug = estimator.estimate(
        image_bgr=image,
        projected_uv=projected_uv,
        text_prompt=text_prompt,
        expected_box_size_px=(120, 120),
        uncertainty_px=50,
        draw_debug=True,
    )

    print("Visibility result")
    print("-----------------")
    print(f"label: {result.label}")
    print(f"score: {result.visibility_score:.3f}")
    print(f"reason: {result.reason}")
    print(f"roi: {result.roi}")
    print(f"detections: {len(result.detections)}")

    if debug is not None:
        cv2.imwrite("groundingdino_debug.jpg", debug)
        print("Saved debug image to groundingdino_debug.jpg")


if __name__ == "__main__":
    example_usage()