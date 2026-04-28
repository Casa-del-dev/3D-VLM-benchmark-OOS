from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


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
