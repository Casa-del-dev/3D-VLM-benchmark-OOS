from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from detic_detector import DeticDetector
from roi_visibility import ROIVisibilityEstimator


def parse_bbox(values):
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("--last-seen-bbox must contain 4 numbers: x1 y1 x2 y2")
    return tuple(float(x) for x in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Detic inside a projected ROI on one image.")
    parser.add_argument("--image", required=True, help="Path to input image/frame.")
    parser.add_argument("--detic-root", required=True, help="Path to FIction-Detic or facebookresearch/Detic repo.")
    parser.add_argument(
        "--config-file",
        default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        help="Detic config path, absolute or relative to --detic-root.",
    )
    parser.add_argument(
        "--weights",
        default="models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        help="Detic weights path, absolute or relative to --detic-root.",
    )

    parser.add_argument("--u", type=float, required=True, help="Projected x/u pixel.")
    parser.add_argument("--v", type=float, required=True, help="Projected y/v pixel.")
    parser.add_argument(
        "--prompt",
        nargs="+",
        required=True,
        help=(
            "Custom vocabulary/class names. Prefer short class names, e.g. "
            "--prompt pot pan saucepan instead of a full sentence."
        ),
    )

    parser.add_argument("--expected-width", type=float, default=120.0)
    parser.add_argument("--expected-height", type=float, default=120.0)
    parser.add_argument("--uncertainty-px", type=int, default=40)
    parser.add_argument("--last-seen-bbox", type=float, nargs=4, default=None)

    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--visible-threshold", type=float, default=0.35)
    parser.add_argument("--partial-threshold", type=float, default=0.18)
    parser.add_argument("--roi-scale", type=float, default=1.8)

    parser.add_argument("--output", default="detic_roi_debug.jpg", help="Output debug image path.")
    parser.add_argument("--json-output", default=None, help="Optional JSON output path.")

    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    detector = DeticDetector(
        detic_root=args.detic_root,
        config_file=args.config_file,
        weights=args.weights,
        vocabulary="custom",
        custom_vocabulary=args.prompt,
        confidence_threshold=args.box_threshold,
        device=args.device,
    )

    estimator = ROIVisibilityEstimator(
        detector=detector,
        detector_name="Detic",
        roi_scale=args.roi_scale,
        box_threshold=args.box_threshold,
        visible_threshold=args.visible_threshold,
        partial_threshold=args.partial_threshold,
        smoother=None,
    )

    result, debug = estimator.estimate(
        image_bgr=image,
        projected_uv=(args.u, args.v),
        text_prompt=args.prompt,
        expected_box_size_px=(args.expected_width, args.expected_height),
        uncertainty_px=args.uncertainty_px,
        draw_debug=True,
        last_seen_bbox=parse_bbox(args.last_seen_bbox),
    )

    print("Visibility result")
    print("-----------------")
    print(f"label: {result.label}")
    print(f"score: {result.visibility_score:.3f}")
    print(f"reason: {result.reason}")
    print(f"roi: [{result.roi.x1}, {result.roi.y1}, {result.roi.x2}, {result.roi.y2}]")
    print(f"detections: {len(result.detections)}")
    for i, det in enumerate(result.detections[:10], start=1):
        print(f"  {i}. {det.phrase} conf={det.confidence:.3f} bbox={det.bbox_xyxy}")

    if debug is not None:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), debug)
        if not ok:
            raise RuntimeError(f"Failed to write debug image: {output_path}")
        print(f"Saved debug image to: {output_path}")

    if args.json_output:
        json_path = Path(args.json_output).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": result.label,
            "visibility_score": result.visibility_score,
            "reason": result.reason,
            "projected_uv": list(result.projected_uv),
            "roi": [result.roi.x1, result.roi.y1, result.roi.x2, result.roi.y2],
            "detections": [
                {
                    "bbox_xyxy": list(det.bbox_xyxy),
                    "confidence": det.confidence,
                    "phrase": det.phrase,
                }
                for det in result.detections
            ],
        }
        json_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved JSON result to: {json_path}")


if __name__ == "__main__":
    main()
