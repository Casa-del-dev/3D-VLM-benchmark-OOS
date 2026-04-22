import os
import cv2
from groundingdino_roi_visibility import (
    GroundingDINODetector,
    ROIGroundingDINOVisibilityEstimator,
    TemporalSmoother,
)

image = cv2.imread("/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/data/for_debug/time_6.jpg")

if image is None:
    raise FileNotFoundError("Failed to load input image.")

detector = GroundingDINODetector(
    model_id="IDEA-Research/grounding-dino-tiny"
)

estimator = ROIGroundingDINOVisibilityEstimator(
    detector=detector,
    roi_scale=1.8,
    box_threshold=0.4,
    text_threshold=0.4,
    visible_threshold=0.62,
    partial_threshold=0.28,
    smoother=None,
)

projected_uv = (408.3171177561637, 528.5396005453204)
text_prompt = ["a pot in the fridge"]

result, debug = estimator.estimate(
    image_bgr=image,
    projected_uv=projected_uv,
    text_prompt=text_prompt,
    expected_box_size_px=(120, 120),
    uncertainty_px=40,
    draw_debug=True,
    last_seen_bbox= [584.86154, 568.01368, 909.78462, 758.15385]
)

print(result.label)
print(result.visibility_score)
print(result.reason)

if debug is not None:
    output_dir = "/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/scripts/outputs/object_detection"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "object_detection_debug.jpg")
    success = cv2.imwrite(output_path, debug)

    if success:
        print(f"Saved debug image to: {output_path}")
    else:
        print(f"Failed to save debug image to: {output_path}")