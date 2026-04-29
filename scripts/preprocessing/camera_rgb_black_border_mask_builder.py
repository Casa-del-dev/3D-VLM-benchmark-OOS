import json
import numpy as np
from PIL import Image

# Path to your uploaded calibration file
calib_path = "hd-epic-annotations/Intermediate_data/P01/P01-20240202-110250/device_calibration.json"

# Output path
output_path = "hd-epic-annotations/camera-rgb_black_border_mask.png"

# Load calibration
with open(calib_path, "r") as f:
    calib = json.load(f)

# Get camera-rgb calibration
cam = calib["cameras"]["camera-rgb"]

# Image size
width, height = cam["image_size"]

# Aria FISHEYE624 projection_params for camera-rgb:
# [f, cx, cy, ...]
projection_params = cam["projection_params"]
cx = projection_params[1]
cy = projection_params[2]

# Valid image radius
valid_radius = cam["valid_radius"]
if valid_radius is None:
    raise ValueError("camera-rgb does not have a valid_radius in the calibration file.")

# Build pixel grid
x, y = np.meshgrid(np.arange(width), np.arange(height))

# Valid region: inside the calibrated circle
valid_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= (valid_radius ** 2)

# Black border mask: True where invalid / black border
black_border_mask = ~valid_mask

# Save as PNG:
# white = black border region, black = valid region
mask_img = (black_border_mask.astype(np.uint8) * 255)
Image.fromarray(mask_img).save(output_path)

print(f"Saved black border mask to: {output_path}")