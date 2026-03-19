# 📦 Egocentric Object Geometry & Visibility Tools

This folder contains two Python scripts for analyzing object positions and visibility in egocentric video data (e.g., HD-EPIC / Aria datasets):

* **`relative_relation_with_viz.py`** -> Computes **relative spatial relationships** between objects in an egocentric coordinate system.
* **`in_view_objects_with_viz.py`** → Determines **which objects are visible** in the RGB camera at a given time by checking if the projected 3D world coordinates fall inside the image plane.

---

## Overview

These tools operate on:

* Object movement annotations from `assoc_info.json` and `mask_info.json`
* `T_world_device` from `framewise_info.jsonl`
* `T_device_camera`  & camera intrinsics from `device_calibration.json`
* Then `T_world_camera = T_world_device @ T_device_camera` and inverts that to get `T_camera_world`

---
## Expected Directory Structure for the Data 

```
hd-epic-annotations/
  ├── scene-and-object-movements/
  │     ├── assoc_info.json
  │     ├── mask_info.json
  ├── Intermediate_data/
        ├── P01/
             ├── P01-xxxx/
                  ├── device_calibration.json
                  ├── framewise_info.jsonl
```
You can also manually change the path to the data according to your own directory structure.
---

# 1. Relative Object Relations (Egocentric Frame)

File: relative_relation_with_viz.py

## Purpose

Computes the position of **every object A relative to an anchor object B** using an **egocentric coordinate system**:

* Origin -> Anchored object **B**
* Axes -> Aligned with camera at a frame where B is visible

Also:

* Determines visibility of objects in that anchor frame
* Optionally overlays coordinate axes on the video

---
## General Command Template
```bash
python relative_relation_with_vis.py \
  --video <video_id> \
  --obj_b "<anchor_object_name>" \
  --t <time_in_seconds> \
  --output <output_yaml_path> \
  --video_path <path_to_video_file> \
  --vis_output <output_visualization_path>
```
---
## Example Usage 

```bash
python relative_relation_with_vis.py \
  --video P04-20240413-142619 \
  --obj_b "stacked of 3 bowls" \
  --t 208 \
  --output output.yaml \
  --video_path /HD-EPIC/Videos/P04_copy/P04-20240413-142619.mp4 \
  --vis_output anchor_coordinate_visualization.jpg
```

---

# 2. Object Visibility in RGB View

File: `in_view_objects_with_viz.py`

## Purpose

Determines **which objects are visible in the RGB camera** at time `t` by checking if the projected coordinates are out-of-frame.

---
## General Command Template
```bash
python relative_relation_with_vis.py \
  --video <video_id> \
  --obj_b "<anchor_object_name>" \
  --t <time_in_seconds> \
  --output <output_yaml_path> \
  --video_path <path_to_video_file> \
  --vis_output <output_visualization_path>
```
---
## Example Usage

```bash
python scripts/preprocessing/in_view_objects_with_viz.py \
  --video P04-20240413-142619 \
  --t 208 \
  --video_file /HD-EPIC/Videos/P04_copy/P04-20240413-142619.mp4 \
  --viz_output annotated.png \
  --draw_labels
```
---



