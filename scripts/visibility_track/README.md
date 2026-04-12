# Visibility Track

Produces per-object visibility intervals for HD-EPIC videos: for every
object in `assoc_info.json`, when is it actually visible to the head-mounted
camera? The output drives out-of-sight (OOS) VQA generation downstream.

## Pipeline (4 stages)

### Stage 1 -- In-view track (`in_view_track/in_view_track_generator.py`)

Samples every object at `in_view_sampling_fps`, projects its 3D location
into the RGB frame through the FISHEYE624 camera model, and classifies
each sample.

| Status       | Meaning                                            |
|--------------|----------------------------------------------------|
| `in_view`    | Projection lands within the camera image           |
| `out_of_view`| Projection falls outside the image (or no 3D data) |
| `in_motion`  | Person is manipulating the object (no fixed point)  |

Output: `in_view_tracks.jsonl`

### Stage 2 -- Geometric refinement (`geometric_visibility/geometric_view_refinement.py`) *(optional)*

For each sample currently marked `in_view`, casts a ray from the camera
to the object's 3D world location and checks whether fixed scene geometry
(from digital-twin OBJ meshes) blocks the line of sight.

| Status                  | Meaning                                     |
|-------------------------|---------------------------------------------|
| `in_view`               | In view and line of sight is clear           |
| `out_of_view`           | (passthrough from stage 1)                   |
| `in_motion`             | (passthrough from stage 1)                   |
| `geometrically_occluded`| Line of sight blocked by scene geometry      |

Output: `geometric_refined_in_view_tracks.jsonl`

### Stage 3 -- Fixture overlay (`combine.py`)

First generates per-fixture open/closed intervals from narrations (same
logic as `open_close_track/build_tracks.py`). Then checks every sample
currently marked `in_view` or `geometrically_occluded`:

- Inside a **closed** fixture (medium+ confidence) -> `occluded_inside_closed_fixture`
- Inside an **open** fixture -> `potentially_visible_inside_open_fixture`
- Not inside any fixture -> keeps current state

| Status                                  | Meaning                                        |
|-----------------------------------------|------------------------------------------------|
| `in_view`                               | In view, not inside any fixture                 |
| `out_of_view`                           | (passthrough)                                   |
| `in_motion`                             | (passthrough)                                   |
| `geometrically_occluded`                | Occluded by geometry, not inside any fixture    |
| `occluded_inside_closed_fixture`        | Inside a closed fixture                         |
| `potentially_visible_inside_open_fixture`| Inside an open fixture, needs detection check  |

Output: `fixture_intervals.jsonl` + `coarse_visibility_track.jsonl`

### Stage 4 -- Detection refinement (`detection_refinement.py`) *(optional)*

Runs Grounding DINO on sampled frames for each interval marked
`potentially_visible_inside_open_fixture`. All other statuses pass through.

| Status                                | Meaning                                 |
|---------------------------------------|-----------------------------------------|
| `in_view`                             | (passthrough)                            |
| `out_of_view`                         | (passthrough)                            |
| `in_motion`                           | (passthrough)                            |
| `geometrically_occluded`              | (passthrough)                            |
| `occluded_inside_closed_fixture`      | (passthrough)                            |
| `observed_visible_in_open_fixture`    | Detected by Grounding DINO              |
| `observed_not_visible_in_open_fixture`| Not detected by Grounding DINO          |

Output: `visibility_track.jsonl` + `visibility_track_summary.json`

## Run

```bash
# Full pipeline -- all videos listed in config
python -m scripts.visibility_track.generate_visibility_track

# Process all videos for a participant (auto-discovered from intermediate_data_root)
python -m scripts.visibility_track.generate_visibility_track --participant P01

# Process specific video(s)
python -m scripts.visibility_track.generate_visibility_track --video P01-20240203-184045
python -m scripts.visibility_track.generate_visibility_track --video P01-20240203-184045 --video P01-20240202-161948

# Mix participant + extra video, skip optional stages
python -m scripts.visibility_track.generate_visibility_track --participant P01 --no-detection
python -m scripts.visibility_track.generate_visibility_track --participant P01 --no-geometric

# Individual stages
python -m scripts.visibility_track.in_view_track.in_view_track_generator --video P01-20240203-184045
python -m scripts.visibility_track.geometric_visibility.geometric_view_refinement --video P01-20240203-184045
python -m scripts.visibility_track.combine --video P01-20240203-184045
python -m scripts.visibility_track.detection_refinement --video P01-20240203-184045

# Debug overlay video (coloured dots + out-of-view sidebar)
python -m scripts.visibility_track.overlay_video --video P01-20240203-184045 --fps 1.0
python -m scripts.visibility_track.overlay_video --participant P01 --fps 1.0
```

## Config

All entry points read `visibility_track_config.yaml`. Key fields:

- `annotations_root`, `data_root`, `intermediate_data_root`, `output_root`
- `inputs.participants` -- used by `--participant` auto-discovery when no CLI flag is given
- `inputs.videos` -- fallback list when neither `--video` nor `--participant` is passed
- `in_view_sampling_fps`, `object_detection_sampling_fps`, `video_fps`
- `geometric_occlusion.enabled` and `geometric_occlusion.tolerance_m`
- `object_detection.enabled` and its Grounding DINO thresholds

## Outputs

Outputs are written to `<project_root>/outputs/visibility_track/<video_id>/`:

```
outputs/visibility_track/<video_id>/
  in_view_tracks.jsonl                       (stage 1)
  geometric_refined_in_view_tracks.jsonl     (stage 2, optional)
  fixture_intervals.jsonl                    (stage 3)
  coarse_visibility_track.jsonl              (stage 3)
  visibility_track.jsonl                     (stage 4)
  visibility_track_summary.json              (stage 4)
  visibility_track_overlay.mp4               (overlay_video, optional)
```
