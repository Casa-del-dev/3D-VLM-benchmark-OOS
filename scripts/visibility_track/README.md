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

For each query time, the per-object track is chosen by precedence:
*in_motion* (if the time falls inside a movement segment) > latest *past*
track > earliest *future* track. Projection rejects samples behind the
camera, outside image bounds, or that land on the fisheye black-border
mask (see "Run" below for how the mask is built).

Output: `in_view_tracks.jsonl`

The core projection math lives in `in_view_track/in_view_determination.py`
(shared with stage 4); it is also where `VideoCache` caches per-video
JSON blobs to avoid redundant I/O across time steps.

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

Meshes are loaded per participant from
`<data_root>/HD-EPIC/Digital-Twin/blenders/meshes/<participant>/`. Videos
are grouped by participant so each mesh is loaded once. Ray-mesh queries
use `pyembree` if available (falling back to trimesh's pure-Python
intersector), wrapped in `geometric_visibility/mesh_scene.py`.

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

Fixture intervals come from the `open_close_track/` submodule:
narrations are parsed (`narrations.py`) and optionally snapped to nearby
open/close sounds from `HD_EPIC_Sounds.csv`; the kitchen catalog
(`fixtures.py`) is built from `mask_info.json`; `resolver.py` maps each
open/close event to a concrete fixture by combining gaze alignment with
camera-to-fixture proximity and assigns a confidence tier; `state_machine.py`
forward- and backward-fills `Interval` objects from the resolved events.
Only confidence tiers `very_high`, `high`, and `medium` hide objects --
lower tiers are ignored when occluding samples.

Output: `fixture_intervals.jsonl` + `coarse_visibility_track.jsonl`

### Stage 4 -- Detection refinement (`detection_refinement.py`) *(optional)*

Runs an object detector on sampled frames for each interval marked
`potentially_visible_inside_open_fixture`. All other statuses pass
through. Two detector backends are available, selected via
`object_detection.backend` in the config:

- `groundingdino` -- HF Transformers Grounding DINO, run on the projected ROI
- `detic` -- FIction-Detic with LVIS/Objects365/OpenImages/COCO/custom vocabularies

Both share the same ROI estimator (`object_detection/roi_visibility.py`)
and result schema. Within each interval, sampling is adaptive: a single
center frame for very short intervals (<=1 s), start/mid/end for medium
intervals (<=4 s), and quartile-spaced (5 frames) for longer intervals,
all rate-limited by `object_detection_sampling_fps`.

| Status                                | Meaning                                 |
|---------------------------------------|-----------------------------------------|
| `in_view`                             | (passthrough)                            |
| `out_of_view`                         | (passthrough)                            |
| `in_motion`                           | (passthrough)                            |
| `geometrically_occluded`              | (passthrough)                            |
| `occluded_inside_closed_fixture`      | (passthrough)                            |
| `observed_visible_in_open_fixture`    | Detected by the configured backend       |
| `observed_not_visible_in_open_fixture`| Not detected by the configured backend   |

Output: `visibility_track.jsonl` + `visibility_track_summary.json`

## Run

First generate a binary valid-region mask for the fisheye camera:      
white / nonzero pixel = black border / invalid region
black / zero pixel    = valid image region

Run
```
python scripts/preprocessing/camera_rgb_black_border_mask_builder.py
```
will generate the binary mask to the hd-epic-annotations folder. (You may need to change the path in the python script, jsut make sure to output to the annotation folder, as the in-view generator will need to use this mask and load the mask from the annotation folder by default).


Second:
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

# Individual stages (--video repeatable; --participant NOT supported here)
python -m scripts.visibility_track.in_view_track.in_view_track_generator --video P01-20240203-184045
python -m scripts.visibility_track.geometric_visibility.geometric_view_refinement --video P01-20240203-184045
python -m scripts.visibility_track.combine --video P01-20240203-184045
python -m scripts.visibility_track.detection_refinement --video P01-20240203-184045

# Debug overlay video (coloured dots + out-of-view sidebar)
python -m scripts.visibility_track.overlay_video --video P01-20240203-184045 --fps 1.0
python -m scripts.visibility_track.overlay_video --participant P01 --fps 1.0
python -m scripts.visibility_track.overlay_video --video P01-20240203-184045 --output /tmp/overlay.mp4

# Static timeline plot (Gantt + time-distribution chart)
python -m scripts.visibility_track.visualize_track --video P01-20240203-184045
python -m scripts.visibility_track.visualize_track --participant P01
python -m scripts.visibility_track.visualize_track --video P01-20240203-184045 --output /tmp/my_plot.png
```

Only `generate_visibility_track.py`, `overlay_video.py`, and `visualize_track.py`
accept `--participant`; the per-stage scripts take `--video` and `--config` only.
`overlay_video.py`'s `--fps` defaults to 1.0.

## Config

All entry points read `visibility_track_config.yaml`. Relative paths in
the YAML are resolved against the config file's directory, so the file
can be moved without code changes.

Key fields:

- `annotations_root`, `data_root`, `intermediate_data_root`, `output_root`
- `inputs.participants` -- used by `--participant` auto-discovery when no CLI flag is given
- `inputs.videos` -- fallback list when neither `--video` nor `--participant` is passed
- `in_view_sampling_fps`, `object_detection_sampling_fps`, `video_fps`
- `geometric_occlusion.enabled`, `geometric_occlusion.tolerance_m`
- `object_detection.enabled`, `object_detection.backend` (`groundingdino` or `detic`)
- Grounding DINO: `model_id` (e.g. `IDEA-Research/grounding-dino-tiny|-base|-large`)
- Detic: `detic_root`, `config_file`, `weights`, `vocabulary`, `device`
- Shared detection params: `roi_scale`, `box_threshold`, `text_threshold`,
  `visible_threshold`, `partial_threshold`, `uncertainty_px`,
  `default_expected_size_px`
- `random_seed`

Config loading and per-video path resolution (annotations, framewise,
video file, output dir) live in `common.py` (`PipelineConfig`), which
also provides the JSONL read/write helpers used by every stage.

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
  visibility_track_plot.png                  (visualize_track, optional)
```

## Directory layout

```
scripts/visibility_track/
  common.py                       Config loading, path resolution, JSONL I/O
  generate_visibility_track.py    Driver for stages 1-4
  combine.py                      Stage 3
  detection_refinement.py         Stage 4
  visualize_track.py              Gantt + distribution plot
  overlay_video.py                Visibility overlay on source video
  visibility_track_config.yaml

  in_view_track/
    in_view_track_generator.py    Stage 1 entry point
    in_view_determination.py      Per-frame projection + VideoCache (shared)

  geometric_visibility/
    geometric_view_refinement.py  Stage 2 entry point
    mesh_scene.py                 RayOcclusionChecker (trimesh / pyembree)

  open_close_track/
    build_tracks.py               Wires the submodule into combine.py
    narrations.py                 Open/close events (optional sound-snap)
    fixtures.py                   Per-participant kitchen catalog
    resolver.py                   Event -> fixture_id + confidence
    state_machine.py              Events -> per-fixture Interval rows
    framewise.py                  Camera pose / gaze loader
    config.py                     OPENABLE_FIXTURE_TYPES + frame/time helpers

  object_detection/
    roi_visibility.py             Detector-agnostic ROI estimator
    groundingdino_roi_visibility.py  Grounding DINO backend
    detic_detector.py             Detic backend
    detection_types.py            ROIBox / DetectionResult / VisibilityResult
```
