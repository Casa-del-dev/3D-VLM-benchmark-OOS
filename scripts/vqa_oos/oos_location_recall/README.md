# HD-EPIC Out-of-Sight Location Recall (Question Type 1)

This document defines a complete plan for generating the first out-of-sight (OOS) location recall question type for HD-EPIC.

The goal is to create VQA pairs where an object is currently out of view at time `t`, and the model must recall where that object is located.

## Quick Start (Setup, Run, Test)

### Dependencies

- This pipeline was run using a Conda environment created from `env.yaml` at the repository root.
- `ffmpeg` must also be installed on the system for frame extraction and visualization/debug scripts.

Recommended setup from repository root:

```bash
conda env create -f env.yaml
conda activate hdepic-vlm-sanity
```

Install ffmpeg on macOS (Homebrew example):

```bash
brew install ffmpeg
```

### Run Question Generation

#### To generate in staged format

1. precompute in-view tracks of all objects in a video
```bash
python scripts/vqa_oos/oos_location_recall/precompute_inview_tracks.py --config scripts/vqa_oos/oos_location_recall/staged_oos_question_generator_config.yaml  
```
2. Generate staged questions 
So far, the question include: visibility check, last visible frame detetcion, nearest fixture detection, camera viewing direction tracking
```bash
python scripts/vqa_oos/oos_location_recall/staged_oos_question_generator.py --config scripts/vqa_oos/oos_location_recall/staged_oos_question_generator_config.yaml
```

#### Ivo's version
From repository root:

```bash
python scripts/vqa_oos/oos_location_recall/question_generator.py --config scripts/vqa_oos/oos_location_recall/oos_location_recall_config.yaml
```

Optional output override:

```bash
python scripts/vqa_oos/oos_location_recall/question_generator.py --config scripts/vqa_oos/oos_location_recall/oos_location_recall_config.yaml --output_json outputs/oos_location_recall_questions.json --clipOffset 20 --videoStart
```

### Run Tests / Debug Scripts

This module currently provides script-based integration checks in `scripts/vqa_oos/oos_location_recall/tests/`.

1. Visibility track integration grid test:

```bash
python scripts/vqa_oos/oos_location_recall/tests/test_in_view_track_generator.py --video_id P01-20240203-184045
```

2. End-to-end question generation visual tester:

```bash
python scripts/vqa_oos/oos_location_recall/tests/oos_location_recall_tester.py --video_id P01-20240203-184045 --clipOffset 20 --videoStart
```

3. New question types

```bash
python scripts/vqa_oos/oos_location_recall/tests/staged_oos_benchmark.py --config scripts/vqa_oos/oos_location_recall/tests/staged_oos_benchmark_config.yaml --pre_context_sec 4.0
```

And to view the question

```bash
python scripts/vqa_oos/oos_location_recall/tests/staged_oos_benchmark_review.py --config scripts/vqa_oos/oos_location_recall/tests/staged_oos_benchmark_review_config.yaml
```

Both scripts write debug artifacts into the `outputs/` folder.

## 1. Benchmark Goal

Given a video and a query time `t`, ask about object `A` that was seen earlier but is out of view at `t`.

We generate two question families:

1. Absolute fixture location question
2. Relative location question with respect to anchor object `B`

Both are generated from the same key frame/object selection pipeline.

## 2. Question Templates

### 2.1 Absolute fixture question

Template:

`[OBJECT A] was seen earlier in the video. At the current time [TIME] what fixture is the object nearest to?`

Choices:

- 5 fixture choices total
- exactly 1 canonical correct choice (`correct_idx`)

Answer source:

- fixture from `mask_info.json` / track association for object `A` at query context

### 2.2 Relative location question

Template:

`[OBJECT A] was seen earlier in the video. At the current time [TIME] what is its position relative to [OBJECT B]?`

Choices (fixed order):

1. above
2. below
3. to the right
4. to the left
5. in front of
6. behind

Answer source:

- anchored 3D coordinates of `A` relative to `B`
- direction derived from axis-dominant decomposition with boundary tolerance (Section 6)

## 3. Definitions and Constraints

### 3.1 Time token format

Follow HD-EPIC VQA style, e.g.:

`<TIME 00:00:27.2 video 1>`

`video 1` maps to the first entry in `inputs` for this question object.

### 3.2 Object A naming policy

For now, use object names exactly as provided in annotations (`assoc_info.json` naming).

Note for future revision:

- annotation names may need normalization for natural language quality
- keep this as an explicit TODO, not silent behavior

### 3.3 Object B selection policy

`B` is the most centrally located visible object at time `t`.

Operational definition:

- collect all objects visible in the RGB frame at `t`
- project each candidate center to image coordinates
- choose object minimizing distance to image center
- if no visible object exists, do not generate relative question for that key frame

## 4. Required Data Inputs

Expected HD-EPIC sources used by this pipeline:

- `assoc_info.json`: object identities, tracks, object names
- `mask_info.json`: per-mask 3D location, frame number, fixture labels
- video stream (for frame indexing and optional visualization)
- calibration and trajectory metadata required by existing visibility/coordinate scripts

The OOS pipeline should reuse the exact logic from:

- `scripts/preprocessing/in_view_objects_with_viz.py`
- `scripts/preprocessing/relative_relation_with_viz.py`

## 5. JSON Output Schema

One JSON object per question.

Top-level key format:

`[questionclass]_[horizon]_[#]`

where:

- `questionclass` identifies family, recommended values:
  - `oos_abs_fixture_location`
  - `oos_rel_anchor_location`
- `horizon` is the out-of-sight horizon value `h` in seconds (or normalized token, e.g. `h2p0`)
- `#` is question index

### 5.1 Core fields (HD-EPIC-style)

Each question object should contain:

- `inputs`
- `question`
- `choices`
- `correct_idx`

Recommended metadata fields for reproducibility:

- `video_id`
- `query_time_sec`
- `horizon_sec`
- `object_a_assoc_id`
- `object_a_name`
- `object_b_assoc_id` (null for absolute questions)
- `object_b_name` (null for absolute questions)
- `question_class`
- `acceptable_idxs` (optional, for boundary-tolerant relative answers)
- `generation_info` (selected frame index, sampling fps, seed/version)

### 5.2 Example (absolute)

```json
{
  "oos_abs_fixture_location_h2p0_0": {
    "inputs": {
      "video 1": {
        "id": "P01-20240202-161948"
      }
    },
    "question": "plate was seen earlier in the video. At the current time <TIME 00:00:27.2 video 1> what fixture is the object nearest to?",
    "choices": ["counter", "sink", "stove", "island", "table"],
    "correct_idx": 3,
    "video_id": "P01-20240202-161948",
    "query_time_sec": 27.2,
    "horizon_sec": 2.0,
    "object_a_assoc_id": "assoc_17",
    "object_a_name": "plate",
    "object_b_assoc_id": null,
    "object_b_name": null,
    "question_class": "oos_abs_fixture_location"
  }
}
```

### 5.3 Example (relative with tolerance)

```json
{
  "oos_rel_anchor_location_h2p0_1": {
    "inputs": {
      "video 1": {
        "id": "P01-20240202-161948"
      }
    },
    "question": "plate was seen earlier in the video. At the current time <TIME 00:00:27.2 video 1> what is its position relative to cutting board?",
    "choices": [
      "above",
      "below",
      "to the right",
      "to the left",
      "in front of",
      "behind"
    ],
    "correct_idx": 2,
    "acceptable_idxs": [2, 4],
    "video_id": "P01-20240202-161948",
    "query_time_sec": 27.2,
    "horizon_sec": 2.0,
    "object_a_assoc_id": "assoc_17",
    "object_a_name": "plate",
    "object_b_assoc_id": "assoc_03",
    "object_b_name": "cutting board",
    "question_class": "oos_rel_anchor_location"
  }
}
```

## 6. Relative Direction Rule (Fully Specified)

Given anchored coordinates `d = p_A - p_B = (dx, dy, dz)` in `B`-centric camera-aligned frame:

- +x: right
- +y: up
- +z: in front

### 6.1 Primary label

Compute axis magnitudes:

- `ax = abs(dx)`
- `ay = abs(dy)`
- `az = abs(dz)`

Primary direction is sign on the dominant axis:

- if `ay` is largest: `above` (`dy > 0`) else `below`
- if `ax` is largest: `to the right` (`dx > 0`) else `to the left`
- if `az` is largest: `in front of` (`dz > 0`) else `behind`

### 6.2 Border tolerance (+/- 10 degrees)

If the vector is near a boundary between top-2 axes, allow both labels.

Practical criterion:

1. Let `m1 >= m2 >= m3` be sorted magnitudes among `(ax, ay, az)`.
2. Define angular gap from boundary in the 2D plane of top-2 axes:
   - `theta = arctan(m2 / m1)` in degrees
   - boundary between them is 45 degrees
   - near-boundary if `abs(45 - theta) <= 10`
3. If near-boundary, include both candidate directional labels in `acceptable_idxs`.

`correct_idx` is still required for compatibility and should be the dominant-axis label.

## 7. Absolute Fixture Answer Rule

For query time `t` and object `A`:

1. Resolve best mask observation around `t` using track logic.
2. Read fixture from `mask_info.json` entry for selected observation.
3. Set this fixture as correct answer.
4. Sample 4 distractors from fixture vocabulary observed in dataset split.

Distractor constraints:

- unique choices
- no duplicate of correct fixture

## 8. Key Frame and Object Selection Algorithm

Goal: choose candidate `(video, t, A)` tuples that satisfy OOS horizon and maximize diversity.

Inputs:

- horizon `h` (seconds)
- sampling rate for candidate timeline (default 2 fps)
- max questions per video

### 8.1 Build object visibility tracks

For each video:

1. Enumerate objects from `assoc_info.json`.
2. For sampled frames/timestamps, determine in-view vs out-of-view for each object using reused visibility logic.
3. Build itinerary per object: alternating in-view/out-of-view spans.

### 8.2 Rank objects by relocation activity

Compute relocation score per object (higher means object changes location more often), e.g.:

- number of distinct fixture transitions
- number of stable track segments with meaningfully different 3D centroids

Sort objects descending by score.

### 8.3 Select candidate OOS frames

For each object in ranked order:

1. Find OOS gaps with duration `>= h`.
2. For each valid gap, choose frame `t` as close as possible to `gap_start + h`.
3. Enforce stronger context rule for recall quality:
   - object out of frame for at least `2*h` around selected episode where applicable, so recall is not trivial from immediate visibility.
4. Prefer one frame per distinct visited location (fixture) before adding more from same location.

Stop when max question frames per video is reached.

## 9. End-to-End Generation Flow

For each selected `(video, t, A)`:

1. Determine anchor `B` (central visible object at `t`).
2. Generate absolute question if fixture label for `A` is available.
3. Generate relative question if `B` exists and relative coordinates are valid.
4. Append question objects to output JSON.

Skip policies:

- if no anchor `B`, skip relative only
- if fixture unresolved, skip absolute only
- if both fail, drop candidate

## 10. Module Responsibilities (Planned File Structure)

- `in_view_determination.py`
  - wrapper around existing in-view logic (same behavior as preprocessing script)
  - API to query visibility of objects at `(video, t)`

- `in_view_track_generator.py`
  - builds per-object visibility itineraries over sampled timeline
  - stores out-of-view spans and transitions

- `key_frame_generator.py`
  - ranks objects by relocation score
  - selects candidate `(t, A)` frames under horizon and max-question constraints

- `anchored_coords.py`
  - computes `A` and `B` anchored coordinates in shared local frame
  - exposes relative displacement vector

- `relative_answer_determ.py`
  - converts displacement vector to relative label(s)
  - applies 10 degree border tolerance and returns `correct_idx` + `acceptable_idxs`

- `abs_answer_determ.py`
  - resolves fixture for `A` at query context
  - samples distractors and returns 5-choice list + `correct_idx`

- `question_generator.py`
  - builds final question text with HD-EPIC time token style
  - emits schema-compliant JSON objects with stable IDs

## 11. Quality Control and Validation

Minimum checks before accepting generated questions:

1. Schema validation for every question object.
2. Verify object `A` is out of view at query time `t`.
3. Verify OOS duration satisfies configured `h`.
4. For relative questions, verify anchor `B` is visible and central at `t`.
5. For tolerance cases, verify `correct_idx` belongs to `acceptable_idxs`.

## 12. Known Limitations and TODOs

- Question wording is not finalized; current templates were drafted primarily to validate pipeline feasibility.
- Object naming currently mirrors raw annotation names; language normalization is deferred.
- Object visibility is currently determined from camera geometry only; occlusion is not handled. As a result, both the queried object and the reference object may be treated as “in view” even when actually occluded, which can produce invalid relative questions.
- Relative location questions may therefore anchor against objects that are technically visible by geometry but effectively hidden in enclosed spaces or behind other structures.
- Fixture answer options are not yet curated or normalized; near-duplicate labels such as counter002 and counter009 may both appear.
- Fixture distractor options are currently sampled from a global fixture vocabulary rather than constrained per video; it would likely be better to sample only fixtures that actually occur in the source video.
