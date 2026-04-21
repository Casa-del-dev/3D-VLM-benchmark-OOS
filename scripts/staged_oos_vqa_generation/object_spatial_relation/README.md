# Staged OOS VQA Pipeline

TThis module generates staged Visual Question Answering (VQA) data for Out-of-Sight (OOS) scenarios, covering object–camera relations, object–object relations, and object–object distances, from video and annotations using a 4-stage pipeline:

1. **Generate occlusion aware visibility tracks**
2. **Merge in-view tracks & visibility tracks**
3. **Generate staged VQA questions**
4. **Visualize generated questions (optional)**

Example generated VQA structure
```
{
  "video_id": "...",
  "object_a": "...",
  "query_time_sec": 123.0,

  "incremental_steps": [
    {
      "step": "1",
      "type": "visibility",
      "question": "Is object A visible?",
      "choices": ["Yes", "No"],
      "answer": "No"
    },
    {
      "step": "2",
      "type": "last_visible",
      "question": "When and where was it last seen?",
      "answer": {
        "time": hh:mm:ss,
        "pixel": [x, y]
      }
    },
    {
      "step": "3",
      "type": "fixture",
      "question": "Which fixture is closest to its last position?",
      "choices": [...],
      "answer": "counter"
    }
  ],

  "branches": {
    "post_step3": [
      {
        "step": "4a",
        "type": "object_camera_relation",
        "question": "Where is object A relative to the camera?",
        "choices": ["front-left", "front-right", "back-left", "back-right"],
        "answer": "front-left"
      },
      {
        "step": "4b",
        "type": "object_object_relation",
        "question": "Where is object A relative to object B?",
        "anchor_object": "object B",
        "choices": ["front-left", "front-right", "back-left", "back-right"],
        "answer": "back-right"
      },
      {
        "step": "4c",
        "type": "object_object_distance",
        "question": "How far is object A from object B?",
        "anchor_object": "object B",
        "choices": ["very close", "close", "medium", "far"],
        "answer": "near"
      }
    ]
  }
}
```
---
## 1 Generate Visibility Tracks

First, run the visibility track generation script:

```bash
python -m scripts.visibility_track.generate_visibility_track --video P04-20240413-142619
```

This will produce several intermediate outputs, including these 2:

* `in_view_tracks.jsonl` — in-view / out-of-view
* `visibility_track.jsonl` — refined visibility states (including occlusion and motion)



---
## 2 Merge In-view tracks & Visibility Tracks
### 2.1 Run
```bash
VIDEO_ID=P04-20240413-142619

python scripts/staged_oos_vqa_generation/object_spatial_relation/prepare_tracks_for_generation_via_merging.py \
  --video_id $VIDEO_ID \
  --in_view_jsonl outputs/visibility_track/$VIDEO_ID/in_view_tracks.jsonl \
  --visibility_jsonl outputs/visibility_track/$VIDEO_ID/visibility_track.jsonl \
  --output_json scripts/staged_oos_vqa_generation/object_spatial_relation/outputs/merged_tracks/P04/P04-20240413-142619_merged_visibility_track.json
```
Output:
```
merged_visibility_track.json
```

Example:

```json
{
  "sampled_times_sec": [...],
  "status_samples": [...],
  "visibility_samples": [...],
  "stable_visibility_samples": [...],
  "last_visible_index_before_each_sample": [...]
}
```

---

### 2.2 Status definitation logic when merging

```python
VISIBLE_STATUSES = {
    "in_view",
    "observed_visible_in_open_fixture",
    "in_motion",
}

STABLE_VISIBLE_STATUSES = {
    "in_view",
    "observed_visible_in_open_fixture",
}
```
| Status                           | Visible | Stable |
| -------------------------------- | ------- | ------ |
| in_view                          | ✅       | ✅      |
| observed_visible_in_open_fixture | ✅       | ✅      |
| in_motion                        | ✅       | ❌      |
---

## 3 VQA Generation

### 3.1 Run
```bash
python scripts/staged_oos_vqa_generation/object_spatial_relation/staged_oos_question_generator.py --config scripts/staged_oos_vqa_generation/object_spatial_relation/staged_oos_question_generator_config.yaml
```


### 3.2 Key Design

#### 3.2.1 Shared incremental structure
##### 3.2.1.1 Step 1 — Current Visibility

If the object is considered visible (i.e., `in_view`, `observed_visible_in_open_fixture`, or `in_motion`), we ask:

> *Is the object visible at the current time?*

---

##### 3.2.1.2 Step 2 — Last Visible State

We identify the last time the object was **stably visible** using:
- `last_visible_index_before_each_sample`
- `stable_visibility_samples`

These 2 fields are precomputed in the merged file.

**Fallback rule:**
- With the above 2 fields computed, we can apply a fallback rule: If the last visible state is labeled as `in_motion`, the entire trajectory is **skipped**, since the location is not reliable.

---

##### 3.2.1.3 Step 3 — Spatial Reasoning (Fixture)

We ask:

> *Which nearby fixture or landmark is closest to the object at its last visible location?*

---
#### 3.2.2 Parallel branches after step 3
##### 3.2.2.1 Step 4a — Object-Camera Relative Direction

Camera coordinates are already precomputed for each sampled frame in the in_view tracks (also stored in the merged file). These are used to determine the object's position relative to the camera (e.g., front-left, back-right).

**Robustness rule:**
- If the object lies too close to a decision boundary (e.g., `x ≈ 0` or `z ≈ 0`), the trajectory is **skipped** to avoid ambiguity.

##### 3.2.2.2 Step 4b — Object-Object Relative Direction
The anchored object is currently visible in the frame, while the target object is not.

##### 3.2.2.3 Step 4c — Object-Object Distance
very close : <0.5m
close : 0.5-1m
medium : 1-2 m
far : > 2m


## 4 🎨 Visualization

### 4.1 Run

```bash
python scripts/staged_oos_vqa_generation/object_spatial_relation/visualize_staged_oos_vqa.py \
  --questions outputs/generated_vqa/P04-20240413-142619_vqa.json \
  --video data/HD-EPIC/Videos/P04/P04-20240413-142619.mp4 \
  --output_dir scripts/staged_oos_vqa_generation/object_spatial_relation/outputs/visualizations
```

---

