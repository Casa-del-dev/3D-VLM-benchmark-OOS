from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Any
import json
import re

from in_view_determination import (
    DEFAULT_INTERMEDIATE_ROOT,
    choose_track_for_time,
    get_mask_from_track,
    load_json,
)


@dataclass(frozen=True)
class VisibilitySpan:
    start_sec: float
    end_sec: float
    in_view: bool


@dataclass(frozen=True)
class ObjectVisibilityTrack:
    assoc_id: str
    name: str
    sampled_times_sec: list[float]
    visibility_samples: list[bool]
    stable_visibility_samples: list[bool]
    status_samples: list[str]
    projected_pixel_samples: list[list[float] | None]
    camera_coordinate_samples: list[list[float] | None]
    frame_index_samples: list[int | None]
    fixture_samples: list[str | None]
    world_coordinate_samples: list[list[float] | None]
    last_visible_index_before_each_sample: list[int | None]
    spans: list[VisibilitySpan]


@dataclass(frozen=True)
class RelocationScore:
    assoc_id: str
    name: str
    fixture_transition_count: int
    centroid_shift_segment_count: int
    total_score: int


@dataclass(frozen=True)
class KeyFrameCandidate:
    video_id: str
    assoc_id: str
    object_name: str
    query_time_sec: float
    oos_span_start_sec: float
    oos_span_end_sec: float
    oos_duration_sec: float
    horizon_sec: float
    fixture_at_query: str | None
    relocation_score: int
    clip_start_time_sec: float
    clip_end_time_sec: float
    clip_duration_sec: float


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _normalize_object_name_for_ambiguity(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    # remove trailing digits, e.g. bowl2 -> bowl, black jar1 -> black jar
    s = re.sub(r"\d+$", "", s).strip()
    return s

def _build_ambiguous_assoc_ids_for_video(
    video_id: str,
    annotations_root: str | Path,
) -> set[str]:
    annotations_root = Path(annotations_root)
    assoc_info = load_json(annotations_root / "scene-and-object-movements" / "assoc_info.json")
    video_objects: dict[str, Any] = assoc_info.get(video_id, {})

    base_to_assoc_ids: dict[str, list[str]] = {}
    for assoc_id, obj in video_objects.items():
        raw_name = str(obj.get("name", assoc_id))
        base = _normalize_object_name_for_ambiguity(raw_name)
        if not base:
            continue
        base_to_assoc_ids.setdefault(base, []).append(str(assoc_id))

    ambiguous_assoc_ids: set[str] = set()
    for _, assoc_ids in base_to_assoc_ids.items():
        if len(assoc_ids) > 1:
            ambiguous_assoc_ids.update(assoc_ids)

    return ambiguous_assoc_ids

def load_precomputed_visibility_tracks(path: str | Path) -> dict[str, ObjectVisibilityTrack]:
    raw = _load_json(Path(path))
    object_tracks = raw.get("object_tracks", {})

    out: dict[str, ObjectVisibilityTrack] = {}
    for assoc_id, tr in object_tracks.items():
        sampled_times = [float(t) for t in tr["sampled_times_sec"]]
        visibility_samples = [bool(v) for v in tr.get("visibility_samples", [False] * len(sampled_times))]
        stable_visibility_samples = [
            bool(v) for v in tr.get("stable_visibility_samples", [False] * len(sampled_times))
        ]
        status_samples = [str(v) for v in tr.get("status_samples", ["unknown"] * len(sampled_times))]

        out[assoc_id] = ObjectVisibilityTrack(
            assoc_id=str(assoc_id),
            name=str(tr.get("name", assoc_id)),
            sampled_times_sec=sampled_times,
            visibility_samples=visibility_samples,
            stable_visibility_samples=stable_visibility_samples,
            status_samples=status_samples,
            projected_pixel_samples=[
                [float(x) for x in v] if v is not None else None
                for v in tr.get("projected_pixel_samples", [None] * len(sampled_times))
            ],
            camera_coordinate_samples=[
                [float(x) for x in v] if v is not None else None
                for v in tr.get("camera_coordinate_samples", [None] * len(sampled_times))
            ],
            frame_index_samples=[
                int(v) if v is not None else None
                for v in tr.get("frame_index_samples", [None] * len(sampled_times))
            ],
            fixture_samples=[
                str(v) if v is not None else None
                for v in tr.get("fixture_samples", [None] * len(sampled_times))
            ],
            world_coordinate_samples=[
                [float(x) for x in v] if v is not None else None
                for v in tr.get("world_coordinate_samples", [None] * len(sampled_times))
            ],
            last_visible_index_before_each_sample=[
                int(v) if v is not None else None
                for v in tr.get("last_visible_index_before_each_sample", [None] * len(sampled_times))
            ],
            spans=[
                VisibilitySpan(
                    start_sec=float(sp["start_sec"]),
                    end_sec=float(sp["end_sec"]),
                    in_view=bool(sp["in_view"]),
                )
                for sp in tr.get("spans", [])
            ],
        )
    return out


def _collect_masks_for_track(mask_info_video: dict[str, Any], track: dict[str, Any]) -> list[dict[str, Any]]:
    masks: list[dict[str, Any]] = []
    for mask_id in track.get("masks", []):
        entry = mask_info_video.get(mask_id)
        if entry is None:
            continue
        masks.append(
            {
                "mask_id": mask_id,
                "frame_number": int(entry["frame_number"]),
                "fixture": entry.get("fixture"),
                "3d_location": entry.get("3d_location"),
            }
        )
    masks.sort(key=lambda m: m["frame_number"])
    return masks


def _track_centroid(track_masks: list[dict[str, Any]]) -> list[float] | None:
    points = [m.get("3d_location") for m in track_masks if isinstance(m.get("3d_location"), list) and len(m.get("3d_location")) == 3]
    if not points:
        return None
    n = float(len(points))
    return [
        sum(float(p[0]) for p in points) / n,
        sum(float(p[1]) for p in points) / n,
        sum(float(p[2]) for p in points) / n,
    ]


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def compute_relocation_score(
    video_id: str,
    assoc_id: str,
    annotations_root: str | Path,
    centroid_shift_threshold_m: float = 0.15,
) -> RelocationScore:
    annotations_root = Path(annotations_root)
    assoc_info = load_json(annotations_root / "scene-and-object-movements" / "assoc_info.json")
    mask_info = load_json(annotations_root / "scene-and-object-movements" / "mask_info.json")

    video_objects: dict[str, Any] = assoc_info.get(video_id, {})
    mask_info_video: dict[str, Any] = mask_info.get(video_id, {})
    if assoc_id not in video_objects:
        raise KeyError(f"Object assoc_id {assoc_id} not found in video {video_id}")

    obj = video_objects[assoc_id]
    tracks = sorted(obj.get("tracks", []), key=lambda tr: float(tr["time_segment"][0]))

    fixture_sequence: list[str] = []
    centroids: list[list[float]] = []
    for tr in tracks:
        track_masks = _collect_masks_for_track(mask_info_video, tr)
        if track_masks:
            fixtures_here = [m["fixture"] for m in track_masks if m.get("fixture")]
            if fixtures_here:
                fixture_sequence.append(str(fixtures_here[-1]))
        centroid = _track_centroid(track_masks)
        if centroid is not None:
            centroids.append(centroid)

    fixture_transition_count = 0
    for i in range(1, len(fixture_sequence)):
        if fixture_sequence[i] != fixture_sequence[i - 1]:
            fixture_transition_count += 1

    centroid_shift_segment_count = 0
    for i in range(1, len(centroids)):
        if _euclidean_distance(centroids[i], centroids[i - 1]) >= centroid_shift_threshold_m:
            centroid_shift_segment_count += 1

    return RelocationScore(
        assoc_id=assoc_id,
        name=str(obj.get("name", assoc_id)),
        fixture_transition_count=fixture_transition_count,
        centroid_shift_segment_count=centroid_shift_segment_count,
        total_score=fixture_transition_count + centroid_shift_segment_count,
    )


def rank_objects_by_relocation(
    video_id: str,
    annotations_root: str | Path,
    centroid_shift_threshold_m: float = 0.15,
) -> list[RelocationScore]:
    annotations_root = Path(annotations_root)
    assoc_info = load_json(annotations_root / "scene-and-object-movements" / "assoc_info.json")
    if video_id not in assoc_info:
        raise KeyError(f"Video {video_id} not found in assoc_info.json")

    scores = [
        compute_relocation_score(
            video_id=video_id,
            assoc_id=assoc_id,
            annotations_root=annotations_root,
            centroid_shift_threshold_m=centroid_shift_threshold_m,
        )
        for assoc_id in assoc_info[video_id].keys()
    ]
    return sorted(scores, key=lambda s: (-s.total_score, -s.fixture_transition_count, -s.centroid_shift_segment_count, s.assoc_id))


def _select_time_for_oos_span(
    track: ObjectVisibilityTrack,
    span: VisibilitySpan,
    horizon_sec: float,
) -> float | None:
    target = span.start_sec + horizon_sec
    eligible_times = [
        t for t, v in zip(track.sampled_times_sec, track.visibility_samples)
        if (not v) and span.start_sec - 1e-9 <= t <= span.end_sec + 1e-9
    ]
    if not eligible_times:
        return None
    return min(eligible_times, key=lambda t: abs(t - target))


def _passes_stronger_context_rule(
    span: VisibilitySpan,
    query_time_sec: float,
    horizon_sec: float,
    video_start_sec: float,
    video_end_sec: float,
    step_sec: float,
) -> bool:
    left = max(video_start_sec, query_time_sec - horizon_sec)
    right = min(video_end_sec, query_time_sec + horizon_sec)
    window = right - left
    eps = max(1e-9, 0.5 * step_sec)

    full_window_available = (left > video_start_sec + eps) and (right < video_end_sec - eps)
    if full_window_available and window < (2.0 * horizon_sec - eps):
        return False

    return (left >= span.start_sec - eps) and (right <= span.end_sec + eps)


def _fixture_for_object_at_time(
    video_id: str,
    assoc_id: str,
    time_sec: float,
    annotations_root: str | Path,
) -> str | None:
    annotations_root = Path(annotations_root)
    assoc_info = load_json(annotations_root / "scene-and-object-movements" / "assoc_info.json")
    mask_info = load_json(annotations_root / "scene-and-object-movements" / "mask_info.json")

    video_objects: dict[str, Any] = assoc_info.get(video_id, {})
    if assoc_id not in video_objects:
        return None

    obj = video_objects[assoc_id]
    mask_info_video = mask_info.get(video_id, {})
    track, mode, _ = choose_track_for_time(obj.get("tracks", []), time_sec)
    if track is None or mode == "in_motion":
        return None

    pick = "latest" if mode == "past" else "first"
    mask = get_mask_from_track(mask_info_video, track, pick=pick)
    if mask is None:
        return None
    fixture = mask.get("fixture")
    if fixture is None:
        return None
    return str(fixture)


def _order_candidates_by_location_diversity(candidates: list[KeyFrameCandidate]) -> list[KeyFrameCandidate]:
    seen_fixtures: set[str | None] = set()
    first_pass: list[KeyFrameCandidate] = []
    second_pass: list[KeyFrameCandidate] = []
    for cand in candidates:
        if cand.fixture_at_query not in seen_fixtures:
            seen_fixtures.add(cand.fixture_at_query)
            first_pass.append(cand)
        else:
            second_pass.append(cand)
    return first_pass + second_pass


def _has_prior_visible_context(track: ObjectVisibilityTrack, query_time_sec: float) -> bool:
    for t, v in zip(track.sampled_times_sec, track.visibility_samples):
        if t >= query_time_sec:
            break
        if v:
            return True
    return False


def _has_prior_stable_visible_context(track: ObjectVisibilityTrack, query_time_sec: float) -> bool:
    for t, v in zip(track.sampled_times_sec, track.stable_visibility_samples):
        if t >= query_time_sec:
            break
        if v:
            return True
    return False


def _get_object_tracks(
    video_id: str,
    assoc_id: str,
    annotations_root: str | Path,
) -> list[dict[str, Any]]:
    annotations_root = Path(annotations_root)
    assoc_info = load_json(annotations_root / "scene-and-object-movements" / "assoc_info.json")
    video_objects: dict[str, Any] = assoc_info.get(video_id, {})
    obj = video_objects.get(assoc_id)
    if obj is None:
        return []
    return sorted(obj.get("tracks", []), key=lambda tr: float(tr["time_segment"][0]))


def _eligible_prior_tracks(
    video_id: str,
    assoc_id: str,
    query_time_sec: float,
    annotations_root: str | Path,
) -> list[dict[str, Any]]:
    tracks = _get_object_tracks(video_id=video_id, assoc_id=assoc_id, annotations_root=annotations_root)
    return [tr for tr in tracks if float(tr["time_segment"][1]) <= query_time_sec + 1e-9]


def _stable_start_after_last_past_track(
    video_id: str,
    assoc_id: str,
    span_start_sec: float,
    annotations_root: str | Path,
    fps_for_frame_lookup: float = 30.0,
) -> float | None:
    annotations_root = Path(annotations_root)
    assoc_info = load_json(annotations_root / "scene-and-object-movements" / "assoc_info.json")
    mask_info = load_json(annotations_root / "scene-and-object-movements" / "mask_info.json")

    video_objects: dict[str, Any] = assoc_info.get(video_id, {})
    obj = video_objects.get(assoc_id)
    if obj is None:
        return None

    last_past_track: dict[str, Any] | None = None
    for tr in sorted(obj.get("tracks", []), key=lambda tr: float(tr["time_segment"][0])):
        track_end_sec = float(tr["time_segment"][1])
        if track_end_sec <= span_start_sec + 1e-9:
            last_past_track = tr
        else:
            break

    if last_past_track is None:
        return None

    track_end_sec = float(last_past_track["time_segment"][1])
    mask_info_video: dict[str, Any] = mask_info.get(video_id, {})
    track_masks = _collect_masks_for_track(mask_info_video, last_past_track)
    if not track_masks:
        return track_end_sec

    latest_frame_number = max(int(m["frame_number"]) for m in track_masks)
    latest_mask_time_sec = latest_frame_number / float(fps_for_frame_lookup)
    return max(track_end_sec, latest_mask_time_sec)


def _last_stable_visible_time_before(track: ObjectVisibilityTrack, query_time_sec: float) -> float | None:
    last_t = None
    for t, v in zip(track.sampled_times_sec, track.stable_visibility_samples):
        if t >= query_time_sec:
            break
        if v:
            last_t = t
    return last_t

def _state_at_time(track: ObjectVisibilityTrack, query_time_sec: float, eps: float = 1e-6) -> dict[str, Any] | None:
    """
    Return this object's state at the sampled time matching query_time_sec.
    Falls back to nearest sample if no exact match exists.
    """
    times = track.sampled_times_sec
    if not times:
        return None

    best_idx = min(range(len(times)), key=lambda i: abs(float(times[i]) - float(query_time_sec)))
    if abs(float(times[best_idx]) - float(query_time_sec)) > max(eps, 0.51 * (times[1] - times[0]) if len(times) > 1 else eps):
        return None

    return {
        "assoc_id": track.assoc_id,
        "name": track.name,
        "time_sec": float(times[best_idx]),
        "status": track.status_samples[best_idx] if best_idx < len(track.status_samples) else None,
        "projected_pixel": track.projected_pixel_samples[best_idx] if best_idx < len(track.projected_pixel_samples) else None,
        "world_coordinates": track.world_coordinate_samples[best_idx] if best_idx < len(track.world_coordinate_samples) else None,
    }


def _eligible_anchor_objects_at_time(
    tracks: dict[str, ObjectVisibilityTrack],
    query_time_sec: float,
    target_assoc_id: str,
) -> list[dict[str, Any]]:
    """
    Return all non-target objects that are eligible to serve as anchors at query_time_sec.
    Anchor rule matches your step-4 selector:
      - not the target object
      - status in {"in_view", "observed_visible_in_open_fixture"}
      - projected_pixel is not None
      - world_coordinates is not None
    """
    eligible: list[dict[str, Any]] = []
    for assoc_id, tr in tracks.items():
        if str(assoc_id) == str(target_assoc_id):
            continue

        st = _state_at_time(tr, query_time_sec)
        if st is None:
            continue

        if st["status"] not in {"in_view", "observed_visible_in_open_fixture"}:
            continue
        if st["projected_pixel"] is None:
            continue
        if st["world_coordinates"] is None:
            continue

        eligible.append(st)
    return eligible


def _has_valid_anchor_at_time(
    tracks: dict[str, ObjectVisibilityTrack],
    query_time_sec: float,
    target_assoc_id: str,
) -> bool:
    return len(_eligible_anchor_objects_at_time(tracks, query_time_sec, target_assoc_id)) > 0

def generate_key_frames_for_video(
    video_id: str,
    annotations_root: str | Path,
    horizon_sec: float,
    max_questions_per_video: int,
    sampling_fps: float = 2.0,
    fps_for_frame_lookup: float = 30.0,
    intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT,
    centroid_shift_threshold_m: float = 0.15,
    start_time_sec: float | None = None,
    end_time_sec: float | None = None,
    random_seed: int = 42,
    pre_context_sec: float = 2.0,
    max_random_clip_margin_sec: float = 20.0,
    precomputed_tracks: dict[str, ObjectVisibilityTrack] | None = None,
    precomputed_tracks_json: str | Path | None = None,
) -> list[KeyFrameCandidate]:
    if horizon_sec <= 0:
        raise ValueError("horizon_sec must be > 0")
    if max_questions_per_video <= 0:
        return []

    if precomputed_tracks is None:
        if precomputed_tracks_json is None:
            raise ValueError(
                "generate_key_frames_for_video now requires precomputed visibility tracks. "
                "Pass precomputed_tracks or precomputed_tracks_json."
            )
        tracks = load_precomputed_visibility_tracks(precomputed_tracks_json)
    else:
        tracks = precomputed_tracks

    if not tracks:
        return []

    rng = random.Random((video_id, random_seed).__repr__())

    ranked = rank_objects_by_relocation(
        video_id=video_id,
        annotations_root=annotations_root,
        centroid_shift_threshold_m=centroid_shift_threshold_m,
    )

    # avoid ambiguious object (e.g. bowl 1 and bowl 2)
    ambiguous_assoc_ids = _build_ambiguous_assoc_ids_for_video(
        video_id=video_id,
        annotations_root=annotations_root,
    )
    any_track = next(iter(tracks.values()))
    video_start_sec = min(any_track.sampled_times_sec)
    video_end_sec = max(any_track.sampled_times_sec)
    step_sec = 1.0 / sampling_fps

    selected: list[KeyFrameCandidate] = []
    for score in ranked:
        # skip objects with ambiguous names to avoid confusion in the generated questions
        if str(score.assoc_id) in ambiguous_assoc_ids:
            print(
                "[SKIP_AMBIGUOUS_OBJECT_NAME]",
                {
                    "video_id": video_id,
                    "assoc_id": score.assoc_id,
                    "object_name": score.name,
                },
            )
            continue
        if len(selected) >= max_questions_per_video:
            break

        track = tracks.get(score.assoc_id)
        if track is None:
            continue

        object_tracks = _get_object_tracks(video_id=video_id, assoc_id=score.assoc_id, annotations_root=annotations_root)
        if len(object_tracks) < 1:
            continue

        object_candidates: list[KeyFrameCandidate] = []
        for span in track.spans:
            if span.in_view:
                continue

            span_duration = span.end_sec - span.start_sec
            if span_duration + 1e-9 < horizon_sec:
                continue

            stable_start_sec = _stable_start_after_last_past_track(
                video_id=video_id,
                assoc_id=score.assoc_id,
                span_start_sec=span.start_sec,
                annotations_root=annotations_root,
                fps_for_frame_lookup=fps_for_frame_lookup,
            )
            if stable_start_sec is None:
                continue

            if len(_eligible_prior_tracks(
                video_id=video_id,
                assoc_id=score.assoc_id,
                query_time_sec=span.start_sec,
                annotations_root=annotations_root,
            )) == 0:
                continue

            usable_start_sec = max(span.start_sec, stable_start_sec)
            usable_end_sec = span.end_sec
            usable_duration_sec = usable_end_sec - usable_start_sec
            if usable_duration_sec + 1e-9 < horizon_sec:
                continue

            effective_span = VisibilitySpan(start_sec=usable_start_sec, end_sec=usable_end_sec, in_view=False)
            t_sec = _select_time_for_oos_span(track, effective_span, horizon_sec)
            if t_sec is None:
                continue

            last_visible_time_sec = _last_stable_visible_time_before(track, t_sec)
            if last_visible_time_sec is None:
                continue

            clip_end_time_sec = t_sec
            clip_start_time_sec = max(video_start_sec, last_visible_time_sec - pre_context_sec)
            clip_duration_sec = clip_end_time_sec - clip_start_time_sec

            if not _passes_stronger_context_rule(
                span=span,
                query_time_sec=t_sec,
                horizon_sec=horizon_sec,
                video_start_sec=video_start_sec,
                video_end_sec=video_end_sec,
                step_sec=step_sec,
            ):
                continue

            if not _has_prior_visible_context(track, t_sec):
                continue

            if not _has_prior_stable_visible_context(track, t_sec):
                continue

            if not (clip_start_time_sec <= last_visible_time_sec < t_sec):
                continue

            fixture = _fixture_for_object_at_time(video_id, score.assoc_id, t_sec, annotations_root)
            if not _has_valid_anchor_at_time(
                tracks=tracks,
                query_time_sec=t_sec,
                target_assoc_id=score.assoc_id,
            ):
                print(
                    "[SKIP_NO_ANCHOR_AT_QUERY_TIME]",
                    {
                        "video_id": video_id,
                        "assoc_id": score.assoc_id,
                        "object_name": score.name,
                        "query_time_sec": t_sec,
                        "span_start_sec": span.start_sec,
                        "span_end_sec": span.end_sec,
                    },
                )
                continue

            fixture = _fixture_for_object_at_time(video_id, score.assoc_id, t_sec, annotations_root)
            object_candidates.append(
                KeyFrameCandidate(
                    video_id=video_id,
                    assoc_id=score.assoc_id,
                    object_name=score.name,
                    query_time_sec=t_sec,
                    oos_span_start_sec=span.start_sec,
                    oos_span_end_sec=span.end_sec,
                    oos_duration_sec=span_duration,
                    horizon_sec=horizon_sec,
                    fixture_at_query=fixture,
                    relocation_score=score.total_score,
                    clip_start_time_sec=clip_start_time_sec,
                    clip_end_time_sec=clip_end_time_sec,
                    clip_duration_sec=clip_duration_sec,
                )
            )

        object_candidates.sort(key=lambda c: c.query_time_sec)
        object_candidates = _order_candidates_by_location_diversity(object_candidates)
        for cand in object_candidates:
            if len(selected) >= max_questions_per_video:
                break
            selected.append(cand)

    return selected


def generate_key_frames_for_videos(
    video_ids: list[str],
    annotations_root: str | Path,
    horizon_sec: float,
    max_questions_per_video: int,
    sampling_fps: float = 2.0,
    fps_for_frame_lookup: float = 30.0,
    intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT,
    centroid_shift_threshold_m: float = 0.15,
    start_time_sec: float | None = None,
    end_time_sec: float | None = None,
    random_seed: int = 42,
    max_random_clip_margin_sec: float = 20.0,
    pre_context_sec: float = 2.0,
    precomputed_tracks_by_video: dict[str, dict[str, ObjectVisibilityTrack]] | None = None,
    precomputed_tracks_json_by_video: dict[str, str | Path] | None = None,
) -> dict[str, list[KeyFrameCandidate]]:
    out: dict[str, list[KeyFrameCandidate]] = {}
    for video_id in video_ids:
        tracks = None
        tracks_json = None

        if precomputed_tracks_by_video is not None:
            tracks = precomputed_tracks_by_video.get(video_id)
        if precomputed_tracks_json_by_video is not None:
            tracks_json = precomputed_tracks_json_by_video.get(video_id)

        out[video_id] = generate_key_frames_for_video(
            video_id=video_id,
            annotations_root=annotations_root,
            horizon_sec=horizon_sec,
            max_questions_per_video=max_questions_per_video,
            sampling_fps=sampling_fps,
            fps_for_frame_lookup=fps_for_frame_lookup,
            intermediate_root=intermediate_root,
            centroid_shift_threshold_m=centroid_shift_threshold_m,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
            random_seed=random_seed,
            max_random_clip_margin_sec=max_random_clip_margin_sec,
            pre_context_sec=pre_context_sec,
            precomputed_tracks=tracks,
            precomputed_tracks_json=tracks_json,
        )
    return out


def key_frames_to_dict(candidates: list[KeyFrameCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "video_id": c.video_id,
            "assoc_id": c.assoc_id,
            "object_name": c.object_name,
            "query_time_sec": c.query_time_sec,
            "oos_span_start_sec": c.oos_span_start_sec,
            "oos_span_end_sec": c.oos_span_end_sec,
            "oos_duration_sec": c.oos_duration_sec,
            "horizon_sec": c.horizon_sec,
            "fixture_at_query": c.fixture_at_query,
            "relocation_score": c.relocation_score,
            "clip_start_time_sec": c.clip_start_time_sec,
            "clip_end_time_sec": c.clip_end_time_sec,
            "clip_duration_sec": c.clip_duration_sec,
        }
        for c in candidates
    ]
