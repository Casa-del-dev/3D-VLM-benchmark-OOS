from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import bisect
import json
import math
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import key_frame_generator as kfg
from abs_answer_determ import build_fixture_vocabulary, determine_absolute_answer, semantic_fixture_name
from in_view_determination import (
    DEFAULT_INTERMEDIATE_ROOT,
    determine_in_view_objects,
    load_frame_context,
    choose_track_for_time,
    get_mask_from_track,
    load_json,
    VideoCache,
    transform_point,
    project_fisheye624,
)
from key_frame_generator import ObjectVisibilityTrack, VisibilitySpan, KeyFrameCandidate
from anchored_coords import relation_from_world_points
from relative_answer_determ import determine_relative_answer_from_vector, determine_relative_answer_from_vector


@dataclass(frozen=True)
class BenchmarkConfig:
    annotations_root: Path
    video_ids: list[str]
    sampling_fps: float = 2.0
    fps_for_frame_lookup: float = 30.0
    out_of_sight_horizon_sec: float = 2.0
    max_questions_per_video: int = 20
    absolute_num_choices: int = 5
    random_seed: int = 42
    intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT
    output_json: Path | None = None
    visibility_tracks_json_by_video: dict[str, Path] | None = None
    pre_context_sec: float = 2.0
    raw_video_width: float | None = None
    raw_video_height: float | None = None
    last_placement_source: str = "raw_tracks"  # or "merged_tracks"


@dataclass(frozen=True)
class LastVisibleInfo:
    sampled_time_sec: float
    projected_pixel: list[float] | None
    camera_coordinates: list[float] | None
    frame_index: int | None
    status: str
    fixture: str | None = None
    world_coordinates: list[float] | None = None
    reference_source: str | None = None


CAMERA_QUADRANT_CHOICES = [
    "Front-left",
    "Front-right",
    "Back-left",
    "Back-right",
]

DISTANCE_CHOICES = [
    "very close",
    "close",
    "medium",
    "far",
]


class VisibilityTrackStore:
    def __init__(self, raw_by_video: dict[str, dict[str, Any]] | None):
        self.raw_by_video = raw_by_video or {}
        self._parsed_tracks_cache: dict[str, dict[str, ObjectVisibilityTrack]] = {}

    def has_video(self, video_id: str) -> bool:
        video = self.raw_by_video.get(video_id)
        return bool(video and video.get("object_tracks") is not None)

    def get_track_dict(self, video_id: str) -> dict[str, Any]:
        video = self.raw_by_video.get(video_id, {})
        return video.get("object_tracks", {})

    @staticmethod
    def _times_index(times: list[Any], time_sec: float) -> int | None:
        idx = bisect.bisect_left(times, time_sec)
        if idx >= len(times) or abs(float(times[idx]) - float(time_sec)) > 1e-6:
            return None
        return idx

    @staticmethod
    def _list_value(seq: list[Any] | None, idx: int) -> Any:
        if seq is None or idx < 0 or idx >= len(seq):
            return None
        return seq[idx]

    def get_object_tracks(self, video_id: str) -> dict[str, ObjectVisibilityTrack]:
        cached = self._parsed_tracks_cache.get(video_id)
        if cached is not None:
            return cached

        out: dict[str, ObjectVisibilityTrack] = {}
        for assoc_id, tr in self.get_track_dict(video_id).items():
            sampled_times = [float(t) for t in tr["sampled_times_sec"]]
            visibility_samples = [
                bool(v) for v in tr.get("visibility_samples", [False] * len(sampled_times))
            ]

            out[assoc_id] = ObjectVisibilityTrack(
                assoc_id=assoc_id,
                name=str(tr["name"]),
                sampled_times_sec=sampled_times,
                visibility_samples=visibility_samples,
                stable_visibility_samples=[
                    bool(v)
                    for v in tr.get("stable_visibility_samples", [False] * len(sampled_times))
                ],
                status_samples=[
                    str(v)
                    for v in tr.get(
                        "status_samples",
                        ["visible" if bool(v) else "not_visible" for v in visibility_samples],
                    )
                ],
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

        self._parsed_tracks_cache[video_id] = out
        return out

    def get_states_by_assoc_id(self, video_id: str, time_sec: float) -> dict[str, dict[str, Any]]:
        tracks = self.get_track_dict(video_id)
        if not tracks:
            return {}

        result: dict[str, dict[str, Any]] = {}
        for assoc_id, tr in tracks.items():
            times = tr.get("sampled_times_sec", [])
            idx = self._times_index(times, time_sec)
            if idx is None:
                continue

            visibility_samples = tr.get("visibility_samples", [])
            stable_visibility_samples = tr.get("stable_visibility_samples", [False] * len(times))
            status_samples = tr.get("status_samples", [])
            projected_pixel_samples = tr.get("projected_pixel_samples", [])
            camera_coordinate_samples = tr.get("camera_coordinate_samples", [])
            frame_index_samples = tr.get("frame_index_samples", [])
            fixture_samples = tr.get("fixture_samples", [])
            world_coordinate_samples = tr.get("world_coordinate_samples", [])

            is_visible = bool(self._list_value(visibility_samples, idx))
            is_stably_visible = bool(self._list_value(stable_visibility_samples, idx))
            status = self._list_value(status_samples, idx)
            if status is None:
                status = "visible" if is_visible else "not_visible"

            result[assoc_id] = {
                "assoc_id": assoc_id,
                "name": tr.get("name"),
                "status": str(status),
                "is_visible": is_visible,
                "is_stably_visible": is_stably_visible,
                "projected_pixel": self._list_value(projected_pixel_samples, idx),
                "camera_coordinates": self._list_value(camera_coordinate_samples, idx),
                "frame_number": self._list_value(frame_index_samples, idx),
                "fixture": self._list_value(fixture_samples, idx),
                "world_coordinates": self._list_value(world_coordinate_samples, idx),
            }
        return result

    def find_last_visible_before(
        self,
        video_id: str,
        assoc_id: str,
        query_time_sec: float,
        clip_start_time_sec: float,
    ) -> LastVisibleInfo | None:
        track = self.get_track_dict(video_id).get(assoc_id)
        if track is None:
            return None

        times = track.get("sampled_times_sec", [])
        if not times:
            return None

        idx = self._times_index(times, query_time_sec)
        if idx is None:
            idx = bisect.bisect_left(times, query_time_sec)
            if idx >= len(times):
                idx = len(times) - 1

        last_visible_idxs = track.get("last_visible_index_before_each_sample")
        if last_visible_idxs is not None and idx < len(last_visible_idxs):
            last_idx = last_visible_idxs[idx]
            if last_idx is not None and float(times[last_idx]) >= clip_start_time_sec - 1e-9:
                return LastVisibleInfo(
                    sampled_time_sec=float(times[last_idx]),
                    projected_pixel=self._list_value(track.get("projected_pixel_samples", []), last_idx),
                    camera_coordinates=self._list_value(track.get("camera_coordinate_samples", []), last_idx),
                    frame_index=self._list_value(track.get("frame_index_samples", []), last_idx),
                    status=str(self._list_value(track.get("status_samples", []), last_idx) or "ok"),
                    fixture=self._list_value(track.get("fixture_samples", []), last_idx),
                    world_coordinates=self._list_value(track.get("world_coordinate_samples", []), last_idx),
                    reference_source="precomputed_visibility_track",
                )

        stable_visibility_samples = track.get("stable_visibility_samples", [])
        idx = min(idx - 1 if self._times_index(times, query_time_sec) is not None else idx, len(times) - 1)
        while idx >= 0 and float(times[idx]) >= clip_start_time_sec - 1e-9:
            if bool(self._list_value(stable_visibility_samples, idx)):
                return LastVisibleInfo(
                    sampled_time_sec=float(times[idx]),
                    projected_pixel=self._list_value(track.get("projected_pixel_samples", []), idx),
                    camera_coordinates=self._list_value(track.get("camera_coordinate_samples", []), idx),
                    frame_index=self._list_value(track.get("frame_index_samples", []), idx),
                    status=str(self._list_value(track.get("status_samples", []), idx) or "ok"),
                    fixture=self._list_value(track.get("fixture_samples", []), idx),
                    world_coordinates=self._list_value(track.get("world_coordinate_samples", []), idx),
                    reference_source="precomputed_visibility_track",
                )
            idx -= 1
        return None
    def find_last_placement_before_from_merged(
        self,
        video_id: str,
        assoc_id: str,
        query_time_sec: float,
        clip_start_time_sec: float,
    ) -> LastVisibleInfo | None:
        track = self.get_track_dict(video_id).get(assoc_id)
        if track is None:
            return None

        times = track.get("sampled_times_sec", [])
        statuses = track.get("status_samples", [])
        stable_visibility_samples = track.get("stable_visibility_samples", [])
        if not times:
            return None

        idx = self._times_index(times, query_time_sec)
        if idx is None:
            idx = bisect.bisect_left(times, query_time_sec)
            if idx >= len(times):
                idx = len(times) - 1

        # We want the ending position of the last completed track before query time.
        # In sampled merged form, approximate this as the last stable non-in_motion sample
        # before query time.
        idx = min(idx - 1 if self._times_index(times, query_time_sec) is not None else idx, len(times) - 1)

        while idx >= 0 and float(times[idx]) >= clip_start_time_sec - 1e-9:
            status = str(self._list_value(statuses, idx) or "")
            is_stable = bool(self._list_value(stable_visibility_samples, idx))
            if is_stable and status != "in_motion":
                return LastVisibleInfo(
                    sampled_time_sec=float(times[idx]),
                    projected_pixel=self._list_value(track.get("projected_pixel_samples", []), idx),
                    camera_coordinates=self._list_value(track.get("camera_coordinate_samples", []), idx),
                    frame_index=self._list_value(track.get("frame_index_samples", []), idx),
                    status=status or "stable_sample",
                    fixture=self._list_value(track.get("fixture_samples", []), idx),
                    world_coordinates=self._list_value(track.get("world_coordinate_samples", []), idx),
                    reference_source="merged_tracks_last_stable_sample_before_query",
                )
            idx -= 1

        return None


class RuntimeCaches:
    def __init__(self, cfg: BenchmarkConfig, visibility_store: VisibilityTrackStore | None = None):
        self.cfg = cfg
        self.visibility_store = visibility_store or VisibilityTrackStore(None)
        self._video_cache_by_video: dict[str, VideoCache] = {}

    @staticmethod
    def _norm_time(time_sec: float) -> float:
        return round(float(time_sec), 6)

    @lru_cache(maxsize=20000)
    def live_states_by_assoc_id(self, video_id: str, time_sec: float) -> dict[str, Any]:
        time_sec = self._norm_time(time_sec)
        states = determine_in_view_objects(
            video_id=video_id,
            time_sec=time_sec,
            annotations_root=self.cfg.annotations_root,
            fps=self.cfg.fps_for_frame_lookup,
            intermediate_root=self.cfg.intermediate_root,
        )
        return {s.assoc_id: s for s in states}

    @lru_cache(maxsize=20000)
    def states_by_assoc_id(self, video_id: str, time_sec: float) -> dict[str, Any]:
        time_sec = self._norm_time(time_sec)

        if self.visibility_store.has_video(video_id):
            states = self.visibility_store.get_states_by_assoc_id(video_id, time_sec)
            if states:
                return states

        return self.live_states_by_assoc_id(video_id, time_sec)
    def video_cache(self, video_id: str) -> VideoCache:
        cached = self._video_cache_by_video.get(video_id)
        if cached is not None:
            return cached

        built = VideoCache.build(
            video_id=video_id,
            annotations_root=self.cfg.annotations_root,
            intermediate_root=self.cfg.intermediate_root,
        )
        self._video_cache_by_video[video_id] = built
        return built


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load the benchmark config. Install it with: pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_horizon_token(horizon_sec: float) -> str:
    return f"h{horizon_sec:.1f}".replace(".", "p")


def _format_time_hms_1dp(time_sec: float) -> str:
    t = max(0.0, float(time_sec))
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = t - (hours * 3600 + minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"


def _time_token(time_sec: float, input_key: str = "video 1") -> str:
    return f"<TIME {_format_time_hms_1dp(time_sec)} {input_key}>"


def _is_visible_state(state: Any) -> bool:
    if state is None:
        return False
    if isinstance(state, dict):
        return bool(state.get("is_visible"))
    return bool((state.status == "ok" and state.in_view) or state.status == "in_motion")


def _state_attr(state: Any, name: str, default: Any = None) -> Any:
    if state is None:
        return default
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _find_last_visible_info(
    candidate: KeyFrameCandidate,
    cfg: BenchmarkConfig,
    caches: RuntimeCaches,
) -> LastVisibleInfo | None:
    if caches.visibility_store.has_video(candidate.video_id):
        last_visible = caches.visibility_store.find_last_visible_before(
            video_id=candidate.video_id,
            assoc_id=candidate.assoc_id,
            query_time_sec=candidate.query_time_sec,
            clip_start_time_sec=candidate.clip_start_time_sec,
        )
        if last_visible is not None:
            return last_visible

    step = 1.0 / cfg.sampling_fps
    t = candidate.query_time_sec - step
    while t >= candidate.clip_start_time_sec - 1e-9:
        state = caches.live_states_by_assoc_id(candidate.video_id, round(t, 6)).get(candidate.assoc_id)
        is_stable_visible = bool(
            (not isinstance(state, dict) and state is not None and state.status == "ok" and bool(state.in_view))
            or (isinstance(state, dict) and bool(state.get("is_stably_visible")))
        )
        if is_stable_visible:
            projected_pixel = _state_attr(state, "projected_pixel")
            frame_number = _state_attr(state, "frame_number")
            camera_coordinates = _state_attr(state, "camera_coordinates")
            return LastVisibleInfo(
                sampled_time_sec=float(round(t, 6)),
                projected_pixel=[float(v) for v in projected_pixel] if projected_pixel is not None else None,
                camera_coordinates=[float(v) for v in camera_coordinates] if camera_coordinates is not None else None,
                frame_index=int(frame_number) if frame_number is not None else None,
                status=str(_state_attr(state, "status", "ok")),
                fixture=_state_attr(state, "fixture"),
                world_coordinates=_state_attr(state, "world_coordinates"),
                reference_source="live_visibility_scan",
            )
        t -= step
    return None

def _project_world_point_to_query_frame(
    *,
    video_id: str,
    query_time_sec: float,
    world_coordinates: list[float] | None,
    cfg: BenchmarkConfig,
    caches: RuntimeCaches,
) -> tuple[list[float] | None, list[float] | None]:
    if world_coordinates is None:
        return None, None

    cache = caches.video_cache(video_id)
    ctx = load_frame_context(
        video_id=video_id,
        time_sec=query_time_sec,
        annotations_root=cfg.annotations_root,
        fps=cfg.fps_for_frame_lookup,
        intermediate_root=cfg.intermediate_root,
        cache=cache,
    )

    cam_xyz = transform_point(ctx.T_camera_world, world_coordinates)
    pixel_xy, _, valid = (
        project_fisheye624(cam_xyz, ctx.projection_params)
        if ctx.model_name == "CameraModelType.FISHEYE624"
        else (None, cam_xyz[2], False)
    )
    projected = pixel_xy if valid else None
    return projected, [float(v) for v in cam_xyz]

def _project_world_point_at_time(
    *,
    video_id: str,
    time_sec: float,
    world_coordinates: list[float] | None,
    cfg: BenchmarkConfig,
    caches: RuntimeCaches,
) -> tuple[list[float] | None, list[float] | None, int | None]:
    if world_coordinates is None:
        return None, None, None

    cache = caches.video_cache(video_id)
    ctx = load_frame_context(
        video_id=video_id,
        time_sec=time_sec,
        annotations_root=cfg.annotations_root,
        fps=cfg.fps_for_frame_lookup,
        intermediate_root=cfg.intermediate_root,
        cache=cache,
    )

    cam_xyz = transform_point(ctx.T_camera_world, world_coordinates)
    pixel_xy, _, valid = (
        project_fisheye624(cam_xyz, ctx.projection_params)
        if ctx.model_name == "CameraModelType.FISHEYE624"
        else (None, cam_xyz[2], False)
    )
    projected = pixel_xy if valid else None
    return projected, [float(v) for v in cam_xyz], int(ctx.frame_index)

def _find_last_placement_info_raw(
    candidate: KeyFrameCandidate,
    cfg: BenchmarkConfig,
    caches: RuntimeCaches,
) -> LastVisibleInfo | None:
    cache = caches.video_cache(candidate.video_id)
    obj = cache.assoc_objects.get(candidate.assoc_id)
    if obj is None:
        return None

    track, mode, _ = choose_track_for_time(obj.get("tracks", []), candidate.query_time_sec)
    if track is None or mode != "past":
        return None

    mask = get_mask_from_track(cache.mask_info_video, track, pick="latest")
    if mask is None:
        return None

    frame_index = mask.get("frame_number")
    placement_time_sec = (
        float(frame_index) / float(cfg.fps_for_frame_lookup)
        if frame_index is not None
        else float(track["time_segment"][1])
    )
    world_coordinates = mask.get("3d_location")
    projected_pixel, camera_coordinates, projected_frame_index = _project_world_point_at_time(
        video_id=candidate.video_id,
        time_sec=placement_time_sec,
        world_coordinates=world_coordinates,
        cfg=cfg,
        caches=caches,
    )

    return LastVisibleInfo(
        sampled_time_sec=placement_time_sec,
        projected_pixel=projected_pixel,
        camera_coordinates=camera_coordinates,
        frame_index=int(frame_index) if frame_index is not None else None,
        status="last_past_track_end",
        fixture=mask.get("fixture"),
        world_coordinates=[float(v) for v in world_coordinates] if world_coordinates is not None else None,
        reference_source="raw_assoc_info_mask_info_latest_mask_of_last_past_track",
    )

def _find_last_placement_info(
    candidate: KeyFrameCandidate,
    cfg: BenchmarkConfig,
    caches: RuntimeCaches,
) -> LastVisibleInfo | None:
    source = str(cfg.last_placement_source).strip().lower()

    if source == "raw_tracks":
        return _find_last_placement_info_raw(candidate, cfg, caches)

    if source == "merged_tracks":
        if caches.visibility_store.has_video(candidate.video_id):
            return caches.visibility_store.find_last_placement_before_from_merged(
                video_id=candidate.video_id,
                assoc_id=candidate.assoc_id,
                query_time_sec=candidate.query_time_sec,
                clip_start_time_sec=candidate.clip_start_time_sec,
            )
        return None

    raise ValueError(
        f"Unsupported last_placement_source={cfg.last_placement_source!r}. "
        "Use 'raw_tracks' or 'merged_tracks'."
    )

def _parse_visibility_tracks_json(
    raw_value: Any,
    root: Path,
    video_ids: list[str],
) -> dict[str, Path] | None:
    if raw_value is None:
        return None

    if isinstance(raw_value, str):
        if len(video_ids) != 1:
            raise ValueError(
                "visibility_tracks_json is a single path string, but multiple videos were provided. "
                "Use a mapping: visibility_tracks_json: {video_id: path}"
            )
        return {video_ids[0]: (root / raw_value).resolve()}

    if isinstance(raw_value, dict):
        output: dict[str, Path] = {}
        for video_id in video_ids:
            if video_id not in raw_value:
                raise ValueError(f"Missing visibility_tracks_json entry for video '{video_id}'.")
            output[video_id] = (root / str(raw_value[video_id])).resolve()
        return output

    raise TypeError(
        "visibility_tracks_json must be either a string path or a mapping from video_id to path."
    )



def _load_config(path: Path) -> BenchmarkConfig:
    raw = _load_yaml(path)
    root = path.parent
    inputs = raw.get("inputs", {})
    video_ids = [str(v) for v in inputs.get("videos", [])]

    return BenchmarkConfig(
        annotations_root=(root / raw["annotations_root"]).resolve(),
        video_ids=video_ids,
        sampling_fps=float(raw.get("sampling_fps", 2.0)),
        fps_for_frame_lookup=float(raw.get("fps_for_frame_lookup", 30.0)),
        out_of_sight_horizon_sec=float(raw.get("out_of_sight_horizon_sec", 2.0)),
        max_questions_per_video=int(raw.get("max_questions_per_video", 20)),
        absolute_num_choices=int(raw.get("absolute", {}).get("num_choices", 5)),
        random_seed=int(raw.get("random_seed", 42)),
        intermediate_root=str(raw.get("intermediate_root", DEFAULT_INTERMEDIATE_ROOT)),
        output_json=(root / raw.get("output_json", "staged_oos_questions.json")).resolve(),
        visibility_tracks_json_by_video=_parse_visibility_tracks_json(
            raw.get("visibility_tracks_json"),
            root=root,
            video_ids=video_ids,
        ),
        pre_context_sec=float(raw.get("pre_context_sec", 2.0)),
        raw_video_width=float(raw["raw_video_width"]) if raw.get("raw_video_width") is not None else None,
        raw_video_height=float(raw["raw_video_height"]) if raw.get("raw_video_height") is not None else None,
        last_placement_source=str(raw.get("last_placement_source", "raw_tracks")),
    )


def _normalize_projected_pixel(
    projected_pixel: list[float] | None,
    raw_video_width: float | None,
    raw_video_height: float | None,
) -> list[float] | None:
    if (
        projected_pixel is None
        or raw_video_width is None
        or raw_video_height is None
        or raw_video_width == 0
        or raw_video_height == 0
        or len(projected_pixel) < 2
    ):
        return None

    return [
        float(projected_pixel[0]) / float(raw_video_width),
        float(projected_pixel[1]) / float(raw_video_height),
    ]


def _build_common_fields(candidate: KeyFrameCandidate) -> dict[str, Any]:
    query_time_in_clip_sec = candidate.query_time_sec - candidate.clip_start_time_sec
    return {
        "inputs": {"video 1": {"id": candidate.video_id}},
        "video_id": candidate.video_id,
        "object_a_assoc_id": candidate.assoc_id,
        "object_a_name": candidate.object_name,
        "query_time_sec": candidate.query_time_sec,
        "query_time_in_clip_sec": query_time_in_clip_sec,
        "clip_start_time_sec": candidate.clip_start_time_sec,
        "clip_end_time_sec": candidate.clip_end_time_sec,
        "clip_duration_sec": candidate.clip_duration_sec,
        "horizon_sec": candidate.horizon_sec,
        "generation_info": asdict(candidate),
    }

def _finalize_choices(
    choices: list[str],
    correct_answer: str,
    rng: random.Random,
    *,
    shuffle: bool = True,
) -> tuple[list[str], int]:
    final_choices = list(choices)
    if shuffle:
        rng.shuffle(final_choices)
    correct_idx = final_choices.index(correct_answer)
    return final_choices, correct_idx


ANCHOR_STABLE_VISIBLE_STATUSES = {
    "in_view",
    "observed_visible_in_open_fixture",
}

def _pick_branch_anchor(
    candidate: KeyFrameCandidate,
    cfg: BenchmarkConfig,
    caches: RuntimeCaches,
) -> dict[str, Any] | None:
    states = caches.states_by_assoc_id(candidate.video_id, candidate.query_time_sec)
    if not states:
        return None

    ctx = load_frame_context(
        video_id=candidate.video_id,
        time_sec=candidate.query_time_sec,
        annotations_root=cfg.annotations_root,
        fps=cfg.fps_for_frame_lookup,
    )
    cx = float(ctx.image_width) / 2.0
    cy = float(ctx.image_height) / 2.0

    visible = []
    for assoc_id, s in states.items():
        if str(assoc_id) == str(candidate.assoc_id):
            continue
        if s.get("status") not in ANCHOR_STABLE_VISIBLE_STATUSES:
            continue
        if s.get("projected_pixel") is None:
            continue
        if s.get("world_coordinates") is None:
            continue
        visible.append(s)

    if not visible:
        return None

    best = min(
        visible,
        key=lambda s: (float(s["projected_pixel"][0]) - cx) ** 2
                    + (float(s["projected_pixel"][1]) - cy) ** 2,
    )

    return {
        "assoc_id": str(best["assoc_id"]),
        "name": str(best.get("name", best["assoc_id"])),
        "projected_pixel": best.get("projected_pixel"),
        "world_coordinates": best.get("world_coordinates"),
        "camera_coordinates": best.get("camera_coordinates"),
        "status": best.get("status"),
        "reference_source": "precomputed_visibility_track"
            if caches.visibility_store.has_video(candidate.video_id)
            else "live_visibility_state",
    }

def _build_step1_visibility(candidate: KeyFrameCandidate, object_state: Any, time_tok: str, rng) -> dict[str, Any]:
    is_visible = _is_visible_state(object_state)
    correct_answer = "Yes" if is_visible else "No"
    choices, correct_idx = _finalize_choices(
        choices=["Yes", "No"],
        correct_answer=correct_answer,
        rng=rng,
        shuffle=True,
    )
    return {
        "step": 1,
        "question_class": "oos_step1_visibility",
        "question": (
            f"At the current time {time_tok}, is the previously moved {candidate.object_name} visible in the current frame?"
        ),
        "choices": choices,
        "correct_idx": correct_idx,
        "answer_metadata": {
            "status": _state_attr(object_state, "status"),
            "is_visible": _state_attr(object_state, "is_visible"),
            "is_stably_visible": _state_attr(object_state, "is_stably_visible"),
            "projected_pixel": _state_attr(object_state, "projected_pixel"),
            "camera_coordinates": _state_attr(object_state, "camera_coordinates"),
            "frame_index": _state_attr(object_state, "frame_number"),
        },
    }


def _build_step2_last_visible(
    candidate: KeyFrameCandidate,
    time_tok: str,
    last_visible: LastVisibleInfo,
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    normalized_projected_pixel = _normalize_projected_pixel(
        last_visible.projected_pixel,
        cfg.raw_video_width,
        cfg.raw_video_height,
    )
    last_visible_time_token = _time_token(last_visible.sampled_time_sec, input_key="video 1")

    return {
        "step": 2,
        "question_class": "oos_step2_last_visible",
        "question": (
            #f"At the current time {time_tok}, the {candidate.object_name} that was moved earlier is not visible. "
            f"When was the previously moved {candidate.object_name} last visible, and where was it located in the image at that moment?"
        ),
        "choices": [],
        "correct_idx": None,
        "answer_metadata": {
            "sampled_last_visible_time_sec": last_visible.sampled_time_sec,
            "sampled_last_visible_time_in_clip_sec": last_visible.sampled_time_sec - candidate.clip_start_time_sec,
            "sampled_last_visible_time_token": last_visible_time_token,
            "projected_pixel": last_visible.projected_pixel,
            "normalized_projected_pixel": normalized_projected_pixel,
            "camera_coordinates": last_visible.camera_coordinates,
            "frame_index": last_visible.frame_index,
            "status": last_visible.status,
            "fixture": last_visible.fixture,
            "world_coordinates": last_visible.world_coordinates,
            "reference_source": last_visible.reference_source,
            "note": (
                "Uses the precomputed visibility track when available and otherwise falls back to "
                "live visibility computation over stable-visible states only. If the last visible "
                "state is in_motion, the trajectory is skipped."
            ),
        },
    }

def _build_step3_last_placement(
    candidate: KeyFrameCandidate,
    time_tok: str,
    last_placement: LastVisibleInfo,
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    normalized_projected_pixel = _normalize_projected_pixel(
        last_placement.projected_pixel,
        cfg.raw_video_width,
        cfg.raw_video_height,
    )
    last_placement_time_token = _time_token(last_placement.sampled_time_sec, input_key="video 1")

    return {
        "step": "3",
        "question_class": "oos_step3_last_placement",
        "question": (
            #f"At the current time {time_tok}, the {candidate.object_name} that was moved earlier is not visible. "
            f"At what time did the previously moved {candidate.object_name} stop moving? Where was it located in the image at that moment?",
        ),
        "choices": [],
        "correct_idx": None,
        "answer_metadata": {
            "last_placement_time_sec": last_placement.sampled_time_sec,
            "last_placement_time_in_clip_sec": last_placement.sampled_time_sec - candidate.clip_start_time_sec,
            "last_placement_time_token": last_placement_time_token,
            "projected_pixel": last_placement.projected_pixel,
            "normalized_projected_pixel": normalized_projected_pixel,
            "camera_coordinates": last_placement.camera_coordinates,
            "frame_index": last_placement.frame_index,
            "status": last_placement.status,
            "fixture": last_placement.fixture,
            "world_coordinates": last_placement.world_coordinates,
            "reference_source": last_placement.reference_source,
            "note": (
                "Uses exact past-track end position when last_placement_source=raw_tracks, "
                "or a sampled approximation from merged tracks when "
                "last_placement_source=merged_tracks."
            ),
        },
    }

def _is_invalid_step4_fixture_label(label: str | None) -> bool:
    if label is None:
        return True
    normalized = str(label).strip().lower()
    return normalized in {"", "mid-air"}


def _build_step4_fixture(
    candidate: KeyFrameCandidate,
    time_tok: str,
    *,
    last_visible: LastVisibleInfo,
    fixture_vocab: list[str],
    rng: random.Random,
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    if last_visible.fixture is not None:
        correct_fixture = semantic_fixture_name(last_visible.fixture)
        if not correct_fixture:
            raise ValueError(f"Could not normalize fixture {last_visible.fixture!r}")
        if _is_invalid_step4_fixture_label(correct_fixture):
            raise ValueError(
                f"Fixture label {last_visible.fixture!r} normalized to {correct_fixture!r} "
                "is not valid for step 4 question."
            )

        distractors = sorted({
            str(f)
            for f in fixture_vocab
            if str(f) and str(f) != correct_fixture and not _is_invalid_step4_fixture_label(str(f))
        })

        if len(distractors) < cfg.absolute_num_choices - 1:
            raise ValueError(
                f"Not enough semantic fixture types for step 3: need {cfg.absolute_num_choices}, "
                f"but only found {len(distractors) + 1} including the correct answer."
            )

        chosen_distractors = rng.sample(distractors, k=cfg.absolute_num_choices - 1)
        choices = [correct_fixture] + chosen_distractors
        rng.shuffle(choices)

        return {
            "step": 4,
            "question_class": "oos_step4_fixture",
            "question": (
                f"At the current time {time_tok}, based on the last known position of the {candidate.object_name} that was moved earlier,"
                f"which fixture is closest to it?"
            ),
            "choices": choices,
            "correct_idx": choices.index(correct_fixture),
            "answer_metadata": {
                "reference_time_sec": last_visible.sampled_time_sec,
                "correct_fixture": correct_fixture,
                "raw_correct_fixture": str(last_visible.fixture),
                "reference_source": last_visible.reference_source,
            },
        }

    abs_answer = determine_absolute_answer(
        video_id=candidate.video_id,
        time_sec=last_visible.sampled_time_sec,
        object_a_assoc_id=candidate.assoc_id,
        annotations_root=cfg.annotations_root,
        num_choices=cfg.absolute_num_choices,
        fixture_vocabulary=fixture_vocab,
        rng=rng,
    )
    if _is_invalid_step4_fixture_label(abs_answer.correct_fixture):
        raise ValueError(f"Invalid step 4 fixture answer: {abs_answer.correct_fixture!r}")

    return {
        "step": 4,
        "question_class": "oos_step4_fixture",
        "question": (
            f"At the current time {time_tok}, based on the last known position of the {candidate.object_name} that was moved earlier,"
            f"which fixture is closest to it?"        
            ),
        "choices": abs_answer.choices,
        "correct_idx": abs_answer.correct_idx,
        "answer_metadata": {
            "reference_time_sec": last_visible.sampled_time_sec,
            "correct_fixture": abs_answer.correct_fixture,
            "reference_source": last_visible.reference_source,
        },
    }


def classify_camera_quadrant_robust(
    camera_coordinates,
    *,
    center_margin=0.05,
    depth_margin=0.05,
    angle_margin_deg=5.0,
):
    if camera_coordinates is None or len(camera_coordinates) < 3:
        return None, None, {"reason": "no_coordinates"}

    x, _, z = [float(v) for v in camera_coordinates[:3]]

    if abs(z) < depth_margin:
        return None, None, {"reason": "near_z_boundary", "x": x, "z": z}

    if abs(x) < center_margin:
        return None, None, {"reason": "near_x_boundary", "x": x, "z": z}

    yaw_deg = math.degrees(math.atan2(x, z))
    if abs(abs(yaw_deg) - 90.0) < angle_margin_deg:
        return None, None, {"reason": "near_diagonal_boundary", "yaw_deg": yaw_deg}

    if z > 0:
        if x < 0:
            return 0, "Front-left", {"x": x, "z": z}
        return 1, "Front-right", {"x": x, "z": z}
    else:
        if x < 0:
            return 2, "Back-left", {"x": x, "z": z}
        return 3, "Back-right", {"x": x, "z": z}

def _require_query_time_target_state(
    candidate: KeyFrameCandidate,
    caches: RuntimeCaches,
) -> dict[str, Any]:
    states = caches.states_by_assoc_id(candidate.video_id, candidate.query_time_sec)
    state = states.get(candidate.assoc_id)
    if state is None:
        raise ValueError(
            f"No query-time state found for assoc_id={candidate.assoc_id} "
            f"at t={candidate.query_time_sec:.3f}"
        )

    if not isinstance(state, dict):
        # normalize ObjectState -> dict fallback
        state = {
            "assoc_id": candidate.assoc_id,
            "name": getattr(state, "name", candidate.object_name),
            "status": getattr(state, "status", None),
            "is_visible": getattr(state, "in_view", None),
            "projected_pixel": getattr(state, "projected_pixel", None),
            "camera_coordinates": getattr(state, "camera_coordinates", None),
            "world_coordinates": getattr(state, "world_coordinates", None),
            "frame_number": getattr(state, "frame_number", None),
            "fixture": getattr(state, "fixture", None),
        }

    return state

def _build_branch_object_camera_relative_position(
    candidate: KeyFrameCandidate,
    time_tok: str,
    *,
    target_state_at_query: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    correct_idx, label, debug = classify_camera_quadrant_robust(
        target_state_at_query.get("camera_coordinates"),
        center_margin=0.05,
        depth_margin=0.05,
        angle_margin_deg=5.0,
    )
    choices = list(CAMERA_QUADRANT_CHOICES)
    if label is not None and label in choices:
        choices, correct_idx = _finalize_choices(
            choices=choices,
            correct_answer=label,
            rng=rng,
            shuffle=True,
        )

    return {
        "step": "5a",
        "depends_on_steps": [1, 2, 3, 4],
        "branch_group": "post_step4",
        "question_class": "oos_branch_object_camera_relative_position",
        "question": (
            f"At the current time {time_tok}, the {candidate.object_name} that was moved earlier is not visible. "
            f"Based on its last known position, in which direction is the {candidate.object_name} from your viewpoint?"        
            ),
        "choices": choices,
        "correct_idx": correct_idx,
        "answer_metadata": {
            "reference_time_sec": candidate.query_time_sec,
            "camera_coordinates": target_state_at_query.get("camera_coordinates"),
            "world_coordinates": target_state_at_query.get("world_coordinates"),
            "status": target_state_at_query.get("status"),
            "correct_label": label,
            "debug": debug,
            "reference_source": "query_time_state_from_merged_tracks_or_live_state",
        },
        "skipped": correct_idx is None,
    }

def log_step5_failure(reason: str, context: dict[str, Any]) -> None:
    record = {"reason": reason, **context}
    print(f"[STEP5 DEBUG] {json.dumps(record, ensure_ascii=False)}")


def _build_branch_object_object_relation(
    candidate: KeyFrameCandidate,
    time_tok: str,
    *,
    anchor: dict[str, Any],
    target_state_at_query: dict[str, Any],
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    if target_state_at_query.get("world_coordinates") is None:
        raise ValueError("Query-time target state has no world coordinates")
    if anchor.get("world_coordinates") is None:
        raise ValueError("Anchor has no world coordinates")

    vector = relation_from_world_points(
        video_id=candidate.video_id,
        time_sec=candidate.query_time_sec,
        object_a_world=target_state_at_query["world_coordinates"],
        object_b_world=anchor["world_coordinates"],
        annotations_root=cfg.annotations_root,
        fps=cfg.fps_for_frame_lookup,
    )

    rel_answer = determine_relative_answer_from_vector(vector)

    return {
        "step": "5b",
        "depends_on_steps": [1, 2, 3, 4],
        "branch_group": "post_step4",
        "question_class": "oos_branch_object_object_relation",
        "question": (
            f"At the current time {time_tok}, the {candidate.object_name} that was moved earlier is not visible. "
            f"Based on the last known position of the {candidate.object_name} and the position of the marked {anchor['name']} in the current frame, "
            f"where is the {candidate.object_name} relative to {anchor['name']} from your viewpoint?"
        ),
        "choices": rel_answer.choices,
        "correct_idx": rel_answer.correct_idx,
        "acceptable_idxs": rel_answer.acceptable_idxs,
        "answer_metadata": {
            "object_x_assoc_id": candidate.assoc_id,
            "object_x_name": candidate.object_name,
            "object_x_reference_time_sec": candidate.query_time_sec,
            "object_x_status": target_state_at_query.get("status"),
            "object_x_world_coordinates": target_state_at_query.get("world_coordinates"),
            "object_x_camera_coordinates": target_state_at_query.get("camera_coordinates"),
            "object_y_assoc_id": anchor["assoc_id"],
            "object_y_name": anchor["name"],
            "object_y_reference_time_sec": candidate.query_time_sec,
            "object_y_world_coordinates": anchor["world_coordinates"],
            "object_y_projected_pixel": anchor.get("projected_pixel"),
            "reference_source": "query_time_state_from_merged_tracks_or_live_state",
        },
    }

def classify_distance_bucket(distance_m: float) -> str | None:
    if distance_m is None:
        return None
    if distance_m < 0.5:
        return "very close"
    if distance_m < 1.0:
        return "close"
    if distance_m < 2.0:
        return "medium"
    return "far"


from anchored_coords import relation_from_world_points

def _build_branch_object_object_distance(
    candidate: KeyFrameCandidate,
    time_tok: str,
    *,
    anchor: dict[str, Any],
    target_state_at_query: dict[str, Any],
    cfg: BenchmarkConfig,
    rng: random.Random,
) -> dict[str, Any]:
    if target_state_at_query.get("world_coordinates") is None:
        raise ValueError("Target state at query has no world coordinates")
    if anchor.get("world_coordinates") is None:
        raise ValueError("Anchor has no world coordinates")

    vector = relation_from_world_points(
        video_id=candidate.video_id,
        time_sec=candidate.query_time_sec,
        object_a_world=target_state_at_query["world_coordinates"],
        object_b_world=anchor["world_coordinates"],
        annotations_root=cfg.annotations_root,
        fps=cfg.fps_for_frame_lookup,
    )

    dx, dy, dz = [float(v) for v in vector]
    distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)
    correct_label = classify_distance_bucket(distance_m)
    if correct_label is None:
        raise ValueError("Could not classify object-object distance")

    choices, correct_idx = _finalize_choices(
        choices=list(DISTANCE_CHOICES),
        correct_answer=correct_label,
        rng=rng,
        shuffle=True,
    )

    return {
        "step": "5c",
        "depends_on_steps": [1, 2, 3, 4],
        "branch_group": "post_step4",
        "question_class": "oos_branch_object_object_distance",
        "question": (
            f"At the current time {time_tok}, the {candidate.object_name} that was moved earlier is not visible. "
            f"Based on the last known position of the {candidate.object_name}, and the position of the marked {anchor['name']} in the current frame, "
            f"how far is the {candidate.object_name} from the{anchor['name']}?"
        ),
        "choices": choices,
        "correct_idx": correct_idx,
        "answer_metadata": {
            "object_x_assoc_id": candidate.assoc_id,
            "object_x_name": candidate.object_name,
            "object_x_reference_time_sec": candidate.query_time_sec,
            "object_x_status_from_track": target_state_at_query.get("status"),
            "object_y_assoc_id": str(anchor["assoc_id"]),
            "object_y_name": str(anchor["name"]),
            "object_y_pixel": anchor.get("projected_pixel"),
            "object_y_status": anchor.get("status"),
            "vector_object_x_relative_to_object_y": vector,
            "distance_m": distance_m,
            "distance_bucket": correct_label,
            "reference_source": {
                "object_x": target_state_at_query.get("reference_source"),
                "object_y": anchor.get("reference_source"),
            },
        },
    }


def _finalize_trajectory(
    trajectory_id: str,
    common: dict[str, Any],
    incremental_steps: list[dict[str, Any]],
    branch_steps: list[dict[str, Any]],
    *,
    stop_reason: str,
) -> tuple[str, dict[str, Any]]:
    return trajectory_id, {
        **common,
        "question_class": "oos_staged_trajectory",
        "trajectory_id": trajectory_id,
        "num_incremental_steps": len(incremental_steps),
        "num_branch_steps": len(branch_steps),
        "terminated_at_step": incremental_steps[-1]["step"] if incremental_steps else None,
        "stop_reason": stop_reason,
        "incremental_steps": incremental_steps,
        "branch_groups": {
            "post_step3": branch_steps,
        },
    }


def _load_visibility_store(cfg: BenchmarkConfig) -> VisibilityTrackStore:
    if not cfg.visibility_tracks_json_by_video:
        return VisibilityTrackStore(None)

    raw_by_video: dict[str, dict[str, Any]] = {}
    for video_id, path in cfg.visibility_tracks_json_by_video.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Configured visibility track JSON for video '{video_id}' does not exist: {path}"
            )
        raw_by_video[video_id] = _load_json(path)

    return VisibilityTrackStore(raw_by_video)

ANCHOR_STABLE_VISIBLE_STATUSES = {
    "in_view",
    "observed_visible_in_open_fixture",
}

def _pick_step4_anchor(candidate, cfg, caches):
    states_by_assoc = caches.states_by_assoc_id(
        candidate.video_id,
        candidate.query_time_sec,
    )

    reject_counts = {
        "same_as_target": 0,
        "bad_status": 0,
        "no_projected_pixel": 0,
        "no_world_coordinates": 0,
    }

    visible = []

    for assoc_id, st in states_by_assoc.items():
        status = st.get("status") if isinstance(st, dict) else getattr(st, "status", None)
        pixel = st.get("projected_pixel") if isinstance(st, dict) else getattr(st, "projected_pixel", None)
        world = st.get("world_coordinates") if isinstance(st, dict) else getattr(st, "world_coordinates", None)

        # --- filtering ---
        if str(assoc_id) == str(candidate.assoc_id):
            reject_counts["same_as_target"] += 1
            continue

        if status not in {"in_view", "observed_visible_in_open_fixture"}:
            reject_counts["bad_status"] += 1
            continue

        if pixel is None:
            reject_counts["no_projected_pixel"] += 1
            continue

        if world is None:
            reject_counts["no_world_coordinates"] += 1
            continue

        visible.append((assoc_id, st))

    # --- compact summary log ---
    print(
        "[ANCHOR]",
        {
            "t": candidate.query_time_sec,
            "use_precomputed": caches.visibility_store.has_video(candidate.video_id),
            "n_states": len(states_by_assoc),
            "n_eligible": len(visible),
        },
    )

    # --- if no anchor, print WHY ---
    if not visible:
        print(
            "[NO_ANCHOR_REASON]",
            {
                "t": candidate.query_time_sec,
                "reject_counts": reject_counts,
            },
        )
        return None

    # --- pick most central object ---
    ctx = load_frame_context(
        video_id=candidate.video_id,
        time_sec=candidate.query_time_sec,
        annotations_root=cfg.annotations_root,
        fps=cfg.fps_for_frame_lookup,
    )

    cx = float(ctx.image_width) / 2.0
    cy = float(ctx.image_height) / 2.0

    best_assoc_id, best_state = min(
        visible,
        key=lambda item: (
            float(item[1]["projected_pixel"][0]) - cx
        ) ** 2 + (
            float(item[1]["projected_pixel"][1]) - cy
        ) ** 2,
    )

    print(
        "[ANCHOR_CHOSEN]",
        {
            "t": candidate.query_time_sec,
            "anchor_id": best_assoc_id,
            "anchor_name": best_state.get("name"),
        },
    )

    return {
        "assoc_id": best_assoc_id,
        "name": best_state.get("name"),
        "projected_pixel": best_state.get("projected_pixel"),
        "world_coordinates": best_state.get("world_coordinates"),
    }

def generate_staged_benchmark(cfg: BenchmarkConfig) -> dict[str, dict[str, Any]]:
    if not cfg.video_ids:
        raise ValueError("No input videos were provided in the config.")

    rng = random.Random(cfg.random_seed)
    horizon_token = _format_horizon_token(cfg.out_of_sight_horizon_sec)
    fixture_vocab = build_fixture_vocabulary(cfg.annotations_root, video_ids=cfg.video_ids)
    candidate_pool_per_video = max(cfg.max_questions_per_video * 5, cfg.max_questions_per_video)

    visibility_store = _load_visibility_store(cfg)
    precomputed_tracks_by_video = {
        video_id: visibility_store.get_object_tracks(video_id)
        for video_id in cfg.video_ids
        if visibility_store.has_video(video_id)
    }

    keyframes_by_video = kfg.generate_key_frames_for_videos(
        video_ids=cfg.video_ids,
        annotations_root=cfg.annotations_root,
        horizon_sec=cfg.out_of_sight_horizon_sec,
        max_questions_per_video=candidate_pool_per_video,
        sampling_fps=cfg.sampling_fps,
        fps_for_frame_lookup=cfg.fps_for_frame_lookup,
        intermediate_root=cfg.intermediate_root,
        random_seed=cfg.random_seed,
        pre_context_sec=cfg.pre_context_sec,
        precomputed_tracks_by_video=precomputed_tracks_by_video,
    )

    caches = RuntimeCaches(cfg, visibility_store)

    results: dict[str, dict[str, Any]] = {}
    running_idx = 0

    for video_id in cfg.video_ids:
        candidates = sorted(keyframes_by_video.get(video_id, []), key=lambda c: c.query_time_sec)
        emitted_for_video = 0
        target_for_video = cfg.max_questions_per_video

        for candidate in candidates:
            if emitted_for_video >= target_for_video:
                break

            try:
                states = caches.states_by_assoc_id(candidate.video_id, candidate.query_time_sec)
                object_state = states.get(candidate.assoc_id)
                if object_state is None:
                    continue

                common = _build_common_fields(candidate)
                time_tok = _time_token(candidate.query_time_sec, input_key="video 1")
                trajectory_id = f"oos_staged_{horizon_token}_{running_idx}"
                incremental_steps: list[dict[str, Any]] = []
                branch_steps: list[dict[str, Any]] = []

                step1 = _build_step1_visibility(candidate, object_state, time_tok, rng)
                incremental_steps.append(step1)

                if _is_visible_state(object_state):
                    key, payload = _finalize_trajectory(
                        trajectory_id,
                        common,
                        incremental_steps,
                        branch_steps,
                        stop_reason="object_visible_at_query_time",
                    )
                    results[key] = payload
                    running_idx += 1
                    emitted_for_video += 1
                    continue

                last_visible = _find_last_visible_info(candidate, cfg, caches)
                if last_visible is None:
                    continue

                if last_visible.status == "in_motion":
                    continue

                candidate_fixture = None
                if last_visible.fixture is not None:
                    candidate_fixture = semantic_fixture_name(last_visible.fixture)

                if candidate_fixture is not None and _is_invalid_step4_fixture_label(candidate_fixture):
                    continue

                incremental_steps.append(_build_step2_last_visible(candidate, time_tok, last_visible, cfg))
                last_placement = _find_last_placement_info(candidate, cfg, caches)
                if last_placement is None:
                    continue
                incremental_steps.append(_build_step3_last_placement(candidate, time_tok, last_placement, cfg))
                incremental_steps.append(
                    _build_step4_fixture(
                        candidate,
                        time_tok,
                        last_visible=last_visible,
                        fixture_vocab=fixture_vocab,
                        rng=rng,
                        cfg=cfg,
                    )
                )
                
                target_state_at_query = _require_query_time_target_state(candidate, caches)

                # 5a must be unambiguous, otherwise drop whole trajectory
                branch_camera = _build_branch_object_camera_relative_position(
                    candidate,
                    time_tok,
                    target_state_at_query=target_state_at_query,
                    rng=rng,
                )
                if branch_camera.get("skipped", False):
                    print(
                        "[DROP_TRAJECTORY_AMBIGUOUS_5A]",
                        {
                            "video_id": candidate.video_id,
                            "assoc_id": candidate.assoc_id,
                            "object_name": candidate.object_name,
                            "query_time_sec": candidate.query_time_sec,
                            "camera_coordinates": target_state_at_query.get("camera_coordinates"),
                            "debug": branch_camera.get("answer_metadata", {}).get("debug"),
                        },
                    )
                    continue
                branch_steps.append(branch_camera)

                # Anchor is required for 5b/5c, otherwise drop whole trajectory
                anchor = _pick_step4_anchor(candidate, cfg, caches)
                if anchor is None:
                    print(
                        "[DROP_TRAJECTORY_NO_ANCHOR]",
                        {
                            "video_id": candidate.video_id,
                            "assoc_id": candidate.assoc_id,
                            "object_name": candidate.object_name,
                            "query_time_sec": candidate.query_time_sec,
                        },
                    )
                    continue

                try:
                    step5b = _build_branch_object_object_relation(
                        candidate,
                        time_tok,
                        anchor=anchor,
                        target_state_at_query=target_state_at_query,
                        cfg=cfg,
                    )
                except Exception as e:
                    print(
                        "[DROP_TRAJECTORY_5B_FAIL]",
                        {
                            "video_id": candidate.video_id,
                            "assoc_id": candidate.assoc_id,
                            "object_name": candidate.object_name,
                            "query_time_sec": candidate.query_time_sec,
                            "anchor_assoc_id": anchor.get("assoc_id"),
                            "err": str(e),
                        },
                    )
                    continue

                try:
                    step5c = _build_branch_object_object_distance(
                        candidate,
                        time_tok,
                        anchor=anchor,
                        target_state_at_query=target_state_at_query,
                        rng=rng,
                        cfg=cfg,
                    )
                except Exception as e:
                    print(
                        "[DROP_TRAJECTORY_5C_FAIL]",
                        {
                            "video_id": candidate.video_id,
                            "assoc_id": candidate.assoc_id,
                            "object_name": candidate.object_name,
                            "query_time_sec": candidate.query_time_sec,
                            "anchor_assoc_id": anchor.get("assoc_id"),
                            "err": str(e),
                        },
                    )
                    continue

                branch_steps.append(step5b)
                branch_steps.append(step5c)

                key, payload = _finalize_trajectory(
                    trajectory_id,
                    common,
                    incremental_steps,
                    branch_steps,
                    stop_reason="completed_out_of_sight_trajectory",
                )
                results[key] = payload
                running_idx += 1
                emitted_for_video += 1

            except ValueError as e:
                print(
                    f"[SKIP] trajectory skipped for video={candidate.video_id}, "
                    f"assoc_id={candidate.assoc_id}, time={candidate.query_time_sec}: {e}"
                )
                continue

        if emitted_for_video < target_for_video:
            print(
                f"[WARN] video {video_id}: emitted {emitted_for_video}/{target_for_video} "
                f"valid trajectories. Candidate pool may be too small."
            )

    return results


def save_benchmark_json(items: dict[str, dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate staged OOS benchmark trajectories")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "staged_oos_trajectory_config.yaml",
        help="Path to the staged benchmark config YAML",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output JSON override",
    )
    parser.add_argument(
        "--visibility_tracks_json",
        type=Path,
        default=None,
        help="Optional precomputed visibility track JSON override",
    )
    parser.add_argument(
        "--pre_context_sec",
        type=float,
        default=None,
        help="Seconds before last visible time to include in clip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config.resolve())

    output_json = args.output_json.resolve() if args.output_json is not None else cfg.output_json
    visibility_tracks_json_by_video = dict(cfg.visibility_tracks_json_by_video or {})

    if args.visibility_tracks_json is not None:
        if len(cfg.video_ids) != 1:
            raise ValueError(
                "--visibility_tracks_json can only be used when exactly one input video is provided."
            )
        visibility_tracks_json_by_video[cfg.video_ids[0]] = args.visibility_tracks_json.resolve()

    pre_context_sec = args.pre_context_sec if args.pre_context_sec is not None else cfg.pre_context_sec

    cfg = BenchmarkConfig(
        **{
            **asdict(cfg),
            "output_json": output_json,
            "visibility_tracks_json_by_video": visibility_tracks_json_by_video,
            "pre_context_sec": pre_context_sec,
        }
    )

    benchmark = generate_staged_benchmark(cfg)
    save_benchmark_json(benchmark, output_json)

    print(f"Generated {len(benchmark)} staged OOS trajectories")
    if visibility_tracks_json_by_video:
        for video_id, path in visibility_tracks_json_by_video.items():
            print(f"[{video_id}] Using visibility tracks: {path}")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
