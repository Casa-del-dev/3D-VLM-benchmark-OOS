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
from pathlib import Path
from typing import Any, Callable

import key_frame_generator as kfg
from abs_answer_determ import build_fixture_vocabulary, determine_absolute_answer
from in_view_determination import DEFAULT_INTERMEDIATE_ROOT, determine_in_view_objects, load_frame_context
from in_view_track_generator import ObjectVisibilityTrack, VisibilitySpan
from key_frame_generator import KeyFrameCandidate


CAMERA_POSE_CHANGE_CHOICES = ["No change", "Rotated left", "Rotated right", "Rotated back"]


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
    camera_turn_small_deg: float = 30.0
    camera_turn_back_deg: float = 135.0
    intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT
    output_json: Path | None = None
    visibility_tracks_json_by_video: dict[str, Path] | None = None
    pre_context_sec: float = 2.0


@dataclass(frozen=True)
class LastVisibleInfo:
    sampled_time_sec: float
    projected_pixel: list[float] | None
    frame_index: int | None
    status: str


class VisibilityTrackStore:
    def __init__(self, raw_by_video: dict[str, dict[str, Any]] | None):
        self.raw_by_video = raw_by_video or {}

    def has_video(self, video_id: str) -> bool:
        video = self.raw_by_video.get(video_id)
        return bool(video and video.get("object_tracks") is not None)

    def get_track_dict(self, video_id: str) -> dict[str, Any]:
        video = self.raw_by_video.get(video_id, {})
        return video.get("object_tracks", {})

    def get_object_tracks(self, video_id: str) -> dict[str, ObjectVisibilityTrack]:
        out: dict[str, ObjectVisibilityTrack] = {}
        for assoc_id, tr in self.get_track_dict(video_id).items():
            out[assoc_id] = ObjectVisibilityTrack(
                assoc_id=assoc_id,
                name=str(tr["name"]),
                sampled_times_sec=[float(t) for t in tr["sampled_times_sec"]],
                visibility_samples=[bool(v) for v in tr["visibility_samples"]],
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

    def get_states_by_assoc_id(self, video_id: str, time_sec: float) -> dict[str, dict[str, Any]]:
        tracks = self.get_track_dict(video_id)
        if not tracks:
            return {}

        result: dict[str, dict[str, Any]] = {}
        for assoc_id, tr in tracks.items():
            times = tr.get("sampled_times_sec", [])
            samples = tr.get("visibility_samples", [])
            idx = bisect.bisect_left(times, time_sec)
            if idx >= len(times) or abs(float(times[idx]) - float(time_sec)) > 1e-6:
                continue

            is_visible = bool(samples[idx])
            result[assoc_id] = {
                "assoc_id": assoc_id,
                "name": tr.get("name"),
                "status": "ok" if is_visible else "not_in_view_from_track",
                "in_view": is_visible,
                "projected_pixel": None,
                "frame_number": None,
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

        times = [float(t) for t in track.get("sampled_times_sec", [])]
        samples = [bool(v) for v in track.get("visibility_samples", [])]
        idx = bisect.bisect_left(times, query_time_sec) - 1

        while idx >= 0 and times[idx] >= clip_start_time_sec - 1e-9:
            if samples[idx]:
                return LastVisibleInfo(
                    sampled_time_sec=float(times[idx]),
                    projected_pixel=None,
                    frame_index=None,
                    status="ok",
                )
            idx -= 1
        return None


class RuntimeCaches:
    def __init__(self, cfg: BenchmarkConfig, visibility_store: VisibilityTrackStore | None = None):
        self.cfg = cfg
        self.visibility_store = visibility_store or VisibilityTrackStore(None)

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

        # Prefer precomputed visibility tracks when available.
        if self.visibility_store.has_video(video_id):
            states = self.visibility_store.get_states_by_assoc_id(video_id, time_sec)
            if states:
                return states

        # Fallback to live visibility computation.
        return self.live_states_by_assoc_id(video_id, time_sec)

    @lru_cache(maxsize=10000)
    def frame_context(self, video_id: str, time_sec: float) -> Any:
        time_sec = self._norm_time(time_sec)
        return load_frame_context(
            video_id=video_id,
            time_sec=time_sec,
            annotations_root=self.cfg.annotations_root,
            fps=self.cfg.fps_for_frame_lookup,
            intermediate_root=self.cfg.intermediate_root,
        )


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
        return bool(state.get("status") == "ok" and state.get("in_view"))
    return bool(state.status == "ok" and state.in_view)


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
    # Prefer precomputed visibility tracks when available to find the timestamp efficiently.
    if caches.visibility_store.has_video(candidate.video_id):
        last_visible = caches.visibility_store.find_last_visible_before(
            video_id=candidate.video_id,
            assoc_id=candidate.assoc_id,
            query_time_sec=candidate.query_time_sec,
            clip_start_time_sec=candidate.clip_start_time_sec,
        )
        if last_visible is not None:
            # Recover richer metadata with a single live lookup at the chosen timestamp.
            state = caches.live_states_by_assoc_id(
                candidate.video_id,
                round(last_visible.sampled_time_sec, 6),
            ).get(candidate.assoc_id)

            projected_pixel = _state_attr(state, "projected_pixel")
            frame_number = _state_attr(state, "frame_number")
            status = _state_attr(state, "status", last_visible.status)

            return LastVisibleInfo(
                sampled_time_sec=last_visible.sampled_time_sec,
                projected_pixel=[float(v) for v in projected_pixel] if projected_pixel is not None else None,
                frame_index=int(frame_number) if frame_number is not None else None,
                status=str(status),
            )

    # Fallback to live visibility computation if no precomputed track is available.
    step = 1.0 / cfg.sampling_fps
    t = candidate.query_time_sec - step
    while t >= candidate.clip_start_time_sec - 1e-9:
        state = caches.live_states_by_assoc_id(candidate.video_id, round(t, 6)).get(candidate.assoc_id)
        if _is_visible_state(state):
            projected_pixel = _state_attr(state, "projected_pixel")
            frame_number = _state_attr(state, "frame_number")
            return LastVisibleInfo(
                sampled_time_sec=float(round(t, 6)),
                projected_pixel=[float(v) for v in projected_pixel] if projected_pixel is not None else None,
                frame_index=int(frame_number) if frame_number is not None else None,
                status="ok",
            )
        t -= step
    return None


def _camera_forward_world(ctx: Any) -> tuple[float, float]:
    R = ctx.T_camera_world
    fx, fz = float(R[2][0]), float(R[2][2])
    norm = math.hypot(fx, fz)
    if norm < 1e-12:
        return (0.0, 1.0)
    return (fx / norm, fz / norm)


def _signed_yaw_delta_deg(forward_a: tuple[float, float], forward_b: tuple[float, float]) -> float:
    ax, az = forward_a
    bx, bz = forward_b
    dot = max(-1.0, min(1.0, ax * bx + az * bz))
    det = ax * bz - az * bx
    return math.degrees(math.atan2(det, dot))


def _classify_camera_pose_change(
    *,
    last_visible_time_sec: float,
    query_time_sec: float,
    candidate: KeyFrameCandidate,
    cfg: BenchmarkConfig,
    caches: RuntimeCaches,
) -> tuple[int, float]:
    ctx_last = caches.frame_context(candidate.video_id, last_visible_time_sec)
    ctx_query = caches.frame_context(candidate.video_id, query_time_sec)

    yaw_delta = _signed_yaw_delta_deg(_camera_forward_world(ctx_last), _camera_forward_world(ctx_query))
    abs_delta = abs(yaw_delta)
    if abs_delta <= cfg.camera_turn_small_deg:
        return 0, yaw_delta
    if abs_delta >= cfg.camera_turn_back_deg:
        return 3, yaw_delta
    return (1, yaw_delta) if yaw_delta > 0.0 else (2, yaw_delta)


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
        camera_turn_small_deg=float(raw.get("camera_pose_change", {}).get("no_change_threshold_deg", 30.0)),
        camera_turn_back_deg=float(raw.get("camera_pose_change", {}).get("rotated_back_threshold_deg", 135.0)),
        intermediate_root=str(raw.get("intermediate_root", DEFAULT_INTERMEDIATE_ROOT)),
        output_json=(root / raw.get("output_json", "staged_oos_questions.json")).resolve(),
        visibility_tracks_json_by_video=_parse_visibility_tracks_json(
            raw.get("visibility_tracks_json"),
            root=root,
            video_ids=video_ids,
        ),
        pre_context_sec=float(raw.get("pre_context_sec", 2.0)),
    )


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


def _build_step1_visibility(candidate: KeyFrameCandidate, object_state: Any, time_tok: str) -> dict[str, Any]:
    is_visible = _is_visible_state(object_state)
    return {
        "step": 1,
        "question_class": "oos_step1_visibility",
        "question": (
            f"At the current time {time_tok}, is the target {candidate.object_name} "
            "(the same object instance queried here) visible in the current frame?"
        ),
        "choices": ["Yes", "No"],
        "correct_idx": 0 if is_visible else 1,
        "answer_metadata": {
            "status": _state_attr(object_state, "status"),
            "in_view": _state_attr(object_state, "in_view"),
            "projected_pixel": _state_attr(object_state, "projected_pixel"),
        },
    }


def _build_step2_last_visible(candidate: KeyFrameCandidate, time_tok: str, last_visible: LastVisibleInfo) -> dict[str, Any]:
    return {
        "step": 2,
        "question_class": "oos_step2_last_visible",
        "question": (
            f"At the current time {time_tok}, the target {candidate.object_name} is not visible. "
            "When was it last visible, and where was it located in the image?"
        ),
        "choices": [],
        "correct_idx": None,
        "answer_metadata": {
            "sampled_last_visible_time_sec": last_visible.sampled_time_sec,
            "sampled_last_visible_time_in_clip_sec": last_visible.sampled_time_sec - candidate.clip_start_time_sec,
            "projected_pixel": last_visible.projected_pixel,
            "frame_index": last_visible.frame_index,
            "note": "This answer uses the precomputed visibility track when available, otherwise falls back to live visibility computation.",
        },
    }


def _build_step3_fixture(
    candidate: KeyFrameCandidate,
    *,
    reference_time_sec: float,
    fixture_vocab: list[str],
    rng: random.Random,
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    abs_answer = determine_absolute_answer(
        video_id=candidate.video_id,
        time_sec=reference_time_sec,
        object_a_assoc_id=candidate.assoc_id,
        annotations_root=cfg.annotations_root,
        num_choices=cfg.absolute_num_choices,
        fixture_vocabulary=fixture_vocab,
        rng=rng,
    )
    return {
        "step": 3,
        "question_class": "oos_step3_fixture",
        "question": (
            f"Based on the last visible position of the target {candidate.object_name}, "
            "which nearby fixture or landmark is closest to it?"
        ),
        "choices": abs_answer.choices,
        "correct_idx": abs_answer.correct_idx,
        "answer_metadata": {
            "reference_time_sec": reference_time_sec,
            "correct_fixture": abs_answer.correct_fixture,
        },
    }


def _build_step4_camera_pose_change(
    candidate: KeyFrameCandidate,
    time_tok: str,
    last_visible: LastVisibleInfo,
    cfg: BenchmarkConfig,
    caches: RuntimeCaches,
) -> dict[str, Any]:
    pose_idx, yaw_delta_deg = _classify_camera_pose_change(
        last_visible_time_sec=last_visible.sampled_time_sec,
        query_time_sec=candidate.query_time_sec,
        candidate=candidate,
        cfg=cfg,
        caches=caches,
    )
    return {
        "step": 4,
        "question_class": "oos_step4_camera_pose_change",
        "question": (
            f"At the current time {time_tok}, compared with when the target {candidate.object_name} was last visible, "
            "what is the net change in the camera's viewing direction?"
        ),
        "choices": CAMERA_POSE_CHANGE_CHOICES,
        "correct_idx": pose_idx,
        "answer_metadata": {
            "yaw_delta_deg": yaw_delta_deg,
            "no_change_threshold_deg": cfg.camera_turn_small_deg,
            "rotated_back_threshold_deg": cfg.camera_turn_back_deg,
            "reference_time_sec": last_visible.sampled_time_sec,
        },
    }


def _finalize_trajectory(trajectory_id: str, common: dict[str, Any], steps: list[dict[str, Any]], *, stop_reason: str) -> tuple[str, dict[str, Any]]:
    return trajectory_id, {
        **common,
        "question_class": "oos_staged_trajectory",
        "trajectory_id": trajectory_id,
        "num_steps": len(steps),
        "terminated_at_step": steps[-1]["step"] if steps else None,
        "stop_reason": stop_reason,
        "steps": steps,
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


def _install_precomputed_track_loader(
    store: VisibilityTrackStore,
) -> Callable[..., dict[str, ObjectVisibilityTrack]] | None:
    if not store.raw_by_video:
        return None

    original = kfg.generate_in_view_tracks

    def _patched_generate_in_view_tracks(
        video_id: str,
        annotations_root: str | Path,
        sampling_fps: float = 2.0,
        fps_for_frame_lookup: float = 30.0,
        intermediate_root: str = DEFAULT_INTERMEDIATE_ROOT,
        start_time_sec: float | None = None,
        end_time_sec: float | None = None,
    ) -> dict[str, ObjectVisibilityTrack]:
        if store.has_video(video_id):
            return store.get_object_tracks(video_id)
        return original(
            video_id=video_id,
            annotations_root=annotations_root,
            sampling_fps=sampling_fps,
            fps_for_frame_lookup=fps_for_frame_lookup,
            intermediate_root=intermediate_root,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
        )

    kfg.generate_in_view_tracks = _patched_generate_in_view_tracks
    return original


def generate_staged_benchmark(cfg: BenchmarkConfig) -> dict[str, dict[str, Any]]:
    if not cfg.video_ids:
        raise ValueError("No input videos were provided in the config.")

    rng = random.Random(cfg.random_seed)
    horizon_token = _format_horizon_token(cfg.out_of_sight_horizon_sec)
    fixture_vocab = build_fixture_vocabulary(cfg.annotations_root, video_ids=cfg.video_ids)

    visibility_store = _load_visibility_store(cfg)
    original_track_loader = _install_precomputed_track_loader(visibility_store)
    try:
        keyframes_by_video = kfg.generate_key_frames_for_videos(
            video_ids=cfg.video_ids,
            annotations_root=cfg.annotations_root,
            horizon_sec=cfg.out_of_sight_horizon_sec,
            max_questions_per_video=cfg.max_questions_per_video,
            sampling_fps=cfg.sampling_fps,
            fps_for_frame_lookup=cfg.fps_for_frame_lookup,
            intermediate_root=cfg.intermediate_root,
            random_seed=cfg.random_seed,
            pre_context_sec=cfg.pre_context_sec,
        )
    finally:
        if original_track_loader is not None:
            kfg.generate_in_view_tracks = original_track_loader

    caches = RuntimeCaches(cfg, visibility_store)

    results: dict[str, dict[str, Any]] = {}
    running_idx = 0

    for video_id in cfg.video_ids:
        candidates = sorted(keyframes_by_video.get(video_id, []), key=lambda c: c.query_time_sec)
        for candidate in candidates:
            states = caches.states_by_assoc_id(candidate.video_id, candidate.query_time_sec)
            object_state = states.get(candidate.assoc_id)
            if object_state is None:
                continue

            common = _build_common_fields(candidate)
            time_tok = _time_token(common["query_time_in_clip_sec"], input_key="video 1")
            trajectory_id = f"oos_staged_{horizon_token}_{running_idx}"
            steps: list[dict[str, Any]] = []

            step1 = _build_step1_visibility(candidate, object_state, time_tok)
            steps.append(step1)

            if step1["correct_idx"] == 0:
                key, payload = _finalize_trajectory(
                    trajectory_id,
                    common,
                    steps,
                    stop_reason="object_visible_at_query_time",
                )
                results[key] = payload
                running_idx += 1
                continue

            last_visible = _find_last_visible_info(candidate, cfg, caches)
            if last_visible is None:
                key, payload = _finalize_trajectory(
                    trajectory_id,
                    common,
                    steps,
                    stop_reason="object_not_visible_but_no_last_visible_reference_found",
                )
                results[key] = payload
                running_idx += 1
                continue

            steps.append(_build_step2_last_visible(candidate, time_tok, last_visible))

            try:
                steps.append(
                    _build_step3_fixture(
                        candidate,
                        reference_time_sec=last_visible.sampled_time_sec,
                        fixture_vocab=fixture_vocab,
                        rng=rng,
                        cfg=cfg,
                    )
                )
            except Exception as exc:
                steps.append(
                    {
                        "step": 3,
                        "question_class": "oos_step3_fixture",
                        "question": (
                            f"Based on the last visible position of the target {candidate.object_name}, "
                            "which nearby fixture or landmark is closest to it?"
                        ),
                        "choices": [],
                        "correct_idx": None,
                        "answer_metadata": {
                            "reference_time_sec": last_visible.sampled_time_sec,
                            "error": repr(exc),
                        },
                        "skipped": True,
                    }
                )

            try:
                steps.append(_build_step4_camera_pose_change(candidate, time_tok, last_visible, cfg, caches))
            except Exception as exc:
                steps.append(
                    {
                        "step": 4,
                        "question_class": "oos_step4_camera_pose_change",
                        "question": (
                            f"At the current time {time_tok}, compared with when the target {candidate.object_name} was last visible, "
                            "what is the net change in the camera's viewing direction?"
                        ),
                        "choices": CAMERA_POSE_CHANGE_CHOICES,
                        "correct_idx": None,
                        "answer_metadata": {
                            "reference_time_sec": last_visible.sampled_time_sec,
                            "error": repr(exc),
                        },
                        "skipped": True,
                    }
                )

            key, payload = _finalize_trajectory(
                trajectory_id,
                common,
                steps,
                stop_reason="completed_out_of_sight_trajectory",
            )
            results[key] = payload
            running_idx += 1

    return results


def save_benchmark_json(items: dict[str, dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate staged OOS benchmark trajectories (optimized)")
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