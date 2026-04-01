from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Any

from in_view_determination import (
	DEFAULT_INTERMEDIATE_ROOT,
	choose_track_for_time,
	get_mask_from_track,
	load_json,
)
from in_view_track_generator import ObjectVisibilityTrack, VisibilitySpan, generate_in_view_tracks


@dataclass(frozen=True)
class RelocationScore:
	"""Relocation ranking signal for one object in one video.

	Inputs: assoc object track metadata plus mask observations.
	Computes: fixture transition count and centroid-shift segment count.
	Outputs: score bundle used to sort candidate object A identities.
	"""
	assoc_id: str
	name: str
	fixture_transition_count: int
	centroid_shift_segment_count: int
	total_score: int


@dataclass(frozen=True)
class KeyFrameCandidate:
	"""One selected OOS key frame candidate (video, t, A)."""
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


def _collect_masks_for_track(mask_info_video: dict[str, Any], track: dict[str, Any]) -> list[dict[str, Any]]:
	"""Collect and sort valid mask entries referenced by one track."""
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
	"""Compute mean 3D centroid from track mask locations."""
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
	"""Return Euclidean distance between two 3D points."""
	return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def compute_relocation_score(
	video_id: str,
	assoc_id: str,
	annotations_root: str | Path,
	centroid_shift_threshold_m: float = 0.15,
) -> RelocationScore:
	"""Compute relocation score used for object ranking.

	Inputs: video/object id and annotation root.
	Computes:
	1) number of fixture transitions across ordered track observations,
	2) number of adjacent stable segments whose centroid shift exceeds threshold.
	Outputs: RelocationScore with summed ranking value.
	"""
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
	"""Rank objects by relocation activity (Section 8.2)."""
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

	# Highest relocation score first; deterministic tie-breakers for reproducibility.
	return sorted(scores, key=lambda s: (-s.total_score, -s.fixture_transition_count, -s.centroid_shift_segment_count, s.assoc_id))


def _times_with_visibility(track: ObjectVisibilityTrack, in_view: bool) -> list[float]:
	"""Return sampled times whose visibility matches the requested state."""
	return [t for t, v in zip(track.sampled_times_sec, track.visibility_samples) if bool(v) == in_view]


def _select_time_for_oos_span(
	track: ObjectVisibilityTrack,
	span: VisibilitySpan,
	horizon_sec: float,
) -> float | None:
	"""Select OOS query time nearest to span start plus horizon."""
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
	"""Check stricter context constraint for sustained out-of-view episodes."""
	# Enforce out-of-view around selected t for a 2*h interval when that interval exists inside the video window.
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
	"""Resolve object's fixture label at query time if stably available."""
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
	"""Prioritize one candidate per fixture before repeating locations."""
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
	"""Return True if the object was visible at least once before the query time."""
	for t, v in zip(track.sampled_times_sec, track.visibility_samples):
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
	"""Load and return sorted movement tracks for one object."""
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
	"""Return tracks fully completed before the queried time."""
	tracks = _get_object_tracks(video_id=video_id, assoc_id=assoc_id, annotations_root=annotations_root)
	return [tr for tr in tracks if float(tr["time_segment"][1]) <= query_time_sec + 1e-9]


def _sample_clip_start_from_prior_track(
	video_id: str,
	assoc_id: str,
	query_time_sec: float,
	annotations_root: str | Path,
	rng,
) -> float | None:
	"""Sample clip start from any completed prior track.

	The returned time is drawn uniformly from one of the object's prior tracks, so the
	clip can begin at any earlier movement episode rather than only after the last one.
	"""
	prior_tracks = _eligible_prior_tracks(
		video_id=video_id,
		assoc_id=assoc_id,
		query_time_sec=query_time_sec,
		annotations_root=annotations_root,
	)
	if not prior_tracks:
		return None

	chosen_track = rng.choice(prior_tracks)
	track_start_sec = float(chosen_track["time_segment"][0])
	track_end_sec = min(float(chosen_track["time_segment"][1]), query_time_sec)
	if track_end_sec < track_start_sec - 1e-9:
		return None
	if track_end_sec <= track_start_sec + 1e-9:
		return track_start_sec
	return rng.uniform(track_start_sec, track_end_sec)


def _stable_start_after_last_past_track(
	video_id: str,
	assoc_id: str,
	span_start_sec: float,
	annotations_root: str | Path,
	fps_for_frame_lookup: float = 30.0,
) -> float | None:
	"""Return earliest stable time after the last completed movement before an OOS span.

	This uses the object's last past movement track relative to span_start_sec and returns
	the later of the track end time and the latest referenced mask timestamp. When no such
	past track exists, None is returned so first-track/no-context cases can be skipped.
	"""
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


def _movement_overlaps_interval(
	video_id: str,
	assoc_id: str,
	clip_start_time_sec: float,
	clip_end_time_sec: float,
	annotations_root: str | Path,
) -> bool:
	"""Return True if any movement track overlaps the requested interval."""
	annotations_root = Path(annotations_root)
	assoc_info = load_json(annotations_root / "scene-and-object-movements" / "assoc_info.json")
	video_objects = assoc_info.get(video_id, {})
	obj = video_objects.get(assoc_id)
	if obj is None:
		return True

	for tr in obj.get("tracks", []):
		start_t, end_t = tr["time_segment"]
		if not (end_t < clip_start_time_sec or start_t > clip_end_time_sec):
			return True
	return False


def _sample_clip_window(
	span: VisibilitySpan,
	video_id: str,
	assoc_id: str,
	annotations_root: str | Path,
	rng,
	max_tries: int = 50,
) -> tuple[float, float, float, float] | None:
	"""Sample (query time, clip start, clip end, clip duration) with a prior-track start.

	Rules enforced:
	- query time and clip end must stay inside the same out-of-sight span;
	- clip start must lie inside one completed track before the query time;
	- the object must remain unseen from query time through clip end.
	"""
	if span.in_view:
		return None
	if span.end_sec < span.start_sec:
		return None

	for _ in range(max_tries):
		if span.end_sec <= span.start_sec + 1e-9:
			t_sec = span.start_sec
		else:
			t_sec = rng.uniform(span.start_sec, span.end_sec)

		clip_start_time_sec = _sample_clip_start_from_prior_track(
			video_id=video_id,
			assoc_id=assoc_id,
			query_time_sec=t_sec,
			annotations_root=annotations_root,
			rng=rng,
		)
		if clip_start_time_sec is None:
			continue

		if span.end_sec <= t_sec + 1e-9:
			clip_end_time_sec = t_sec
		else:
			clip_end_time_sec = rng.uniform(t_sec, span.end_sec)

		clip_duration_sec = clip_end_time_sec - clip_start_time_sec
		if clip_duration_sec < -1e-9:
			continue

		return t_sec, clip_start_time_sec, clip_end_time_sec, max(0.0, clip_duration_sec)

	return None


def _last_visible_time_before(track: ObjectVisibilityTrack, query_time_sec: float) -> float | None:
    last_t = None
    for t, v in zip(track.sampled_times_sec, track.visibility_samples):
        if t >= query_time_sec:
            break
        if v:
            last_t = t
    return last_t


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
) -> list[KeyFrameCandidate]:
	"""Select key frame candidates with random symmetric clip windows around t."""
	if horizon_sec <= 0:
		raise ValueError("horizon_sec must be > 0")
	if max_questions_per_video <= 0:
		return []

	tracks = generate_in_view_tracks(
		video_id=video_id,
		annotations_root=annotations_root,
		sampling_fps=sampling_fps,
		fps_for_frame_lookup=fps_for_frame_lookup,
		intermediate_root=intermediate_root,
		start_time_sec=start_time_sec,
		end_time_sec=end_time_sec,
	)
	if not tracks:
		return []

	rng = random.Random((video_id, random_seed).__repr__())

	ranked = rank_objects_by_relocation(
		video_id=video_id,
		annotations_root=annotations_root,
		centroid_shift_threshold_m=centroid_shift_threshold_m,
	)

	any_track = next(iter(tracks.values()))
	video_start_sec = min(any_track.sampled_times_sec)
	video_end_sec = max(any_track.sampled_times_sec)
	step_sec = 1.0 / sampling_fps

	selected: list[KeyFrameCandidate] = []
	for score in ranked:
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

			last_visible_time_sec = _last_visible_time_before(track, t_sec)
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

			# Make sure the answer is actually present in the clip.
			if not (clip_start_time_sec <= last_visible_time_sec < t_sec):
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
) -> dict[str, list[KeyFrameCandidate]]:
	"""Batch wrapper returning per-video key frame candidate lists."""
	out: dict[str, list[KeyFrameCandidate]] = {}
	for video_id in video_ids:
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
		)
	return out


def key_frames_to_dict(candidates: list[KeyFrameCandidate]) -> list[dict[str, Any]]:
	"""Convert key frame candidates to a JSON-serializable list."""
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
