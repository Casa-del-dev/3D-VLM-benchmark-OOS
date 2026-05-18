"""Per-status audit of the visibility track using OWLv2.

Reads ``visibility_track.jsonl`` for one or more videos and, for each
pipeline status, samples ``--n-intervals`` intervals (pooled across all
selected videos). Inside each sampled interval it picks ``--n-samples``
random frames and runs OWLv2 with the object's name as the prompt.

Identity-aware detection
------------------------
A frame counts as ``detected`` iff at least one OWLv2 box contains the
tracked object's per-frame 3D-to-2D projection (recomputed at every
sampled frame via ``determine_in_view_objects``). When several boxes
contain the projection, the one whose center is closest to it wins;
this lets us pick the right instance in frames with multiple same-class
objects. Frames with no in-frame projection are reported as
``n_frames_skipped`` rather than scored.

Object pruning
--------------
Candidate objects are pruned before sampling:

1. **Duplicate-name filter** (default on; disable with
   ``--keep-duplicate-names``). Drop any object whose normalized name
   is shared by another assoc_id in the same video -- the OWLv2 prompt
   would address all of them ambiguously.
2. **Vocabulary filter** (``--vocabulary-filter``, default ``lvis``):
   ``none`` keeps everything; ``structural`` rejects names with > 2
   tokens, prepositions/ordinals (``of``, ``with``, ``second``, ...),
   or digits (``cocoa can1``); ``lvis`` adds an LVIS-1203 membership
   check on the full name or its head noun (OWLv2's reference
   benchmark vocabulary).

Per-status policy
-----------------
The audit covers seven statuses -- every status the pipeline emits
except ``out_of_view`` and ``unobservable_no_data``. ``in_motion`` is
sampled so its pool size is reported, but every frame ends up
``skipped`` because the pipeline returns no projection during
manipulation. ``out_of_view`` is excluded entirely -- the tracked
object projects outside the camera frame, so any same-class box must
be a different instance.

The script reads ``<output_root>/<video_id>/visibility_track.jsonl``,
the annotations/intermediate data, and raw video frames. It writes one
aggregated JSON report to ``--output`` (default
``<output_root>/owlv2_audit.json``).

OWLv2 (``google/owlv2-large-patch14-ensemble``) is architecturally
distinct from the Grounding DINO and Detic backends used inside the
pipeline, so disagreement is a cleaner signal than re-running the same
family. Disagreement is *not* ground truth: treat the report as
triage, not an oracle.

Visual inspection
-----------------
Pass ``--inspect-status <status>`` to save annotated JPEGs of audit
decisions for one status. Up to ``--inspect-n`` ``detected`` frames go
to ``positive/`` and up to ``--inspect-n`` ``not_detected`` frames go
to ``negative/``. Each JPEG overlays every OWLv2 box (cyan), the
matched box (green) when present, and the projected pixel (orange).

Example::

    # Audit one video, 50 intervals per status, 5 frames per interval
    python -m scripts.visibility_track.owlv2_audit --video P01-20240203-184045 --n-intervals 100 --n-samples 8

    # Pool across every video of P01 and P02
    python -m scripts.visibility_track.owlv2_audit --participant P01 --participant P02 --output /tmp/audit_P01_P02.json

    # Save 20 positive + 20 negative in_view frames for inspection
    python -m scripts.visibility_track.owlv2_audit --participant P01 --inspect-status in_view --inspect-n 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.visibility_track.common import PipelineConfig, load_config, read_jsonl  # noqa: E402
    from scripts.visibility_track.detection_refinement import VideoFrameReader  # noqa: E402
    from scripts.visibility_track.in_view_track.in_view_determination import (  # noqa: E402
        VideoCache,
        determine_in_view_objects,
    )
    from scripts.visibility_track.lvis_vocabulary import LVIS_HEAD_NOUNS, LVIS_TERMS  # noqa: E402
else:
    from .common import PipelineConfig, load_config, read_jsonl
    from .detection_refinement import VideoFrameReader
    from .in_view_track.in_view_determination import VideoCache, determine_in_view_objects
    from .lvis_vocabulary import LVIS_HEAD_NOUNS, LVIS_TERMS


# --- Constants -------------------------------------------------------------

DEFAULT_CONFIG = Path(__file__).resolve().parent / "visibility_track_config.yaml"

# Statuses the audit samples. ``out_of_view`` and ``unobservable_no_data``
# are omitted (no usable anchor / no signal). ``in_motion`` is included so
# its pool size is reported, but every frame ends up ``skipped`` because
# the pipeline returns no projection during manipulation.
AUDITED_STATUSES: tuple[str, ...] = (
    "in_motion",
    "occluded_inside_closed_fixture",
    "observed_visible_in_open_fixture",
    "observed_not_visible_in_open_fixture",
    "assumed_not_visible_in_open_fixture",
    "geometrically_occluded",
    "in_view",
)

# Pipeline's binary visibility verdict per status. None = ambiguous (no
# claim is made, so we don't compute a disagreement rate). ``in_motion``
# is ambiguous because visibility computation is suspended during
# manipulation.
PIPELINE_SAYS_VISIBLE: dict[str, bool | None] = {
    "in_motion": None,
    "occluded_inside_closed_fixture": False,
    "observed_visible_in_open_fixture": True,
    "observed_not_visible_in_open_fixture": False,
    "assumed_not_visible_in_open_fixture": False,
    "geometrically_occluded": False,
    "in_view": True,
}

# Pixel tolerance around the projected pixel when checking whether an
# OWLv2 box matches the tracked instance. Conservative enough to absorb
# 3D-to-2D projection error without bridging to an adjacent same-class
# instance. Override with ``--projection-margin-px``.
POINT_IN_BOX_MARGIN_PX = 20.0


# --- Name filtering --------------------------------------------------------
#
# OWLv2's reference benchmark is LVIS, so we keep only names whose tokens
# map onto an LVIS concept. The structural test additionally catches names
# OWLv2 cannot resolve regardless of class membership: compound descriptors
# with prepositions ("box of hand blender"), ordinals ("second tube of
# biscuits"), and digit-suffix instance markers ("cocoa can1").

NAME_REJECT_TOKENS: frozenset[str] = frozenset({
    "of", "with", "for", "in",
    "first", "second", "third", "fourth", "fifth",
    "another", "other",
})

VOCABULARY_FILTER_MODES: tuple[str, ...] = ("none", "structural", "lvis")


def _normalize_name_tokens(name: str) -> list[str]:
    return name.lower().replace("_", " ").split()


def _passes_structural(name: str) -> bool:
    """At most two tokens, no reject tokens, no digits."""
    tokens = _normalize_name_tokens(name)
    if not tokens or len(tokens) > 2:
        return False
    if any(tok in NAME_REJECT_TOKENS for tok in tokens):
        return False
    if any(ch.isdigit() for tok in tokens for ch in tok):
        return False
    return True


def _passes_lvis(name: str) -> bool:
    """Full name (joined with underscores) or its head noun is in LVIS."""
    tokens = _normalize_name_tokens(name)
    if not tokens:
        return False
    if "_".join(tokens) in LVIS_TERMS:
        return True
    head = tokens[-1]
    return head in LVIS_TERMS or head in LVIS_HEAD_NOUNS


# --- Dataclasses -----------------------------------------------------------

@dataclass(frozen=True)
class IntervalSpec:
    video_id: str
    assoc_id: str
    object_name: str
    status: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class FrameSample:
    video_id: str
    assoc_id: str
    object_name: str
    status: str
    interval_key: tuple[str, str, float, float]
    time_sec: float


@dataclass(frozen=True)
class FrameResult:
    sample: FrameSample
    verdict: str  # "detected" | "not_detected" | "skipped"
    n_boxes_returned: int
    n_boxes_matched: int
    best_match_score: float | None


@dataclass(frozen=True)
class InspectCandidate:
    """A frame eligible for the visual-inspection dump."""
    video_id: str
    assoc_id: str
    object_name: str
    status: str
    verdict: str
    time_sec: float
    projected_pixel: tuple[float, float]
    boxes_with_scores: list[tuple[list[float], float]]
    matched_box: list[float] | None


@dataclass(frozen=True)
class FilterStats:
    n_objects_total: int
    n_objects_kept: int
    n_dropped_duplicate_name: int
    n_dropped_structural: int
    n_dropped_lvis: int
    rejected_names: dict[str, str]  # name -> reason

    def to_dict(self) -> dict:
        return {
            "n_objects_total": self.n_objects_total,
            "n_objects_kept": self.n_objects_kept,
            "n_dropped_duplicate_name": self.n_dropped_duplicate_name,
            "n_dropped_structural": self.n_dropped_structural,
            "n_dropped_lvis_vocabulary": self.n_dropped_lvis,
            "rejected_names_sample": sorted(
                ({"name": n, "reason": r} for n, r in self.rejected_names.items()),
                key=lambda r: (r["reason"], r["name"]),
            ),
        }


# --- Pool building / sampling ---------------------------------------------

def _duplicate_name_assoc_ids(
    rows_by_video: dict[str, list[dict]],
) -> set[tuple[str, str]]:
    """Return (video_id, assoc_id) pairs whose normalized name is shared
    by at least one other assoc_id in the same video."""
    by_name: dict[tuple[str, str], set[str]] = defaultdict(set)
    for video_id, rows in rows_by_video.items():
        for row in rows:
            norm = " ".join(_normalize_name_tokens(row.get("object_name") or ""))
            if norm:
                by_name[(video_id, norm)].add(row["assoc_id"])
    return {
        (video_id, aid)
        for (video_id, _), aids in by_name.items()
        if len(aids) > 1
        for aid in aids
    }


def _object_decision(name: str, is_duplicate: bool, vocabulary_filter: str) -> str:
    """Return 'kept' or a rejection reason for a candidate object name."""
    if is_duplicate:
        return "duplicate_name"
    if vocabulary_filter == "none":
        return "kept"
    if not _passes_structural(name):
        return "structural"
    if vocabulary_filter == "lvis" and not _passes_lvis(name):
        return "lvis_vocabulary"
    return "kept"


def _summarize_filter(decisions: dict[tuple[str, str], tuple[str, str]]) -> FilterStats:
    counts = {"kept": 0, "duplicate_name": 0, "structural": 0, "lvis_vocabulary": 0}
    rejected_names: dict[str, str] = {}
    for name, reason in decisions.values():
        counts[reason] += 1
        if reason != "kept":
            rejected_names.setdefault(name, reason)
    return FilterStats(
        n_objects_total=len(decisions),
        n_objects_kept=counts["kept"],
        n_dropped_duplicate_name=counts["duplicate_name"],
        n_dropped_structural=counts["structural"],
        n_dropped_lvis=counts["lvis_vocabulary"],
        rejected_names=rejected_names,
    )


def _build_interval_pool(
    cfg: PipelineConfig,
    video_ids: list[str],
    *,
    vocabulary_filter: str = "lvis",
    drop_duplicate_names: bool = True,
) -> tuple[dict[str, list[IntervalSpec]], list[str], FilterStats]:
    """Walk visibility_track.jsonl for every selected video and group
    rows by status into a flat per-status pool of intervals. Filtering
    is done at the (video, assoc_id) level so every interval for a
    rejected object is excluded together."""
    rows_by_video: dict[str, list[dict]] = {}
    missing: list[str] = []
    for video_id in video_ids:
        vt_path = cfg.video_output_dir(video_id) / "visibility_track.jsonl"
        if vt_path.exists():
            rows_by_video[video_id] = read_jsonl(vt_path)
        else:
            missing.append(video_id)

    duplicates = _duplicate_name_assoc_ids(rows_by_video) if drop_duplicate_names else set()

    # Decide once per (video_id, assoc_id) so multi-interval objects stay
    # consistent and don't inflate the rejection counts.
    decisions: dict[tuple[str, str], tuple[str, str]] = {}  # key -> (name, reason)
    for video_id, rows in rows_by_video.items():
        for row in rows:
            key = (video_id, row["assoc_id"])
            if key in decisions:
                continue
            name = row.get("object_name") or ""
            reason = _object_decision(name, key in duplicates, vocabulary_filter)
            decisions[key] = (name, reason)

    pool: dict[str, list[IntervalSpec]] = defaultdict(list)
    for video_id, rows in rows_by_video.items():
        for row in rows:
            status = row.get("status")
            if status not in PIPELINE_SAYS_VISIBLE:
                continue
            if decisions[(video_id, row["assoc_id"])][1] != "kept":
                continue
            pool[status].append(IntervalSpec(
                video_id=video_id,
                assoc_id=row["assoc_id"],
                object_name=row.get("object_name") or "",
                status=status,
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
            ))

    return pool, missing, _summarize_filter(decisions)


def _sample_intervals(
    pool: dict[str, list[IntervalSpec]],
    n_intervals: int,
    rng: random.Random,
) -> dict[str, list[IntervalSpec]]:
    return {
        status: (list(intervals) if len(intervals) <= n_intervals
                 else rng.sample(intervals, n_intervals))
        for status, intervals in pool.items()
    }


def _sample_frame_times(
    interval: IntervalSpec,
    n_samples: int,
    video_fps: float,
    rng: random.Random,
) -> list[float]:
    """Pick up to *n_samples* distinct frame timestamps in
    [start_sec, end_sec]. Quantized to frame indices so two samples
    never decode the same frame; if the interval is shorter than
    *n_samples* frames we return all available frames."""
    if interval.end_sec < interval.start_sec or video_fps <= 0:
        return []
    first_idx = int(round(interval.start_sec * video_fps))
    last_idx = int(round(interval.end_sec * video_fps))
    available = last_idx - first_idx + 1
    if available <= 0:
        return []
    n = min(n_samples, available)
    indices = rng.sample(range(first_idx, last_idx + 1), n)
    return sorted(round(idx / video_fps, 6) for idx in indices)


def _build_frame_samples(
    sampled_intervals: dict[str, list[IntervalSpec]],
    n_samples: int,
    video_fps: float,
    rng: random.Random,
) -> list[FrameSample]:
    out: list[FrameSample] = []
    for intervals in sampled_intervals.values():
        for iv in intervals:
            for t in _sample_frame_times(iv, n_samples, video_fps, rng):
                out.append(FrameSample(
                    video_id=iv.video_id,
                    assoc_id=iv.assoc_id,
                    object_name=iv.object_name,
                    status=iv.status,
                    interval_key=(iv.video_id, iv.assoc_id, iv.start_sec, iv.end_sec),
                    time_sec=t,
                ))
    return out


# --- OWLv2 detector --------------------------------------------------------

class _OWLv2:
    """Thin wrapper around google/owlv2 zero-shot open-vocabulary detection."""

    def __init__(self, model_id: str, device: str = "mps", score_threshold: float = 0.10) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.model_id = model_id
        self.device = device
        self.score_threshold = float(score_threshold)
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()

    def detect(self, image_bgr, prompt: str) -> list[tuple[list[float], float]]:
        """Return list of (xyxy_box, score) for *prompt* in *image_bgr*."""
        import numpy as np
        import torch
        from PIL import Image

        rgb = image_bgr[:, :, ::-1]
        pil = Image.fromarray(np.ascontiguousarray(rgb))
        h, w = image_bgr.shape[:2]

        inputs = self.processor(text=[[prompt]], images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
            text_labels=[[prompt]],
        )[0]

        boxes = results["boxes"].detach().cpu().numpy().tolist()
        scores = results["scores"].detach().cpu().numpy().tolist()
        return [(list(map(float, b)), float(s)) for b, s in zip(boxes, scores)]


# --- Spatial-match classifier ---------------------------------------------

def _best_match_box(
    boxes_with_scores: list[tuple[list[float], float]],
    projected_pixel: tuple[float, float],
    point_margin_px: float,
) -> tuple[list[float], float, int] | None:
    """Among boxes whose footprint (expanded by ``point_margin_px``)
    contains *projected_pixel*, return ``(box, score, n_containing)``
    for the one whose center is closest to it. ``None`` if no box
    qualifies."""
    u, v = projected_pixel
    candidates: list[tuple[float, list[float], float]] = []
    for box, score in boxes_with_scores:
        x1, y1, x2, y2 = box
        if (x1 - point_margin_px) <= u <= (x2 + point_margin_px) and \
           (y1 - point_margin_px) <= v <= (y2 + point_margin_px):
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            d_sq = (u - cx) ** 2 + (v - cy) ** 2
            candidates.append((d_sq, box, score))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    _, best_box, best_score = candidates[0]
    return best_box, best_score, len(candidates)


def _classify_frame(
    boxes_with_scores: list[tuple[list[float], float]],
    projected_pixel: tuple[float, float] | None,
    point_margin_px: float = POINT_IN_BOX_MARGIN_PX,
) -> tuple[str, int, float | None]:
    """Return ``(verdict, n_matched, best_match_score)``, where verdict
    is ``'detected' | 'not_detected' | 'skipped'``. See module docstring
    for the matching rule."""
    if projected_pixel is None:
        return "skipped", 0, None
    if not boxes_with_scores:
        return "not_detected", 0, None
    match = _best_match_box(boxes_with_scores, projected_pixel, point_margin_px)
    if match is None:
        return "not_detected", 0, None
    _, best_score, n_containing = match
    return "detected", n_containing, best_score


# --- Audit runner ----------------------------------------------------------

def _resolve_anchor(
    cfg: PipelineConfig,
    video_id: str,
    time_sec: float,
    assoc_id: str,
    cache: VideoCache,
) -> tuple[float, float] | None:
    """Return the projected pixel for *assoc_id* at *time_sec*, or
    ``None`` if no in-frame projection is available."""
    states = determine_in_view_objects(
        video_id=video_id,
        time_sec=time_sec,
        annotations_root=cfg.annotations_root,
        fps=cfg.video_fps,
        intermediate_root=str(cfg.intermediate_data_root),
        cache=cache,
    )
    state = next((s for s in states if s.assoc_id == assoc_id), None)
    if state is None or state.status != "ok" or state.projected_pixel is None:
        return None
    return tuple(state.projected_pixel)


def _skipped_result(sample: FrameSample) -> FrameResult:
    return FrameResult(
        sample=sample, verdict="skipped",
        n_boxes_returned=0, n_boxes_matched=0, best_match_score=None,
    )


def _run_video_samples(
    cfg: PipelineConfig,
    video_id: str,
    samples: list[FrameSample],
    detector: _OWLv2,
    point_margin_px: float,
    inspect_status: str | None = None,
    inspect_candidates: list[InspectCandidate] | None = None,
) -> list[FrameResult]:
    results: list[FrameResult] = []
    cache = VideoCache.build(video_id, cfg.annotations_root, str(cfg.intermediate_data_root))

    with VideoFrameReader(cfg.video_file(video_id), cfg.video_fps) as frame_reader:
        for sample in tqdm(samples, desc=video_id, unit="frame", leave=False):
            projected = _resolve_anchor(cfg, video_id, sample.time_sec, sample.assoc_id, cache)
            if projected is None:
                results.append(_skipped_result(sample))
                continue

            frame = frame_reader.read_at(sample.time_sec)
            if frame is None:
                results.append(_skipped_result(sample))
                continue

            boxes_with_scores = detector.detect(frame, sample.object_name)
            verdict, n_matched, best_score = _classify_frame(
                boxes_with_scores, projected, point_margin_px=point_margin_px,
            )
            results.append(FrameResult(
                sample=sample,
                verdict=verdict,
                n_boxes_returned=len(boxes_with_scores),
                n_boxes_matched=n_matched,
                best_match_score=best_score,
            ))

            if (
                inspect_candidates is not None
                and sample.status == inspect_status
                and verdict in ("detected", "not_detected")
            ):
                match = _best_match_box(boxes_with_scores, projected, point_margin_px)
                inspect_candidates.append(InspectCandidate(
                    video_id=video_id,
                    assoc_id=sample.assoc_id,
                    object_name=sample.object_name,
                    status=sample.status,
                    verdict=verdict,
                    time_sec=sample.time_sec,
                    projected_pixel=projected,
                    boxes_with_scores=[(list(b), float(s)) for b, s in boxes_with_scores],
                    matched_box=list(match[0]) if match is not None else None,
                ))
    return results


# --- Inspection-image dump -------------------------------------------------

# Colors are BGR.
_INSPECT_PROMPT_COLOR = (60, 220, 255)       # amber
_INSPECT_PROJECTION_COLOR = (60, 180, 255)   # orange
_INSPECT_OTHER_BOX_COLOR = (220, 220, 80)    # cyan
_INSPECT_POSITIVE_COLOR = (80, 230, 80)      # green
_INSPECT_NEGATIVE_COLOR = (80, 80, 240)      # red


def _safe_filename_part(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_") or "unnamed"


def _draw_text_with_outline(
    img,
    text: str,
    org: tuple[int, int],
    *,
    color: tuple[int, int, int],
    scale: float = 0.7,
    thickness: int = 2,
) -> None:
    """Draw text with a dark outline so it's readable on any background."""
    import cv2  # type: ignore[import-not-found]
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _draw_inspect_overlay(frame, c: InspectCandidate):
    """Render the inspection visualization for a single frame."""
    import cv2  # type: ignore[import-not-found]
    out = frame.copy()
    w = out.shape[1]

    # Every OWLv2 box (thin cyan) with its score.
    for box, score in c.boxes_with_scores:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), _INSPECT_OTHER_BOX_COLOR, 1)
        _draw_text_with_outline(
            out, f"{score:.2f}", (x1 + 2, max(14, y1 - 4)),
            color=_INSPECT_OTHER_BOX_COLOR, scale=0.45, thickness=1,
        )

    # Matched box (thick green) -- only present for 'detected' frames.
    if c.matched_box is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in c.matched_box)
        cv2.rectangle(out, (x1, y1), (x2, y2), _INSPECT_POSITIVE_COLOR, 3)

    # Projected pixel dot + label pinned to it.
    u, v = c.projected_pixel
    ui, vi = int(round(u)), int(round(v))
    cv2.circle(out, (ui, vi), 10, (0, 0, 0), 2)
    cv2.circle(out, (ui, vi), 8, _INSPECT_PROJECTION_COLOR, -1)
    _draw_text_with_outline(
        out, f"looking for: {c.object_name!r}",
        (min(ui + 14, w - 10), max(vi - 14, 18)),
        color=_INSPECT_PROMPT_COLOR, scale=0.6, thickness=2,
    )

    # Top-of-frame banner: prompt + verdict + status.
    banner_color = _INSPECT_POSITIVE_COLOR if c.verdict == "detected" else _INSPECT_NEGATIVE_COLOR
    cv2.rectangle(out, (0, 0), (w, 64), (0, 0, 0), -1)
    _draw_text_with_outline(
        out, f"PROMPT: {c.object_name}", (12, 34),
        color=_INSPECT_PROMPT_COLOR, scale=1.0, thickness=2,
    )
    _draw_text_with_outline(
        out,
        f"pipeline={c.status}   OWLv2={c.verdict}   "
        f"t={c.time_sec:.2f}s   {c.video_id}",
        (12, 58),
        color=banner_color, scale=0.55, thickness=1,
    )
    return out


def _save_inspect_images(
    candidates: list[InspectCandidate],
    cfg: PipelineConfig,
    n_per_class: int,
    output_dir: Path,
    rng: random.Random,
) -> dict:
    """Pick up to ``n_per_class`` 'detected' and 'not_detected'
    candidates, redraw each frame with overlays, and save them under
    ``output_dir/{positive,negative}/``."""
    import cv2  # type: ignore[import-not-found]

    positives = [c for c in candidates if c.verdict == "detected"]
    negatives = [c for c in candidates if c.verdict == "not_detected"]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    selected_pos = positives[:n_per_class]
    selected_neg = negatives[:n_per_class]

    pos_dir, neg_dir = output_dir / "positive", output_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # Group by video so each video is opened once.
    by_video: dict[str, list[tuple[InspectCandidate, Path, int]]] = defaultdict(list)
    for i, c in enumerate(selected_pos):
        by_video[c.video_id].append((c, pos_dir, i + 1))
    for i, c in enumerate(selected_neg):
        by_video[c.video_id].append((c, neg_dir, i + 1))

    n_pos = n_neg = 0
    for video_id, items in by_video.items():
        video_path = cfg.video_file(video_id)
        if not video_path.exists():
            print(
                f"[owlv2_audit] WARNING: cannot save inspect frames for {video_id}, "
                f"video file missing: {video_path}",
                file=sys.stderr,
            )
            continue
        with VideoFrameReader(video_path, cfg.video_fps) as reader:
            for cand, dest_dir, idx in items:
                frame = reader.read_at(cand.time_sec)
                if frame is None:
                    continue
                img = _draw_inspect_overlay(frame, cand)
                fname = (
                    f"{idx:03d}_{_safe_filename_part(cand.video_id)}"
                    f"_{_safe_filename_part(cand.assoc_id)}"
                    f"_t{cand.time_sec:.2f}"
                    f"_{_safe_filename_part(cand.object_name)}.jpg"
                )
                if cv2.imwrite(str(dest_dir / fname), img):
                    if dest_dir is pos_dir:
                        n_pos += 1
                    else:
                        n_neg += 1

    return {
        "inspect_output_dir": str(output_dir),
        "n_positive_candidates": len(positives),
        "n_negative_candidates": len(negatives),
        "n_positive_saved": n_pos,
        "n_negative_saved": n_neg,
    }


# --- Report builder --------------------------------------------------------

_CAVEATS = (
    "A frame counts as 'detected' iff OWLv2 returns at least one box "
    "above --score-threshold that contains the tracked object's per-frame "
    "projected pixel (expanded by --projection-margin-px on each side to "
    "absorb projection error). When several boxes contain the projection, "
    "the one whose center is closest to it wins; this lets us pick the "
    "right instance in frames with multiple same-class objects. Far-away "
    "same-class boxes are rejected because they don't cover the projected "
    "pixel. Frames where the tracked object has no in-frame projection "
    "are counted as 'skipped' rather than scored. Two aggregate rates "
    "are reported per status: 'detection_rate' is the frame-level rate "
    "(fraction of scored frames OWLv2 detected); 'interval_detection_rate' "
    "is the interval-level rate computed by first collapsing each "
    "interval's frame results via majority vote (ties -> detected; "
    "all-skipped intervals -> skipped) and then taking the fraction of "
    "scored intervals declared detected. Before sampling, the object pool "
    "is pruned by --vocabulary-filter and the duplicate-name filter; this "
    "biases the audit toward OWLv2-tractable, unique-name objects, so the "
    "reported rates apply to that subset, not the full pipeline output. "
    "'out_of_view' is excluded from sampling because the tracked object "
    "projects outside the camera frame, so any same-class box is a "
    "different instance. 'in_motion' is sampled but always skipped because "
    "the pipeline returns no projection during manipulation."
)


def _disagreement_rate(detection_rate: float | None, pipeline_visible: bool | None) -> float | None:
    if detection_rate is None or pipeline_visible is None:
        return None
    return (1.0 - detection_rate) if pipeline_visible else detection_rate


def _empty_counts() -> dict[str, int]:
    return {"n_frames_checked": 0, "n_frames_detected": 0, "n_frames_skipped": 0}


def _interval_verdict(row: dict) -> str:
    """Collapse one interval's per-frame counts into a single verdict.

    Majority vote over the scored frames; ties go to 'detected' (i.e.
    n_detected >= n_not_detected). An interval is 'skipped' if it has
    no scored frames (every frame skipped, or none sampled)."""
    n_det = row["n_frames_detected"]
    n_scored = row["n_frames_checked"] - row["n_frames_skipped"]
    if n_scored <= 0:
        return "skipped"
    n_not_det = n_scored - n_det
    return "detected" if n_det >= n_not_det else "not_detected"


def _build_report(
    *,
    pool: dict[str, list[IntervalSpec]],
    sampled_intervals: dict[str, list[IntervalSpec]],
    results: list[FrameResult],
    n_intervals: int,
    n_samples: int,
    video_ids: list[str],
    missing_videos: list[str],
    model_id: str,
    score_threshold: float,
    seed: int,
    point_margin_px: float,
    vocabulary_filter: str,
    drop_duplicate_names: bool,
    filter_stats: FilterStats,
) -> dict:
    per_status: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    per_interval: dict[tuple, dict] = {}

    for r in results:
        s = per_status[r.sample.status]
        s["n_frames_checked"] += 1
        if r.verdict == "detected":
            s["n_frames_detected"] += 1
        elif r.verdict == "skipped":
            s["n_frames_skipped"] += 1

        row = per_interval.setdefault(r.sample.interval_key, {
            "video_id": r.sample.video_id,
            "assoc_id": r.sample.assoc_id,
            "object_name": r.sample.object_name,
            "status": r.sample.status,
            "start_sec": r.sample.interval_key[2],
            "end_sec": r.sample.interval_key[3],
            "n_frames_checked": 0,
            "n_frames_detected": 0,
            "n_frames_skipped": 0,
            "best_match_score": None,
        })
        row["n_frames_checked"] += 1
        if r.verdict == "detected":
            row["n_frames_detected"] += 1
        elif r.verdict == "skipped":
            row["n_frames_skipped"] += 1
        if r.best_match_score is not None:
            row["best_match_score"] = max(row["best_match_score"] or 0.0, r.best_match_score)

    # Collapse each interval's per-frame counts into a single verdict.
    for row in per_interval.values():
        row["interval_verdict"] = _interval_verdict(row)

    per_condition: list[dict] = []
    for status in AUDITED_STATUSES:
        counts = per_status.get(status, _empty_counts())
        n_checked = counts["n_frames_checked"]
        n_det = counts["n_frames_detected"]
        n_skipped = counts["n_frames_skipped"]
        n_scored = n_checked - n_skipped
        detection_rate = (n_det / n_scored) if n_scored > 0 else None
        pipeline_visible = PIPELINE_SAYS_VISIBLE[status]

        # Interval-level tally. Sampled intervals that produced no
        # per-frame results (e.g. no frames could be sampled inside the
        # span) are counted as 'skipped' so the three counts sum to
        # n_intervals_sampled.
        iv_counts = {"detected": 0, "not_detected": 0, "skipped": 0}
        for iv in sampled_intervals.get(status, []):
            key = (iv.video_id, iv.assoc_id, iv.start_sec, iv.end_sec)
            row = per_interval.get(key)
            verdict = row["interval_verdict"] if row is not None else "skipped"
            iv_counts[verdict] += 1
        n_iv_scored = iv_counts["detected"] + iv_counts["not_detected"]
        iv_rate = (iv_counts["detected"] / n_iv_scored) if n_iv_scored > 0 else None

        per_condition.append({
            "status": status,
            "pipeline_says_visible": pipeline_visible,
            "n_intervals_in_pool": len(pool.get(status, [])),
            "n_intervals_sampled": len(sampled_intervals.get(status, [])),
            "n_frames_checked": n_checked,
            "n_frames_detected": n_det,
            "n_frames_skipped": n_skipped,
            "detection_rate": detection_rate,
            "disagreement_rate": _disagreement_rate(detection_rate, pipeline_visible),
            "n_intervals_detected": iv_counts["detected"],
            "n_intervals_not_detected": iv_counts["not_detected"],
            "n_intervals_skipped": iv_counts["skipped"],
            "interval_detection_rate": iv_rate,
            "interval_disagreement_rate": _disagreement_rate(iv_rate, pipeline_visible),
        })

    interval_rows = sorted(
        per_interval.values(),
        key=lambda r: (r["status"], r["video_id"], r["assoc_id"], r["start_sec"]),
    )

    return {
        "model_id": model_id,
        "score_threshold": score_threshold,
        "point_in_box_margin_px": point_margin_px,
        "vocabulary_filter": vocabulary_filter,
        "drop_duplicate_names": drop_duplicate_names,
        "object_filter_stats": filter_stats.to_dict(),
        "seed": seed,
        "n_intervals_per_condition": n_intervals,
        "n_samples_per_interval": n_samples,
        "videos_audited": list(video_ids),
        "videos_missing_visibility_track": list(missing_videos),
        "per_condition": per_condition,
        "per_interval": interval_rows,
        "caveats": _CAVEATS,
    }


def _print_summary(report: dict) -> None:
    def fmt(rate: float | None) -> str:
        return f"{rate:.3f}" if rate is not None else "  n/a"

    print("[owlv2_audit] per-condition results:")
    for row in report["per_condition"]:
        verdict = row["pipeline_says_visible"]
        if verdict is True:
            tag = "pipeline=visible"
        elif verdict is False:
            tag = "pipeline=not_visible"
        else:
            tag = "pipeline=ambiguous"
        print(
            f"  {row['status']:<40s} "
            f"intervals={row['n_intervals_sampled']:>4d}/{row['n_intervals_in_pool']:<5d} "
            f"frames: det={row['n_frames_detected']:>4d}/{row['n_frames_checked']:<5d} "
            f"(skip={row['n_frames_skipped']:>4d}) rate={fmt(row['detection_rate'])}  "
            f"intervals: det={row['n_intervals_detected']:>3d}/{row['n_intervals_sampled']:<4d} "
            f"(skip={row['n_intervals_skipped']:>3d}) rate={fmt(row['interval_detection_rate'])}  "
            f"({tag})"
        )


# --- Figures ---------------------------------------------------------------

# Two-line, human-readable status labels for figure x-ticks. Falls back
# to ``status.replace("_", " ")`` for any status not listed here.
_STATUS_LABELS: dict[str, str] = {
    "in_view": "in view",
    "in_motion": "in motion",
    "occluded_inside_closed_fixture": "occluded\nin closed fixture",
    "observed_visible_in_open_fixture": "obs. visible\nin open fixture",
    "observed_not_visible_in_open_fixture": "obs. not visible\nin open fixture",
    "assumed_not_visible_in_open_fixture": "assumed not visible\nin open fixture",
    "geometrically_occluded": "geometrically\noccluded",
}


def _claim_rank(pipeline_says_visible: bool | None) -> int:
    """Group order for the figure: visible -> ambiguous -> not visible."""
    if pipeline_says_visible is True:
        return 0
    if pipeline_says_visible is None:
        return 1
    return 2


def _save_figures(report: dict, output_stem: Path) -> dict | None:
    """Render a publication-ready bar chart of detection rates by
    pipeline status (frame-level vs interval-level), save it as PDF +
    PNG next to the JSON report, and return the paths. Returns ``None``
    if matplotlib isn't installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    plt.rcParams.update({
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    })

    rows = sorted(
        report["per_condition"],
        key=lambda r: (_claim_rank(r["pipeline_says_visible"]), r["status"]),
    )
    labels = [_STATUS_LABELS.get(r["status"], r["status"].replace("_", " ")) for r in rows]
    frame_rates = [r["detection_rate"] for r in rows]
    interval_rates = [r["interval_detection_rate"] for r in rows]
    pipeline_says = [r["pipeline_says_visible"] for r in rows]
    n_iv_sampled = [r["n_intervals_sampled"] for r in rows]
    n_iv_scored = [r["n_intervals_detected"] + r["n_intervals_not_detected"] for r in rows]
    n_fr_scored = [r["n_frames_checked"] - r["n_frames_skipped"] for r in rows]

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    x = np.arange(len(rows))
    width = 0.38

    frame_color = "#3b6ea5"     # muted blue
    interval_color = "#d97548"  # muted orange

    ax.bar(
        x - width / 2, [r if r is not None else 0.0 for r in frame_rates], width,
        label="frame-level", color=frame_color, edgecolor="white", linewidth=0.5,
    )
    ax.bar(
        x + width / 2, [r if r is not None else 0.0 for r in interval_rates], width,
        label="interval-level (majority vote)", color=interval_color,
        edgecolor="white", linewidth=0.5,
    )

    # Value labels / 'n/a' markers above each bar.
    for i, fr in enumerate(frame_rates):
        if fr is None:
            ax.text(x[i] - width / 2, 0.02, "n/a", ha="center", va="bottom",
                    fontsize=8, color="dimgray")
        else:
            ax.text(x[i] - width / 2, fr + 0.015, f"{fr:.2f}",
                    ha="center", va="bottom", fontsize=8, color=frame_color)
    for i, ir in enumerate(interval_rates):
        if ir is None:
            ax.text(x[i] + width / 2, 0.02, "n/a", ha="center", va="bottom",
                    fontsize=8, color="dimgray")
        else:
            ax.text(x[i] + width / 2, ir + 0.015, f"{ir:.2f}",
                    ha="center", va="bottom", fontsize=8, color=interval_color)

    # Reference lines: pipeline-expected rate (1.0 if pipeline=visible,
    # 0.0 if pipeline=not_visible). Bar height at the line == perfect
    # agreement; the gap to the line is the disagreement.
    for i, v in enumerate(pipeline_says):
        if v is True:
            ax.hlines(1.0, x[i] - 0.45, x[i] + 0.45,
                      color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        elif v is False:
            ax.hlines(0.0, x[i] - 0.45, x[i] + 0.45,
                      color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Group labels above the axes (one per pipeline-claim group).
    group_xs: dict[object, list[float]] = defaultdict(list)
    for i, v in enumerate(pipeline_says):
        group_xs[v].append(float(x[i]))
    for v, xs in group_xs.items():
        text = (
            "pipeline says: visible" if v is True
            else "pipeline says: not visible" if v is False
            else "pipeline says: ambiguous"
        )
        ax.text(sum(xs) / len(xs), 1.07, text, ha="center", va="bottom",
                fontsize=9, color="dimgray", transform=ax.get_xaxis_transform())

    # Light separators between pipeline-claim groups.
    prev = object()
    for i, v in enumerate(pipeline_says):
        if i > 0 and v != prev:
            ax.axvline(x[i] - 0.5, color="lightgray", linewidth=0.6, zorder=0)
        prev = v

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("OWLv2 detection rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    model_name = (report.get("model_id") or "").rsplit("/", 1)[-1] or "OWLv2"
    ax.set_title(f"OWLv2 detection rate by pipeline status  ({model_name})", pad=26)

    # Sample-size footer below the x-tick labels.
    for i in range(len(rows)):
        ax.text(
            x[i], -0.20,
            f"intervals: {n_iv_scored[i]}/{n_iv_sampled[i]}\nframes: {n_fr_scored[i]}",
            ha="center", va="top", fontsize=7, color="dimgray",
            transform=ax.get_xaxis_transform(),
        )

    ax.legend(loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 1.18))

    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.24)

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


# --- CLI -------------------------------------------------------------------

def _resolve_video_ids(cfg: PipelineConfig, args: argparse.Namespace) -> list[str]:
    ids: list[str] = list(args.video or [])
    for participant in args.participant or []:
        ids.extend(cfg.videos_for_participant(participant))
    if not ids:
        ids = list(cfg.videos)
    if not ids:
        raise ValueError(
            "No videos specified. Pass --video, --participant, or set inputs.videos in the config."
        )
    seen: set[str] = set()
    unique: list[str] = []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            unique.append(vid)
    return unique


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", action="append", default=None,
                        help="Video id, repeatable.")
    parser.add_argument("--participant", action="append", default=None,
                        help="Participant id; expands to all of that participant's videos. Repeatable.")
    parser.add_argument("--n-intervals", type=int, default=50,
                        help="Max intervals to sample per status, pooled across all selected videos. "
                             "If fewer intervals exist for a status, all of them are used.")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Frames to sample within each chosen interval. "
                             "If the interval is shorter than this many frames, all available frames are used.")
    parser.add_argument("--device", type=str, default="mps", help="mps | cpu | cuda")
    parser.add_argument("--model-id", type=str,
                        default="google/owlv2-large-patch14-ensemble",
                        help="HuggingFace OWLv2 checkpoint id.")
    parser.add_argument("--score-threshold", type=float, default=0.10,
                        help="OWLv2 score threshold; boxes above this are candidates for the spatial-match check.")
    parser.add_argument("--projection-margin-px", type=float, default=POINT_IN_BOX_MARGIN_PX,
                        help="Pixel tolerance applied around the projected pixel when checking whether an "
                             "OWLv2 box matches the tracked instance. Larger values absorb more projection "
                             "error but raise the risk of matching nearby unrelated boxes.")
    parser.add_argument("--vocabulary-filter", type=str,
                        choices=list(VOCABULARY_FILTER_MODES), default="lvis",
                        help="Vocabulary pruning applied to candidate objects before sampling. "
                             "'none' keeps everything; 'structural' rejects names with >2 tokens, "
                             "prepositions, ordinals, or digits; 'lvis' adds an LVIS-1203 membership "
                             "check on the full name or its head noun (OWLv2's reference vocabulary).")
    parser.add_argument("--keep-duplicate-names", action="store_true",
                        help="Disable the same-video duplicate-name filter. By default an object is "
                             "dropped if another assoc_id in the same video shares its (normalized) name.")
    parser.add_argument("--inspect-status", type=str, default=None,
                        choices=list(AUDITED_STATUSES),
                        help="Save annotated JPEGs for visual inspection of audit decisions on this "
                             "status. Saves up to --inspect-n 'detected' and 'not_detected' frames under "
                             "<output_root>/owlv2_audit_inspect_<status>/.")
    parser.add_argument("--inspect-n", type=int, default=20,
                        help="Number of positive AND negative frames to save when --inspect-status is set.")
    parser.add_argument("--inspect-output-dir", type=Path, default=None,
                        help="Override the directory for --inspect-status JPEGs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path. Defaults to <output_root>/owlv2_audit.json.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip rendering of summary figures next to the JSON report.")
    return parser.parse_args(argv)


def _render_figures_if_requested(report: dict, out_path: Path, *, enabled: bool) -> None:
    """Render PDF + PNG figures next to *out_path* unless disabled or
    matplotlib is missing. Records the figure paths back into the report
    under 'figures'."""
    if not enabled:
        return
    info = _save_figures(report, out_path.with_suffix(""))
    if info is None:
        print(
            "[owlv2_audit] WARNING: matplotlib not installed; skipping figures.",
            file=sys.stderr,
        )
        return
    report["figures"] = info
    print(f"[owlv2_audit] figures -> {info['pdf']}  +  {info['png']}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    video_ids = _resolve_video_ids(cfg, args)
    rng = random.Random(args.seed)
    drop_duplicates = not args.keep_duplicate_names

    print(f"[owlv2_audit] selected {len(video_ids)} video(s)")
    pool, missing, filter_stats = _build_interval_pool(
        cfg, video_ids,
        vocabulary_filter=args.vocabulary_filter,
        drop_duplicate_names=drop_duplicates,
    )
    if missing:
        print(
            f"[owlv2_audit] WARNING: missing visibility_track.jsonl for "
            f"{len(missing)} video(s): {missing[:5]}{'...' if len(missing) > 5 else ''}",
            file=sys.stderr,
        )

    print(
        f"[owlv2_audit] object filter (vocabulary={args.vocabulary_filter}, "
        f"drop_duplicate_names={drop_duplicates}): "
        f"kept {filter_stats.n_objects_kept}/{filter_stats.n_objects_total} "
        f"(dropped duplicate={filter_stats.n_dropped_duplicate_name}, "
        f"structural={filter_stats.n_dropped_structural}, "
        f"lvis={filter_stats.n_dropped_lvis})"
    )

    sampled_intervals = _sample_intervals(pool, args.n_intervals, rng)
    frame_samples = _build_frame_samples(sampled_intervals, args.n_samples, cfg.video_fps, rng)

    print("[owlv2_audit] pool sizes per status:")
    for status in AUDITED_STATUSES:
        print(
            f"  {status:<40s} pool={len(pool.get(status, [])):>5d} "
            f"sampled={len(sampled_intervals.get(status, [])):>4d}"
        )
    print(f"[owlv2_audit] total frames to check: {len(frame_samples)}")

    out_path = args.output or (cfg.output_root / "owlv2_audit.json")

    all_results: list[FrameResult] = []
    inspect_candidates: list[InspectCandidate] | None = [] if args.inspect_status else None
    inspect_summary: dict | None = None

    if not frame_samples:
        report = _build_report(
            pool=pool, sampled_intervals=sampled_intervals, results=[],
            n_intervals=args.n_intervals, n_samples=args.n_samples,
            video_ids=video_ids, missing_videos=missing,
            model_id=args.model_id, score_threshold=args.score_threshold,
            seed=args.seed, point_margin_px=args.projection_margin_px,
            vocabulary_filter=args.vocabulary_filter,
            drop_duplicate_names=drop_duplicates,
            filter_stats=filter_stats,
        )
        report["caveats"] = (
            "Empty audit pool. Check that visibility_track.jsonl files exist "
            "for the selected videos."
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _render_figures_if_requested(report, out_path, enabled=not args.no_figures)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"[owlv2_audit] empty pool -> {out_path}")
        return

    print(f"[owlv2_audit] loading {args.model_id} on {args.device} ...")
    detector = _OWLv2(model_id=args.model_id, device=args.device,
                      score_threshold=args.score_threshold)

    by_video: dict[str, list[FrameSample]] = defaultdict(list)
    for s in frame_samples:
        by_video[s.video_id].append(s)

    for video_id, samples in tqdm(by_video.items(), desc="videos", unit="video"):
        try:
            all_results.extend(_run_video_samples(
                cfg, video_id, samples, detector, args.projection_margin_px,
                inspect_status=args.inspect_status,
                inspect_candidates=inspect_candidates,
            ))
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            print(f"[owlv2_audit] WARNING: skipping {video_id}: {exc}", file=sys.stderr)

    if args.inspect_status is not None and inspect_candidates is not None:
        inspect_dir = (
            args.inspect_output_dir
            if args.inspect_output_dir is not None
            else cfg.output_root / f"owlv2_audit_inspect_{args.inspect_status}"
        )
        inspect_summary = _save_inspect_images(
            candidates=inspect_candidates, cfg=cfg,
            n_per_class=args.inspect_n, output_dir=inspect_dir, rng=rng,
        )
        print(
            f"[owlv2_audit] inspect ({args.inspect_status}): saved "
            f"{inspect_summary['n_positive_saved']} positive / "
            f"{inspect_summary['n_negative_saved']} negative frames "
            f"(from pools of {inspect_summary['n_positive_candidates']} / "
            f"{inspect_summary['n_negative_candidates']}) "
            f"-> {inspect_summary['inspect_output_dir']}"
        )

    report = _build_report(
        pool=pool, sampled_intervals=sampled_intervals, results=all_results,
        n_intervals=args.n_intervals, n_samples=args.n_samples,
        video_ids=video_ids, missing_videos=missing,
        model_id=detector.model_id, score_threshold=detector.score_threshold,
        seed=args.seed, point_margin_px=args.projection_margin_px,
        vocabulary_filter=args.vocabulary_filter,
        drop_duplicate_names=drop_duplicates,
        filter_stats=filter_stats,
    )
    if inspect_summary is not None:
        report["inspect_dump"] = {"status": args.inspect_status, **inspect_summary}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _render_figures_if_requested(report, out_path, enabled=not args.no_figures)
    out_path.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    print(f"[owlv2_audit] -> {out_path}")


if __name__ == "__main__":
    main()
