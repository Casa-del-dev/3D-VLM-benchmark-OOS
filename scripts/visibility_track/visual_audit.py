"""Human-in-the-loop visibility scoring for the visibility-track audit.

``owlv2_audit.py`` cross-checks the pipeline against OWLv2 detections, but its
own report warns that disagreement is *triage, not an oracle*. This script
samples frames the same way owlv2_audit does -- so audits stay comparable --
and walks a human through scoring each unit via single keypresses, persisting
after every keystroke so work can pause and resume across sessions.

Four verdicts, mapped to the left-hand home row so the scorer can work blind::

    [f / 1]  visible       -- clearly there
    [d / 2]  occluded      -- there, but hidden by something in front
    [s / 3]  unsure        -- seems to be there but hard to make out
                              (shadow / glare / low light / small / ...)
    [a / 4]  not visible   -- the place is shown but the object is not there
    [b / j]  back          -- revisit the previous unit (allows overwrite)
    [space]  skip          -- defer; the unit comes back next session
    [q/ESC]  quit          -- save and exit

The scoring loop runs in an OpenCV window. Sampled frames are pre-extracted to
JPEGs with a tidy reticle at the projected pixel where the object is supposed
to be. No model output is drawn on the frame -- the human is the ground truth
and showing detections would bias the score.

Two scoring grains are supported via ``--grain``:

- ``frame`` (default) -- one frame per scoring unit.
- ``interval`` -- N chronologically ordered frames from one interval shown as
  a grid; the human gives a single verdict for the whole interval.

Statuses accumulate across runs. Score ``in_view`` today, re-run tomorrow with
``--status geometrically_occluded`` -- the manifest is extended in place and
yesterday's scores are kept untouched.

Files written under ``<output_root>/visual_audit/``::

    visual_audit/
      manifest.json   # accumulated sample set (per-status); never overwrites
                      # existing entries -- only extends with new statuses
      scores.json     # per-item verdicts (atomic write after every keypress)
      frames/<status>/<frame_id>.jpg

Example::

    # Score in_view first (1 frame per interval, 100 intervals):
    python -m scripts.visibility_track.visual_audit --participant P01 --n-intervals 100 --n-samples 1 --status in_view

    # Later, add geometrically_occluded to the same manifest:
    python -m scripts.visibility_track.visual_audit --participant P01 --n-intervals 100 --n-samples 1 --status geometrically_occluded

    # Per-interval scoring, 5 frames per interval shown in a grid:
    python -m scripts.visibility_track.visual_audit --participant P01 --grain interval --n-samples 5

    # Just print stats from existing scores (no UI):
    python -m scripts.visibility_track.visual_audit --report-only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.visibility_track.common import PipelineConfig, load_config  # noqa: E402
    from scripts.visibility_track.detection_refinement import VideoFrameReader  # noqa: E402
    from scripts.visibility_track.in_view_track.in_view_determination import VideoCache  # noqa: E402
    from scripts.visibility_track.owlv2_audit import (  # noqa: E402
        AUDITED_STATUSES,
        PIPELINE_SAYS_VISIBLE,
        VOCABULARY_FILTER_MODES,
        _INSPECT_PROJECTION_COLOR,
        _INSPECT_PROMPT_COLOR,
        _build_frame_samples,
        _build_interval_pool,
        _draw_text_with_outline,
        _resolve_anchor,
        _resolve_video_ids,
        _safe_filename_part,
        _sample_intervals,
    )
else:
    from .common import PipelineConfig, load_config
    from .detection_refinement import VideoFrameReader
    from .in_view_track.in_view_determination import VideoCache
    from .owlv2_audit import (
        AUDITED_STATUSES,
        PIPELINE_SAYS_VISIBLE,
        VOCABULARY_FILTER_MODES,
        _INSPECT_PROJECTION_COLOR,
        _INSPECT_PROMPT_COLOR,
        _build_frame_samples,
        _build_interval_pool,
        _draw_text_with_outline,
        _resolve_anchor,
        _resolve_video_ids,
        _safe_filename_part,
        _sample_intervals,
    )


# --- Constants -------------------------------------------------------------

DEFAULT_CONFIG = Path(__file__).resolve().parent / "visibility_track_config.yaml"
MANIFEST_VERSION = 2
SCORES_VERSION = 1

# Verdicts the scorer can record:
#   visible      -- clearly there
#   occluded     -- there but hidden by something in front
#   unsure       -- seems to be in frame but hard to make out (shadow / glare /
#                   low light / small / etc.)
#   not_visible  -- the place is shown but the object isn't actually there
# Each one is bound to one home-row finger of the LEFT hand so the user can
# score by feel without looking at the keyboard.
SCORE_VISIBLE = "visible"
SCORE_OCCLUDED = "occluded"
SCORE_UNSURE = "unsure"
SCORE_NOT_VISIBLE = "not_visible"
VALID_SCORES: tuple[str, ...] = (
    SCORE_VISIBLE, SCORE_OCCLUDED, SCORE_UNSURE, SCORE_NOT_VISIBLE,
)

# Whether a verdict agrees with the pipeline's "object is visible" claim.
# Used to compute disagreement rates in the summary. ``None`` means the
# verdict is neutral / has no opinion.
_SCORE_IMPLIES_VISIBLE: dict[str, bool | None] = {
    SCORE_VISIBLE: True,
    SCORE_UNSURE: None,
    SCORE_OCCLUDED: False,
    SCORE_NOT_VISIBLE: False,
}

# Home-row layout (left hand: a/s/d/f), index-finger most-pressed:
#   f (index, has tactile bump)  -> visible
#   d (middle)                   -> occluded
#   s (ring)                     -> unsure
#   a (pinky)                    -> not visible
# Numeric 1/2/3/4 are fallback aliases for users who prefer the number row.
# cv2.waitKey returns the raw key code masked with 0xFF.
KEY_VISIBLE = {ord("f"), ord("1")}
KEY_OCCLUDED = {ord("d"), ord("2")}
KEY_UNSURE = {ord("s"), ord("3")}
KEY_NOT_VISIBLE = {ord("a"), ord("4")}
# Navigation: b (left) and j (right index, home row) both map to back so
# either hand can drive the queue.
KEY_BACK = {ord("b"), ord("j")}
KEY_SKIP = {ord(" ")}
KEY_QUIT = {ord("q"), 27}  # 27 = ESC

# Lookup used by the scoring loop to route a keypress to a score.
_KEY_TO_SCORE: dict[int, str] = {
    **{k: SCORE_VISIBLE for k in KEY_VISIBLE},
    **{k: SCORE_OCCLUDED for k in KEY_OCCLUDED},
    **{k: SCORE_UNSURE for k in KEY_UNSURE},
    **{k: SCORE_NOT_VISIBLE for k in KEY_NOT_VISIBLE},
}

GRAIN_FRAME = "frame"
GRAIN_INTERVAL = "interval"
VALID_GRAINS: tuple[str, ...] = (GRAIN_FRAME, GRAIN_INTERVAL)

# Global signature fields that must match between runs for an existing
# manifest to be reused without --reset-manifest. ``statuses`` is NOT in
# here -- statuses accumulate across runs (see ``per_status`` in the
# manifest), so an old in_view-only manifest can be extended later with
# geometrically_occluded items without invalidating the scoring already
# done.
_SIGNATURE_KEYS: tuple[str, ...] = (
    "seed", "n_intervals", "n_samples",
    "vocabulary_filter", "drop_duplicate_names",
    "videos", "grain",
)

# Default for missing keys in old manifest signatures, so previously-built
# manifests remain compatible without forcing a reset.
_SIGNATURE_BACKCOMPAT: dict = {"grain": GRAIN_FRAME}


# --- AuditItem -------------------------------------------------------------

@dataclass(frozen=True)
class AuditFrameRef:
    """One rendered frame belonging to an AuditItem."""
    time_sec: float
    image_path: str  # relative to output_dir


@dataclass(frozen=True)
class AuditItem:
    """A scoring unit. ``frame`` grain -> exactly one frame; ``interval``
    grain -> n_samples chronologically ordered frames sharing the same
    interval bounds."""
    item_id: str
    video_id: str
    assoc_id: str
    object_name: str
    status: str
    start_sec: float
    end_sec: float
    grain: str
    frames: tuple[AuditFrameRef, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AuditItem":
        # Backcompat: pre-grain manifests stored time_sec + image_path
        # directly on the item.
        if "frames" not in d:
            return cls(
                item_id=d["item_id"],
                video_id=d["video_id"],
                assoc_id=d["assoc_id"],
                object_name=d["object_name"],
                status=d["status"],
                start_sec=float(d["start_sec"]),
                end_sec=float(d["end_sec"]),
                grain=d.get("grain", GRAIN_FRAME),
                frames=(AuditFrameRef(
                    time_sec=float(d["time_sec"]),
                    image_path=d["image_path"],
                ),),
            )
        frames = tuple(
            AuditFrameRef(time_sec=float(f["time_sec"]), image_path=f["image_path"])
            for f in d["frames"]
        )
        return cls(
            item_id=d["item_id"],
            video_id=d["video_id"],
            assoc_id=d["assoc_id"],
            object_name=d["object_name"],
            status=d["status"],
            start_sec=float(d["start_sec"]),
            end_sec=float(d["end_sec"]),
            grain=d.get("grain", GRAIN_FRAME),
            frames=frames,
        )


# --- Manifest --------------------------------------------------------------

def _compute_frame_id(
    video_id: str, assoc_id: str,
    start_sec: float, end_sec: float, time_sec: float,
) -> str:
    """Per-frame hash; used as the JPEG filename. Stable across grain
    choices so the same JPEG can back both frame- and interval-grain
    manifests."""
    payload = f"{video_id}|{assoc_id}|{start_sec:.6f}|{end_sec:.6f}|{time_sec:.6f}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _compute_item_id(
    video_id: str, assoc_id: str,
    start_sec: float, end_sec: float, time_sec: float | None,
    grain: str,
) -> str:
    """Scoring-unit hash. For ``frame`` grain it equals the per-frame hash;
    for ``interval`` grain it omits ``time_sec`` so every frame within the
    same interval shares one item_id."""
    if grain == GRAIN_INTERVAL:
        payload = f"{video_id}|{assoc_id}|{start_sec:.6f}|{end_sec:.6f}|interval"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    if time_sec is None:
        raise ValueError("time_sec is required for frame-grain item_id")
    return _compute_frame_id(video_id, assoc_id, start_sec, end_sec, time_sec)


def _frame_relpath(status: str, frame_id: str) -> str:
    return f"frames/{_safe_filename_part(status)}/{frame_id}.jpg"


def _manifest_signature(args: argparse.Namespace, videos: list[str]) -> dict:
    """Global signature -- does NOT include the status set. Statuses are
    accumulated incrementally in ``manifest['per_status']``."""
    return {
        "seed": int(args.seed),
        "n_intervals": int(args.n_intervals),
        "n_samples": int(args.n_samples),
        "vocabulary_filter": args.vocabulary_filter,
        "drop_duplicate_names": (not args.keep_duplicate_names),
        "videos": list(videos),
        "grain": args.grain,
    }


def _status_rng(seed: int, status: str) -> random.Random:
    """Per-status deterministic RNG. Two runs that target the SAME status
    with the same seed get the same sample regardless of what other
    statuses were sampled before or after."""
    payload = f"{seed}|{status}".encode("utf-8")
    derived = int(hashlib.sha1(payload).hexdigest(), 16) & ((1 << 64) - 1)
    return random.Random(derived)


def _sample_status_items(
    cfg: PipelineConfig, args: argparse.Namespace,
    pool: dict, status: str,
) -> tuple[list[AuditItem], int, int]:
    """Sample audit items for one status. Returns (items, pool_size,
    n_intervals_sampled)."""
    status_pool = {status: pool.get(status, [])}
    rng = _status_rng(int(args.seed), status)
    sampled_intervals = _sample_intervals(status_pool, args.n_intervals, rng)
    frame_samples = _build_frame_samples(
        sampled_intervals, args.n_samples, cfg.video_fps, rng,
    )
    grain = args.grain

    items: list[AuditItem] = []
    if grain == GRAIN_INTERVAL:
        by_interval: dict[tuple, list] = defaultdict(list)
        for fs in frame_samples:
            by_interval[fs.interval_key].append(fs)
        for interval_key, group in by_interval.items():
            ordered = sorted(group, key=lambda fs: fs.time_sec)
            head = ordered[0]
            v, a, s, e = interval_key
            item_id = _compute_item_id(v, a, s, e, None, grain)
            frames = tuple(
                AuditFrameRef(
                    time_sec=fs.time_sec,
                    image_path=_frame_relpath(
                        fs.status,
                        _compute_frame_id(v, a, s, e, fs.time_sec),
                    ),
                )
                for fs in ordered
            )
            items.append(AuditItem(
                item_id=item_id, video_id=v, assoc_id=a,
                object_name=head.object_name, status=head.status,
                start_sec=s, end_sec=e, grain=grain, frames=frames,
            ))
    else:
        for fs in frame_samples:
            v, a, s, e = fs.interval_key
            frame_id = _compute_frame_id(v, a, s, e, fs.time_sec)
            items.append(AuditItem(
                item_id=frame_id, video_id=v, assoc_id=a,
                object_name=fs.object_name, status=fs.status,
                start_sec=s, end_sec=e, grain=grain,
                frames=(AuditFrameRef(
                    time_sec=fs.time_sec,
                    image_path=_frame_relpath(fs.status, frame_id),
                ),),
            ))
    return items, len(pool.get(status, [])), len(sampled_intervals.get(status, []))


def _empty_manifest(
    args: argparse.Namespace, video_ids: list[str], filter_stats, missing: list[str],
) -> dict:
    return {
        "version": MANIFEST_VERSION,
        "signature": _manifest_signature(args, video_ids),
        "filter_stats": filter_stats.to_dict(),
        "missing_videos": list(missing),
        "per_status": {},
    }


def _build_or_extend_manifest(
    cfg: PipelineConfig, args: argparse.Namespace,
    video_ids: list[str], existing: dict | None,
) -> tuple[dict, list[str]]:
    """Ensure the manifest covers every status in ``args.status`` (or all
    audited statuses if not specified). Statuses already present in the
    existing manifest are reused as-is; missing ones are sampled and
    appended. Returns (manifest, newly_added_statuses)."""
    drop_duplicates = not args.keep_duplicate_names
    pool, missing_videos, filter_stats = _build_interval_pool(
        cfg, video_ids,
        vocabulary_filter=args.vocabulary_filter,
        drop_duplicate_names=drop_duplicates,
    )
    if missing_videos:
        print(
            f"[visual_audit] WARNING: missing visibility_track.jsonl for "
            f"{len(missing_videos)} video(s): {missing_videos[:5]}"
            f"{'...' if len(missing_videos) > 5 else ''}",
            file=sys.stderr,
        )

    manifest = existing if existing is not None else _empty_manifest(
        args, video_ids, filter_stats, missing_videos,
    )
    manifest.setdefault("per_status", {})
    # Refresh filter_stats / missing_videos when extending (vocab + videos
    # haven't changed -- signature check enforced that -- but we keep them
    # current so the report reflects the most recent run).
    manifest["filter_stats"] = filter_stats.to_dict()
    manifest["missing_videos"] = list(missing_videos)
    manifest["signature"] = _manifest_signature(args, video_ids)
    manifest["version"] = MANIFEST_VERSION

    requested = list(args.status) if args.status else list(AUDITED_STATUSES)
    added: list[str] = []
    for status in requested:
        if status in manifest["per_status"]:
            continue
        items, pool_size, n_sampled = _sample_status_items(cfg, args, pool, status)
        manifest["per_status"][status] = {
            "pool_size": pool_size,
            "n_intervals_sampled": n_sampled,
            "items": [it.to_dict() for it in items],
        }
        added.append(status)
    return manifest, added


def _signatures_match(existing: dict, current: dict) -> bool:
    for k in _SIGNATURE_KEYS:
        ev = existing.get(k, _SIGNATURE_BACKCOMPAT.get(k))
        cv = current.get(k)
        if ev != cv:
            return False
    return True


def _migrate_v1_manifest(v1: dict) -> dict:
    """Convert a pre-v2 manifest (flat ``items`` list, signature with
    ``statuses``) into the v2 ``per_status`` layout. Items keep their
    item_ids, so any scores already in scores.json remain valid."""
    sig = dict(v1.get("signature", {}))
    sig.pop("statuses", None)

    per_status: dict[str, dict] = {}
    pool_sizes = v1.get("pool_sizes_per_status", {}) or {}
    sampled_sizes = v1.get("sampled_intervals_per_status", {}) or {}
    for d in v1.get("items", []):
        status = d.get("status")
        if status is None:
            continue
        bucket = per_status.setdefault(status, {
            "pool_size": int(pool_sizes.get(status, 0)),
            "n_intervals_sampled": int(sampled_sizes.get(status, 0)),
            "items": [],
        })
        bucket["items"].append(d)

    return {
        "version": MANIFEST_VERSION,
        "signature": sig,
        "filter_stats": v1.get("filter_stats", {}),
        "missing_videos": v1.get("missing_videos", []),
        "per_status": per_status,
    }


def _load_or_build_manifest(
    cfg: PipelineConfig, args: argparse.Namespace,
    video_ids: list[str], manifest_path: Path,
) -> dict:
    current_sig = _manifest_signature(args, video_ids)

    existing: dict | None = None
    if manifest_path.exists() and not args.reset_manifest:
        loaded = json.loads(manifest_path.read_text())
        if int(loaded.get("version", 1)) < 2 or "per_status" not in loaded:
            print(
                f"[visual_audit] migrating manifest at {manifest_path} from "
                "v1 (flat items) to v2 (per_status)."
            )
            loaded = _migrate_v1_manifest(loaded)
        existing_sig = loaded.get("signature", {})
        if _signatures_match(existing_sig, current_sig):
            existing = loaded
        else:
            print(
                "[visual_audit] existing manifest signature differs from "
                "current args. Pass --reset-manifest to rebuild, or re-run "
                "with the same args. Differences:",
                file=sys.stderr,
            )
            for k in _SIGNATURE_KEYS:
                if existing_sig.get(k) != current_sig.get(k):
                    print(
                        f"    {k}: existing={existing_sig.get(k)!r} "
                        f"vs current={current_sig.get(k)!r}",
                        file=sys.stderr,
                    )
            sys.exit(2)

    manifest, added = _build_or_extend_manifest(cfg, args, video_ids, existing)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(manifest_path, manifest)
    if existing is None:
        n_items = sum(len(s["items"]) for s in manifest["per_status"].values())
        print(
            f"[visual_audit] wrote manifest with {n_items} items across "
            f"{len(manifest['per_status'])} status(es) -> {manifest_path}"
        )
    elif added:
        n_added = sum(
            len(manifest["per_status"][s]["items"]) for s in added
        )
        print(
            f"[visual_audit] extended manifest with {n_added} new items "
            f"across {len(added)} status(es): {added}"
        )
    else:
        print(f"[visual_audit] using existing manifest: {manifest_path}")
    return manifest


# --- Atomic JSON writer ----------------------------------------------------

def _atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


# --- Manifest -> items helpers --------------------------------------------

def _flatten_items(
    manifest: dict, statuses: list[str] | None = None,
) -> list[AuditItem]:
    """Return AuditItems across ``statuses`` (default: all statuses present
    in the manifest), grouped so consecutive items share context.

    Sort order:

    1. participant (e.g. ``P01``) -- stay in one person's kitchen,
    2. object name -- look at the same object across all its appearances,
    3. video id -- consecutive frames are from the same recording,
    4. pipeline status (in AUDITED_STATUSES order),
    5. assoc id / start time / first-frame time for deterministic stability.

    Item ids do not change, so scores already in scores.json stay valid; only
    the queue order changes between runs."""
    per_status = manifest.get("per_status", {}) or {}
    selected = set(statuses) if statuses else set(per_status.keys())

    items: list[AuditItem] = []
    for status, bucket in per_status.items():
        if status not in selected:
            continue
        for d in bucket.get("items", []):
            items.append(AuditItem.from_dict(d))

    status_rank: dict[str, int] = {s: i for i, s in enumerate(AUDITED_STATUSES)}
    fallback_rank = len(AUDITED_STATUSES)

    def sort_key(item: AuditItem) -> tuple:
        participant = item.video_id.split("-", 1)[0]
        first_time = item.frames[0].time_sec if item.frames else 0.0
        return (
            participant,
            item.object_name,
            item.video_id,
            status_rank.get(item.status, fallback_rank),
            item.assoc_id,
            item.start_sec,
            first_time,
        )

    items.sort(key=sort_key)
    return items


# --- Frame rendering -------------------------------------------------------

_MARKER_RING_R = 22
_MARKER_ARM_OUTER = 28


def _draw_projection_marker(out, u: float, v: float):
    """Tidy reticle for the projected pixel: outer ring + crosshairs with a
    center gap + a tiny white center dot. Drawn with a black halo so it
    reads against any background, anti-aliased for clean edges."""
    import cv2  # type: ignore[import-not-found]

    ui, vi = int(round(u)), int(round(v))
    color = _INSPECT_PROJECTION_COLOR  # BGR orange
    halo = (0, 0, 0)
    center_dot = (255, 255, 255)

    arm_inner = 8

    cv2.circle(out, (ui, vi), _MARKER_RING_R, halo, 4, cv2.LINE_AA)
    cv2.circle(out, (ui, vi), _MARKER_RING_R, color, 2, cv2.LINE_AA)

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        p1 = (ui + dx * arm_inner, vi + dy * arm_inner)
        p2 = (ui + dx * _MARKER_ARM_OUTER, vi + dy * _MARKER_ARM_OUTER)
        cv2.line(out, p1, p2, halo, 4, cv2.LINE_AA)
        cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)

    cv2.circle(out, (ui, vi), 3, halo, -1, cv2.LINE_AA)
    cv2.circle(out, (ui, vi), 2, center_dot, -1, cv2.LINE_AA)


def _draw_pointer_label(
    out, u: float, v: float, object_name: str, *,
    banner_h: int,
):
    """Place a small ``looking for: NAME`` tag adjacent to the reticle.

    Tries top-right / top-left / bottom-right / bottom-left in order and
    falls back to whichever fits inside the visible frame without colliding
    with the banner. Drawn with the standard halo-outline so it reads
    against any background."""
    import cv2  # type: ignore[import-not-found]

    h, w = out.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    label = f"looking for: {object_name!r}"
    scale, thickness = 0.6, 2

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    # Offset from the reticle's outer extent + a small breathing margin.
    off = _MARKER_ARM_OUTER + 8
    pad = 8

    def _fits(x: int, y: int) -> bool:
        return (pad <= x and x + tw <= w - pad
                and banner_h + th + pad <= y and y <= h - pad)

    candidates = (
        (ui + off, vi - off),                    # top-right
        (ui - off - tw, vi - off),               # top-left
        (ui + off, vi + off + th),               # bottom-right
        (ui - off - tw, vi + off + th),          # bottom-left
    )
    x, y = next(((xi, yi) for xi, yi in candidates if _fits(xi, yi)),
                candidates[0])
    # Clamp into frame as a last resort.
    x = max(pad, min(x, w - pad - tw))
    y = max(banner_h + th + pad, min(y, h - pad))

    _draw_text_with_outline(
        out, label, (int(x), int(y)),
        color=_INSPECT_PROMPT_COLOR, scale=scale, thickness=thickness,
    )


def _draw_audit_overlay(
    frame, projected_pixel,
    *,
    object_name: str, status: str, video_id: str, time_sec: float,
):
    """Annotate *frame* for the human scorer.

    Top banner: prompt name (large) plus status, time, and video id on a
    second line. Body: a tidy reticle at the projected pixel (when
    available). If projection is unavailable, the banner switches to a red
    'NO PROJECTION' tag so the scorer can quickly mark the unit 'unsure'."""
    import cv2  # type: ignore[import-not-found]

    out = frame.copy()
    w = out.shape[1]
    banner_h = 56

    cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 0), -1)
    _draw_text_with_outline(
        out, f"PROMPT: {object_name}", (12, 30),
        color=_INSPECT_PROMPT_COLOR, scale=0.85, thickness=2,
    )
    if projected_pixel is None:
        sub_text = f"NO PROJECTION  *  t={time_sec:.2f}s  *  {video_id}"
        sub_color = (60, 60, 220)
    else:
        sub_text = f"{status}  *  t={time_sec:.2f}s  *  {video_id}"
        sub_color = (200, 200, 200)
    _draw_text_with_outline(
        out, sub_text, (12, 50),
        color=sub_color, scale=0.5, thickness=1,
    )

    if projected_pixel is not None:
        u, v = projected_pixel
        _draw_projection_marker(out, u, v)
        _draw_pointer_label(out, u, v, object_name, banner_h=banner_h)
    return out


def _render_missing_frames(
    cfg: PipelineConfig, manifest: dict, output_dir: Path,
) -> int:
    """Extract + annotate any manifest frame missing its JPEG. Returns the
    number of frames newly written. Works for both grains since rendering
    is keyed off ``AuditFrameRef``s, not items."""
    import cv2  # type: ignore[import-not-found]

    items = _flatten_items(manifest)
    by_video: dict[str, list[tuple[AuditItem, AuditFrameRef]]] = defaultdict(list)
    for item in items:
        for ref in item.frames:
            if not (output_dir / ref.image_path).exists():
                by_video[item.video_id].append((item, ref))

    if not by_video:
        return 0

    total_missing = sum(len(v) for v in by_video.values())
    print(
        f"[visual_audit] extracting {total_missing} frame(s) across "
        f"{len(by_video)} video(s) ..."
    )

    n_written = n_skipped = 0
    for video_id, video_refs in by_video.items():
        try:
            cache = VideoCache.build(
                video_id, cfg.annotations_root, str(cfg.intermediate_data_root),
            )
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            print(
                f"[visual_audit] WARNING: cache build failed for {video_id}: {exc}",
                file=sys.stderr,
            )
            n_skipped += len(video_refs)
            continue

        video_path = cfg.video_file(video_id)
        if not video_path.exists():
            print(
                f"[visual_audit] WARNING: video file missing for {video_id}: "
                f"{video_path}",
                file=sys.stderr,
            )
            n_skipped += len(video_refs)
            continue

        with VideoFrameReader(video_path, cfg.video_fps) as reader:
            for item, ref in video_refs:
                try:
                    projected = _resolve_anchor(
                        cfg, video_id, ref.time_sec, item.assoc_id, cache,
                    )
                except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
                    print(
                        f"[visual_audit] WARNING: anchor resolve failed for "
                        f"{item.item_id} t={ref.time_sec:.2f}s: {exc}",
                        file=sys.stderr,
                    )
                    projected = None

                frame = reader.read_at(ref.time_sec)
                if frame is None:
                    n_skipped += 1
                    continue

                img = _draw_audit_overlay(
                    frame, projected,
                    object_name=item.object_name,
                    status=item.status,
                    video_id=item.video_id,
                    time_sec=ref.time_sec,
                )
                dest = output_dir / ref.image_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(dest), img):
                    n_written += 1
                else:
                    print(
                        f"[visual_audit] WARNING: failed to write {dest}",
                        file=sys.stderr,
                    )

    msg = f"[visual_audit] extracted {n_written}/{total_missing} new frame(s)"
    if n_skipped:
        msg += f" ({n_skipped} unreadable / missing inputs)"
    print(msg)
    return n_written


# --- Scores ---------------------------------------------------------------

def _load_scores(path: Path) -> dict:
    if not path.exists():
        return {"version": SCORES_VERSION, "items": {}}
    data = json.loads(path.read_text())
    data.setdefault("items", {})
    return data


def _participant_of(video_id: str) -> str:
    """Strip everything after the first '-' (HD-EPIC video ids are
    ``P<NN>-YYYYMMDD-HHMMSS``)."""
    return video_id.split("-", 1)[0]


def _record_score(scores: dict, item: AuditItem, score: str) -> None:
    """Write a full score entry: participant, video, object, pipeline
    status, the human verdict, and a timestamp."""
    scores["items"][item.item_id] = {
        "participant": _participant_of(item.video_id),
        "video_id": item.video_id,
        "object_name": item.object_name,
        "status": item.status,
        "score": score,
        "scored_at": _dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def _backfill_score_metadata(manifest: dict, scores: dict) -> int:
    """Add participant/video_id/object_name/status to existing score entries
    that lack them (older scores.json formats only had ``score`` +
    ``scored_at``). Returns the number of entries rewritten. Entries whose
    item_id is no longer in the manifest are left untouched."""
    by_id = {it.item_id: it for it in _flatten_items(manifest)}
    items_dict = scores.get("items", {})
    expected_keys = {"participant", "video_id", "object_name", "status",
                     "score", "scored_at"}
    n_updated = 0
    for item_id, rec in list(items_dict.items()):
        item = by_id.get(item_id)
        if item is None:
            continue
        rebuilt = {
            "participant": _participant_of(item.video_id),
            "video_id": item.video_id,
            "object_name": item.object_name,
            "status": item.status,
            "score": rec.get("score"),
            "scored_at": rec.get("scored_at"),
        }
        # Only rewrite when something actually changed. Comparing dicts here
        # is order-insensitive; we replace the entry anyway so the on-disk
        # key order stays canonical (participant first, scored_at last).
        if set(rec.keys()) == expected_keys and all(
            rec.get(k) == rebuilt[k] for k in expected_keys
        ):
            continue
        items_dict[item_id] = rebuilt
        n_updated += 1
    return n_updated


def _first_unscored_index(items: list[AuditItem], scores: dict) -> int:
    scored_ids = scores.get("items", {})
    for i, item in enumerate(items):
        if item.item_id not in scored_ids:
            return i
    return len(items)


# --- Scoring loop ----------------------------------------------------------

def _letterbox_image(img, target_w: int, target_h: int):
    """Resize *img* to fit (target_w, target_h) preserving aspect ratio,
    padded with black."""
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _compose_frames(
    images: list, target_w: int, target_h: int,
    *,
    label_each: bool = True,
):
    """Tile *images* chronologically into a (target_w, target_h) canvas.

    1 image -> just letterbox. 2-6 images -> a near-square grid (``cols =
    ceil(sqrt(n))``). Each cell is letterboxed and (if ``label_each``)
    stamped with its 1-based chronological index, so the human can read
    them left-to-right, top-to-bottom.
    """
    import math
    import numpy as np

    n = len(images)
    if n == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    if n == 1:
        return _letterbox_image(images[0], target_w, target_h)

    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    cell_w = target_w // cols
    cell_h = target_h // rows
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = i // cols, i % cols
        cell = _letterbox_image(img, cell_w, cell_h)
        if label_each:
            _draw_text_with_outline(
                cell, f"{i + 1}/{n}", (10, 28),
                color=(0, 220, 255), scale=0.9, thickness=2,
            )
        y0, x0 = r * cell_h, c * cell_w
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell
    return canvas


def _load_item_frame_images(
    item: AuditItem, output_dir: Path,
) -> list:
    """Load every available JPEG for *item* (sorted chronologically).
    Missing or unreadable frames are silently dropped -- callers should
    skip the item only if the returned list is empty."""
    import cv2  # type: ignore[import-not-found]

    loaded: list[tuple[float, "object"]] = []
    for ref in item.frames:
        path = output_dir / ref.image_path
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        loaded.append((ref.time_sec, img))
    loaded.sort(key=lambda t: t[0])
    return [img for _, img in loaded]


def _make_status_bar(
    width: int, height: int, *,
    idx: int, total: int, item: AuditItem, existing_score: str | None,
):
    """Bottom status strip: progress + unit metadata + keybinding hints."""
    import numpy as np

    bar = np.zeros((height, width, 3), dtype=np.uint8)
    unit_word = "interval" if item.grain == GRAIN_INTERVAL else "frame"
    progress_text = (
        f"{idx + 1}/{total} {unit_word}s   "
        f"status: {item.status}   object: {item.object_name!r}"
    )
    if existing_score:
        progress_text += f"   (already scored: {existing_score})"
    _draw_text_with_outline(
        bar, progress_text, (12, 22),
        color=(220, 220, 220), scale=0.55, thickness=1,
    )
    _draw_text_with_outline(
        bar,
        "[f/1] visible   [d/2] occluded   [s/3] unsure   [a/4] not visible   "
        "[b/j] back   [space] skip   [q/ESC] quit",
        (12, 48),
        color=(180, 180, 180), scale=0.5, thickness=1,
    )
    return bar


def _run_scorer(
    manifest: dict, scores_path: Path,
    output_dir: Path, window_size: tuple[int, int],
    status_filter: list[str] | None = None,
) -> None:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    items = _flatten_items(manifest, status_filter)
    if not items:
        if status_filter:
            print(
                f"[visual_audit] no items in manifest for statuses "
                f"{status_filter}. Re-run without --status to sample more "
                "statuses, or use --reset-manifest."
            )
        else:
            print("[visual_audit] manifest has 0 items -- nothing to score.")
        return

    scores = _load_scores(scores_path)
    cursor = _first_unscored_index(items, scores)
    total = len(items)
    n_initially_scored = sum(1 for it in items if it.item_id in scores["items"])
    unit_word = "interval" if items[0].grain == GRAIN_INTERVAL else "frame"

    if cursor >= total:
        print(
            f"[visual_audit] all {total} {unit_word}s in scope already "
            "scored. Delete scores.json (or edit individual entries) to redo."
        )
        return

    print(
        f"[visual_audit] starting scorer at {unit_word} {cursor + 1}/{total} "
        f"({n_initially_scored} already scored)"
    )

    win_w, win_h = window_size
    bar_h = 60
    img_canvas_h = max(1, win_h - bar_h)
    window_name = "visual_audit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, win_w, win_h)

    while 0 <= cursor < total:
        item = items[cursor]
        frame_imgs = _load_item_frame_images(item, output_dir)
        if not frame_imgs:
            print(
                f"[visual_audit] WARNING: no readable frames for item "
                f"{item.item_id}; skipping.",
                file=sys.stderr,
            )
            cursor += 1
            continue

        if len(frame_imgs) == 1:
            img_canvas = _letterbox_image(frame_imgs[0], win_w, img_canvas_h)
        else:
            img_canvas = _compose_frames(frame_imgs, win_w, img_canvas_h)

        existing = scores["items"].get(item.item_id)
        bar = _make_status_bar(
            win_w, bar_h,
            idx=cursor, total=total, item=item,
            existing_score=(existing["score"] if existing else None),
        )
        canvas = np.vstack([img_canvas, bar])

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in KEY_QUIT:
            break
        if key in _KEY_TO_SCORE:
            _record_score(scores, item, _KEY_TO_SCORE[key])
            _atomic_write_json(scores_path, scores)
            cursor += 1
        elif key in KEY_BACK:
            cursor = max(0, cursor - 1)
        elif key in KEY_SKIP:
            cursor += 1
        # any other key -> redraw same item

    cv2.destroyWindow(window_name)
    print(f"[visual_audit] saved scores -> {scores_path}")


# --- Summary --------------------------------------------------------------

_SUMMARY_CATEGORIES: tuple[str, ...] = (
    SCORE_VISIBLE, SCORE_OCCLUDED, SCORE_UNSURE, SCORE_NOT_VISIBLE,
)


def _print_summary(
    manifest: dict, scores: dict, status_filter: list[str] | None = None,
) -> None:
    items = _flatten_items(manifest, status_filter)
    score_dict = scores.get("items", {})

    def _empty_counts() -> dict[str, int]:
        d: dict[str, int] = {c: 0 for c in _SUMMARY_CATEGORIES}
        d["unscored"] = 0
        d["other"] = 0  # any unrecognized score value (e.g. legacy)
        return d

    per_status: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    for item in items:
        row = per_status[item.status]
        rec = score_dict.get(item.item_id)
        if rec is None:
            row["unscored"] += 1
            continue
        s = rec.get("score")
        if s in _SUMMARY_CATEGORIES:
            row[s] += 1
        else:
            row["other"] += 1

    def fmt(r: float | None) -> str:
        return f"{r:.3f}" if r is not None else "  n/a"

    print("[visual_audit] per-status summary:")
    for status in AUDITED_STATUSES:
        if status not in per_status:
            continue
        counts = per_status[status]
        n_vis = counts[SCORE_VISIBLE]
        n_occ = counts[SCORE_OCCLUDED]
        n_un = counts[SCORE_UNSURE]
        n_notv = counts[SCORE_NOT_VISIBLE]
        n_unsc = counts["unscored"]
        # 'visible' counts toward the human's "visible" side; 'occluded'
        # and 'not_visible' count toward "not visible". 'unsure' abstains
        # so it doesn't tilt either side of the disagreement metric.
        n_human_vis = n_vis
        n_human_not = n_occ + n_notv
        n_decided = n_human_vis + n_human_not
        rate = (n_human_vis / n_decided) if n_decided > 0 else None
        ps = PIPELINE_SAYS_VISIBLE.get(status)
        if ps is True:
            tag = "pipeline=visible"
            disagreement = (1.0 - rate) if rate is not None else None
        elif ps is False:
            tag = "pipeline=not_visible"
            disagreement = rate
        else:
            tag = "pipeline=ambiguous"
            disagreement = None
        print(
            f"  {status:<40s} "
            f"vis={n_vis:>4d}  occ={n_occ:>4d}  unsure={n_un:>3d}  "
            f"notv={n_notv:>4d}  unscored={n_unsc:>4d}  "
            f"rate={fmt(rate)}  disagreement={fmt(disagreement)}   ({tag})"
        )


# --- CLI ------------------------------------------------------------------

def _parse_window_size(s: str) -> tuple[int, int]:
    try:
        w_str, h_str = s.lower().split("x")
        return int(w_str), int(h_str)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"Invalid window size {s!r}; expected WxH (e.g. 1280x960)"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", action="append", default=None,
                        help="Video id, repeatable.")
    parser.add_argument("--participant", action="append", default=None,
                        help="Participant id; expands to all of that participant's videos. Repeatable.")
    parser.add_argument("--n-intervals", type=int, default=30,
                        help="Max intervals to sample per status (pooled across videos).")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Frames to sample within each chosen interval.")
    parser.add_argument("--grain", type=str, choices=list(VALID_GRAINS), default=GRAIN_FRAME,
                        help="Scoring grain. 'frame' (default): one verdict per frame, "
                             "fastest finger-on-key rhythm. 'interval': n_samples chronologically "
                             "ordered frames shown as a grid and scored as a single interval "
                             "(matches owlv2_audit's interval-level vote).")
    parser.add_argument("--vocabulary-filter", type=str,
                        choices=list(VOCABULARY_FILTER_MODES), default="lvis",
                        help="Vocabulary pruning applied to candidate objects before sampling. "
                             "Default 'lvis' matches owlv2_audit so the two reports cover the same "
                             "object set and can be compared directly. 'none' keeps everything; "
                             "'structural' rejects names with >2 tokens, prepositions, ordinals, "
                             "or digits; 'lvis' adds an LVIS-1203 membership check on the full "
                             "name or its head noun.")
    parser.add_argument("--keep-duplicate-names", action="store_true",
                        help="Disable the same-video duplicate-name filter. By default an object "
                             "is dropped if another assoc_id in the same video shares its "
                             "(normalized) name -- otherwise the human can't tell which instance "
                             "the projection refers to.")
    parser.add_argument("--status", action="append", default=None,
                        choices=list(AUDITED_STATUSES),
                        help="Repeatable. Statuses to sample (if not already in the manifest) AND "
                             "to filter the scoring loop on. Statuses accumulate across runs: "
                             "scoring 'in_view' today and re-running with '--status "
                             "geometrically_occluded' tomorrow extends the same manifest -- "
                             "previous scores are kept untouched. Default: all audited statuses.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Default: <output_root>/visual_audit/.")
    parser.add_argument("--window-size", type=_parse_window_size, default=(1280, 960),
                        help="OpenCV window size as WxH (default 1280x960).")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip frame extraction (assume JPEGs already on disk).")
    parser.add_argument("--report-only", action="store_true",
                        help="Print per-status summary of existing scores.json and exit; no UI.")
    parser.add_argument("--reset-manifest", action="store_true",
                        help="Rebuild manifest.json from scratch. Existing scores.json is kept on "
                             "disk but scores keyed to items no longer in the manifest are ignored.")
    return parser.parse_args(argv)


def _print_manifest_summary(manifest: dict) -> None:
    """Print object-filter + per-status pool/sample sizes embedded in the manifest."""
    sig = manifest.get("signature", {})
    fs = manifest.get("filter_stats", {})
    if fs:
        print(
            f"[visual_audit] object filter (vocabulary={sig.get('vocabulary_filter')}, "
            f"drop_duplicate_names={sig.get('drop_duplicate_names')}): "
            f"kept {fs.get('n_objects_kept', 0)}/{fs.get('n_objects_total', 0)} "
            f"(dropped duplicate={fs.get('n_dropped_duplicate_name', 0)}, "
            f"structural={fs.get('n_dropped_structural', 0)}, "
            f"lvis={fs.get('n_dropped_lvis_vocabulary', 0)})"
        )
    per_status = manifest.get("per_status", {})
    grain = sig.get("grain", GRAIN_FRAME)
    print("[visual_audit] pool sizes per status:")
    for status in AUDITED_STATUSES:
        bucket = per_status.get(status)
        if bucket is None:
            print(f"  {status:<40s} (not sampled)")
            continue
        print(
            f"  {status:<40s} pool={bucket.get('pool_size', 0):>5d}  "
            f"sampled_intervals={bucket.get('n_intervals_sampled', 0):>4d}  "
            f"items={len(bucket.get('items', [])):>4d}"
        )
    n_items_total = sum(len(s.get("items", [])) for s in per_status.values())
    unit_word = "intervals" if grain == GRAIN_INTERVAL else "frames"
    print(
        f"[visual_audit] grain={grain}  total {unit_word} in manifest: "
        f"{n_items_total} across {len(per_status)} status(es)"
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config)

    output_dir = args.output_dir or (cfg.output_root / "visual_audit")
    manifest_path = output_dir / "manifest.json"
    scores_path = output_dir / "scores.json"

    status_filter: list[str] | None = list(args.status) if args.status else None

    if args.report_only:
        if not manifest_path.exists():
            print(f"[visual_audit] no manifest at {manifest_path}; nothing to report.")
            return
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("version", 1)) < 2 or "per_status" not in manifest:
            manifest = _migrate_v1_manifest(manifest)
        scores = _load_scores(scores_path)
        n_backfilled = _backfill_score_metadata(manifest, scores)
        if n_backfilled:
            _atomic_write_json(scores_path, scores)
            print(
                f"[visual_audit] backfilled metadata for {n_backfilled} "
                f"existing score(s) -> {scores_path}"
            )
        _print_summary(manifest, scores, status_filter)
        return

    video_ids = _resolve_video_ids(cfg, args)
    print(f"[visual_audit] selected {len(video_ids)} video(s); output -> {output_dir}")

    manifest = _load_or_build_manifest(cfg, args, video_ids, manifest_path)
    _print_manifest_summary(manifest)

    if not manifest.get("per_status"):
        print(
            "[visual_audit] empty manifest; nothing to score. Adjust --status / "
            "--n-intervals / --vocabulary-filter and re-run with --reset-manifest."
        )
        return

    # Bring existing scores.json entries up to the current schema before the
    # scorer reloads them; new scores are written with full metadata.
    if scores_path.exists():
        scores = _load_scores(scores_path)
        n_backfilled = _backfill_score_metadata(manifest, scores)
        if n_backfilled:
            _atomic_write_json(scores_path, scores)
            print(
                f"[visual_audit] backfilled metadata for {n_backfilled} "
                f"existing score(s) -> {scores_path}"
            )

    if not args.no_render:
        _render_missing_frames(cfg, manifest, output_dir)

    _run_scorer(
        manifest, scores_path, output_dir, args.window_size,
        status_filter=status_filter,
    )
    scores = _load_scores(scores_path)
    _print_summary(manifest, scores, status_filter)


if __name__ == "__main__":
    main()
