"""Visualize the visibility track for one or more videos.

Produces a two-panel figure per video:

  Top:    Gantt-style timeline with:
            • a coverage-density strip showing what fraction of objects are
              simultaneously visible at each moment in the video
            • one row per object, colour-coded by status, with a background
              rail spanning the full video so coverage gaps are visible
            • a "% visible" annotation on the right of every row
  Bottom: Stacked horizontal bars — fraction of total video duration spent in
          each status per object, with duration labels inside wide segments.

Objects are sorted by total confirmed-visible time (most visible at top).
Confirmed-visible = ``in_view`` + ``observed_visible_in_open_fixture``.

Reads ``visibility_track.jsonl`` (stage 4) when present, falling back to
``coarse_visibility_track.jsonl`` (stage 3).  The source file is shown in
the subtitle so it is always clear which pipeline stage the data comes from.

Output: ``visibility_track_plot.png`` in the video's output directory, or at
the path given by ``--output``.

Usage
-----
  python -m scripts.visibility_track.visualize_track --video P01-20240203-184045
  python -m scripts.visibility_track.visualize_track --participant P01
  python -m scripts.visibility_track.visualize_track --video P01-20240203-184045 --output /tmp/plot.png
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.visibility_track.common import PipelineConfig, load_config, read_jsonl  # noqa: E402
else:
    from .common import PipelineConfig, load_config, read_jsonl


DEFAULT_CONFIG = Path(__file__).resolve().parent / "visibility_track_config.yaml"

# ---------------------------------------------------------------------------
# Colour palette — GitHub-dark inspired; semantically grouped by status type
# ---------------------------------------------------------------------------
#   cool bright family  → in-sight / directly visible status types
#   muted dark family   → out-of-sight / occlusion status types
#   neutral grey        → baseline ground state outside the current view

_STATUS_META: Dict[str, Tuple[str, str]] = {
    # in-sight / directly visible -------------------------------------------
    "in_view":                                  ("#3fb950", "In view"),
    "in_motion":                                ("#17b890", "In motion"),
    "observed_visible_in_open_fixture":         ("#58a6ff", "Visible in open fixture"),
    "potentially_visible_inside_open_fixture":  ("#7ee787", "Potentially visible (open fixture)"),
    # out-of-sight / occlusion ----------------------------------------------
    "geometrically_occluded":                   ("#8b6f47", "Occluded by geometry"),
    "occluded_inside_closed_fixture":           ("#6b5b95", "Occluded in closed fixture"),
    "observed_not_visible_in_open_fixture":     ("#7d8590", "Not visible in open fixture"),
    "assumed_not_visible_in_open_fixture":      ("#5b6068", "Assumed not visible (open fixture)"),
    # neutral ground state ---------------------------------------------------
    "out_of_view":                              ("#4b5563", "Out of view"),
    "unobservable_no_data":                     ("#374151", "Unobservable (no data)"),
}

_STATUS_ORDER = [
    "in_view",
    "observed_visible_in_open_fixture",
    "potentially_visible_inside_open_fixture",
    "in_motion",
    "geometrically_occluded",
    "occluded_inside_closed_fixture",
    "observed_not_visible_in_open_fixture",
    "assumed_not_visible_in_open_fixture",
    "out_of_view",
    "unobservable_no_data",
]

# Statuses that count as "confirmed visible" for the % annotation and density strip.
_VISIBLE = frozenset({"in_view", "observed_visible_in_open_fixture"})

# Theme colours
_BG    = "#0d1117"   # figure background
_PANEL = "#161b22"   # axes background
_GRID  = "#21262d"   # grid lines
_SPINE = "#30363d"   # axis spines
_RAIL  = "#21262d"   # full-width background rail inside Gantt rows
_TEXT  = "#c9d1d9"   # tick / axis labels
_MUTED = "#8b949e"   # secondary labels, panel headers
_TITLE = "#f0f6fc"   # video ID title


def _color(status: str) -> str:
    meta = _STATUS_META.get(status)
    return meta[0] if meta else "#4a5568"


def _label(status: str) -> str:
    meta = _STATUS_META.get(status)
    return meta[1] if meta else status


def _fmt_time(sec: float) -> str:
    """Format a duration as MM:SS."""
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m:02d}:{s:02d}"


def _nice_tick_step(duration: float) -> int:
    """Return a human-friendly x-axis tick interval in seconds (target ≤14 ticks)."""
    for step in (30, 60, 120, 180, 300, 600):
        if duration / step <= 14:
            return step
    return 600


def _contrast_text(hex_bg: str) -> str:
    """Return black or light-grey for readable text on top of hex_bg."""
    h = hex_bg.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    # Relative luminance (WCAG formula)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0d1117" if lum > 0.35 else "#c9d1d9"


def _truncate(name: str, max_len: int = 26) -> str:
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_intervals(out_dir: Path) -> Tuple[List[dict], str]:
    """Return (rows, source_filename) from the most complete available output."""
    for fn in ("visibility_track.jsonl", "coarse_visibility_track.jsonl"):
        p = out_dir / fn
        if p.exists():
            return read_jsonl(p), fn
    raise FileNotFoundError(
        f"No visibility track output found in {out_dir}.\n"
        "Run at least stage 3 (combine) first."
    )


def _group_by_object(rows: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        groups[row["assoc_id"]].append(row)
    for ivs in groups.values():
        ivs.sort(key=lambda r: float(r["start_sec"]))
    return dict(groups)


def _interval_span(iv: dict, step: float) -> float:
    """Return the discrete covered duration for one sampled interval."""
    return max(0.0, float(iv["end_sec"]) - float(iv["start_sec"]) + step)


def _visible_secs(intervals: List[dict], step: float) -> float:
    return sum(
        _interval_span(iv, step)
        for iv in intervals
        if iv.get("status") in _VISIBLE
    )


def _sample_step_from_cfg(cfg: PipelineConfig) -> float:
    """Infer the visibility-track sampling step from config, defaulting to 1 s.

    The visibility-track config has changed shape across iterations, so this
    probes a few likely attribute / mapping layouts before falling back to 1 Hz.
    """
    candidates = []

    # Direct attributes on the config object.
    for name in (
        "visibility_track_fps",
        "visibility_fps",
        "track_fps",
        "fps",
    ):
        if hasattr(cfg, name):
            candidates.append(getattr(cfg, name))

    # Nested sections that may be dict-like or attribute-like.
    for section_name in ("visibility_track", "visibility", "tracking"):
        if not hasattr(cfg, section_name):
            continue
        section = getattr(cfg, section_name)
        if isinstance(section, dict):
            for key in ("fps", "sample_fps", "sampling_fps"):
                if key in section:
                    candidates.append(section[key])
        else:
            for key in ("fps", "sample_fps", "sampling_fps"):
                if hasattr(section, key):
                    candidates.append(getattr(section, key))

    for value in candidates:
        try:
            fps = float(value)
        except (TypeError, ValueError):
            continue
        if fps > 0:
            return 1.0 / fps

    return 1.0


def _sorted_objects(
    groups: Dict[str, List[dict]],
    step: float,
) -> List[Tuple[str, str, List[dict]]]:
    entries = [
        (aid, ivs[0].get("object_name", aid), ivs)
        for aid, ivs in groups.items()
    ]
    entries.sort(key=lambda e: (_visible_secs(e[2], step), e[1]), reverse=True)
    return entries


def _coverage_density(
    objects: List[Tuple[str, str, List[dict]]],
    total_dur: float,
    step: float,
    n_buckets: int = 600,
) -> np.ndarray:
    """Fraction [0, 1] of objects simultaneously in a *visible* state per time bucket.

    Each interval covers [start_sec, end_sec + step) — the extra *step* ensures
    the last sample in a run fills its full bin rather than ending one sample-width
    short of the next interval.  Used for the coverage-density heatmap strip.
    """
    n = len(objects)
    if n == 0 or total_dur <= 0:
        return np.zeros(n_buckets)

    bucket_w = total_dur / n_buckets
    counts = np.zeros(n_buckets)
    for _, _, intervals in objects:
        for iv in intervals:
            if iv.get("status") not in _VISIBLE:
                continue
            t0 = float(iv["start_sec"])
            t1 = float(iv["end_sec"]) + step   # inclusive sampled bin coverage
            b0 = max(0, int(t0 / bucket_w))
            b1 = min(n_buckets - 1, int(t1 / bucket_w))
            counts[b0 : b1 + 1] += 1

    return counts / n


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def _style_ax(ax) -> None:
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE)
        spine.set_linewidth(0.5)
    ax.tick_params(colors=_TEXT)
    ax.spines["right"].set_visible(False)


def _build_figure(
    objects: List[Tuple[str, str, List[dict]]],
    video_id: str,
    duration_sec: float,
    source_file: str,
    step: float = 1.0,
):
    # Defer import so the module can be loaded without matplotlib installed.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    n = len(objects)

    # total_dur is the discrete extent implied by the sampling step: a sample
    # stamped at duration_sec occupies the bin [duration_sec, duration_sec + step).
    # Using total_dur keeps adjacent sampled states contiguous unless the track
    # truly contains unannotated bins.
    total_dur = duration_sec + step

    # ---- Layout ---------------------------------------------------------
    # The Gantt has n object rows + 1 coverage strip row → scale accordingly.
    gantt_h = max(3.2, (n + 1) * 0.44)
    bar_h   = max(1.6, n * 0.22)
    fig_h   = gantt_h + bar_h + 2.2   # title strip + legend strip + hspace

    fig, (ax_g, ax_b) = plt.subplots(
        2, 1,
        figsize=(16, fig_h),
        facecolor=_BG,
        gridspec_kw={"height_ratios": [gantt_h, bar_h]},
    )
    fig.subplots_adjust(left=0.175, right=0.925, top=0.90, bottom=0.10, hspace=0.40)

    _style_ax(ax_g)
    _style_ax(ax_b)

    # ================================================================== #
    # Panel 1 — Gantt timeline
    # ================================================================== #

    # -- Coverage-density heatmap strip (y = n, above all object rows) ----
    #    Shows the fraction of objects simultaneously visible at each moment.
    density = _coverage_density(objects, total_dur, step)
    cov_cmap = mcolors.LinearSegmentedColormap.from_list(
        "cov", [_GRID, "#3fb950"]
    )
    ax_g.imshow(
        density[np.newaxis, :],
        aspect="auto",
        extent=[0, total_dur, n - 0.32, n + 0.32],
        cmap=cov_cmap,
        vmin=0.0,
        vmax=1.0,
        zorder=3,
        interpolation="bilinear",
    )

    # Label using a blended transform: x in axes [0,1], y in data coords.
    ax_g.text(
        -0.005, n,
        "Coverage",
        transform=blended_transform_factory(ax_g.transAxes, ax_g.transData),
        fontsize=7, color=_MUTED, fontstyle="italic",
        va="center", ha="right", clip_on=False,
    )

    # Separator between coverage strip and object rows.
    ax_g.axhline(n - 0.48, color=_SPINE, linewidth=0.7, zorder=5)

    # -- Per-object rows ---------------------------------------------------
    for i, (_, name, intervals) in enumerate(objects):
        y = n - 1 - i

        # Subtle alternating tint.
        if i % 2 == 0:
            ax_g.axhspan(y - 0.5, y + 0.5, color="#ffffff", alpha=0.018, zorder=0)

        # Full-width background rail so coverage gaps are visible.
        ax_g.barh(y, total_dur, left=0, height=0.52,
                  color=_RAIL, linewidth=0, zorder=1)

        # Actual status intervals — each bar extends one step past its last
        # sample so adjacent intervals share no gap.
        for iv in intervals:
            t0 = float(iv["start_sec"])
            t1 = float(iv["end_sec"]) + step
            ax_g.barh(
                y, t1 - t0, left=t0,
                height=0.54,
                color=_color(iv.get("status", "")),
                linewidth=0,
                zorder=2,
            )

        # % visible annotation — right of each row.
        vis_secs = sum(
            _interval_span(iv, step)
            for iv in intervals
            if iv.get("status") in _VISIBLE
        )
        vis_pct = vis_secs / total_dur * 100
        if vis_pct >= 20:
            ann_color = "#3fb950"
        elif vis_pct >= 5:
            ann_color = _MUTED
        else:
            ann_color = _SPINE
        ax_g.text(
            total_dur * 1.012, y,
            f"{vis_pct:.0f}%",
            va="center", ha="left",
            fontsize=7, color=ann_color, fontweight="bold",
            clip_on=False,
        )

    # Column header for % column.
    ax_g.text(
        total_dur * 1.012, n - 0.1,
        "vis.",
        va="center", ha="left",
        fontsize=6.5, color=_MUTED, fontstyle="italic",
        clip_on=False,
    )

    # -- X axis (at top) --------------------------------------------------
    tick_step = _nice_tick_step(duration_sec)
    xticks = list(range(0, int(duration_sec) + 1, tick_step))
    ax_g.set_xticks(xticks)
    ax_g.set_xticklabels([_fmt_time(t) for t in xticks], fontsize=7.5, color=_TEXT)
    ax_g.xaxis.tick_top()
    ax_g.xaxis.set_label_position("top")
    ax_g.tick_params(axis="x", length=3, color=_SPINE, pad=3)
    ax_g.set_xlim(0, duration_sec)

    # -- Y axis -----------------------------------------------------------
    ax_g.set_ylim(-0.56, n + 0.50)
    ax_g.set_yticks(list(range(n)))
    ax_g.set_yticklabels(
        [_truncate(name) for _, name, _ in reversed(objects)],
        fontsize=8, color=_TEXT,
    )
    ax_g.tick_params(axis="y", length=0, pad=5)

    ax_g.grid(axis="x", color=_GRID, linewidth=0.4, linestyle="--", alpha=0.7, zorder=0)
    ax_g.spines["top"].set_visible(False)   # ticks replace the spine

    ax_g.text(0, 1.035, "Timeline",
              transform=ax_g.transAxes,
              fontsize=9.5, color=_MUTED, fontstyle="italic", va="bottom")

    # ================================================================== #
    # Panel 2 — Time distribution (stacked, normalised to duration_sec)
    # ================================================================== #

    statuses_used = {
        iv.get("status") for _, _, ivs in objects for iv in ivs
    } - {None}
    ordered_s = [s for s in _STATUS_ORDER if s in statuses_used]

    bottoms = [0.0] * n
    for status in ordered_s:
        col = _color(status)
        widths = [
            sum(
                _interval_span(iv, step)
                for iv in ivs
                if iv.get("status") == status
            ) / (total_dur or 1.0)
            for _, _, ivs in objects
        ]
        ys = [n - 1 - i for i in range(n)]
        bars = ax_b.barh(ys, widths, left=bottoms, height=0.56,
                         color=col, linewidth=0, zorder=2)

        # Duration label inside segments that are wide enough to be readable.
        txt_col = _contrast_text(col)
        for bar, w, b in zip(bars, widths, bottoms):
            if w < 0.07:
                continue
            label = _fmt_time(w * total_dur)
            cx = b + w / 2
            cy = bar.get_y() + bar.get_height() / 2
            ax_b.text(cx, cy, label,
                      va="center", ha="center",
                      fontsize=6.5, color=txt_col,
                      clip_on=True)

        bottoms = [b + w for b, w in zip(bottoms, widths)]

    # Alternating tints.
    for i in range(0, n, 2):
        ax_b.axhspan(n - 1 - i - 0.5, n - 1 - i + 0.5,
                     color="#ffffff", alpha=0.018, zorder=0)

    pct_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax_b.set_xticks(pct_ticks)
    ax_b.set_xticklabels(
        [f"{int(p * 100)}%" for p in pct_ticks],
        fontsize=7.5, color=_TEXT,
    )
    ax_b.set_xlim(0, 1.0)
    ax_b.set_ylim(-0.56, n - 0.44)
    ax_b.set_yticks(list(range(n)))
    ax_b.set_yticklabels(
        [_truncate(name) for _, name, _ in reversed(objects)],
        fontsize=8, color=_TEXT,
    )
    ax_b.tick_params(axis="y", length=0, pad=5)
    ax_b.tick_params(axis="x", length=3, color=_SPINE, pad=3)
    ax_b.grid(axis="x", color=_GRID, linewidth=0.4, linestyle="--", alpha=0.7, zorder=0)

    ax_b.text(0, 1.06, "Time distribution per object",
              transform=ax_b.transAxes,
              fontsize=9.5, color=_MUTED, fontstyle="italic", va="bottom")

    # ================================================================== #
    # Figure-level title + subtitle
    # ================================================================== #

    mean_vis = (
        sum(_visible_secs(ivs, step) for _, _, ivs in objects)
        / (n * total_dur)
        * 100
    ) if n > 0 else 0.0

    fig.text(0.175, 0.966, video_id,
             fontsize=15, fontweight="bold", color=_TITLE,
             va="top", ha="left")
    fig.text(
        0.175, 0.946,
        f"{n} objects  ·  {_fmt_time(duration_sec)}  "
        f"·  avg visible {mean_vis:.0f}%  ·  {source_file}",
        fontsize=8.5, color=_MUTED, va="top", ha="left",
    )

    # ================================================================== #
    # Legend
    # ================================================================== #

    handles = [
        mpatches.Patch(facecolor=_color(s), label=_label(s), linewidth=0)
        for s in ordered_s
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.555, 0.005),
        ncol=min(4, len(handles)),
        fontsize=8,
        frameon=False,
        labelcolor=_TEXT,
        handlelength=1.2,
        handleheight=0.75,
        columnspacing=1.4,
        handletextpad=0.45,
    )

    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_plot(
    cfg: PipelineConfig,
    video_id: str,
    output_path: Optional[Path] = None,
) -> Path:
    out_dir = cfg.video_output_dir(video_id)
    rows, source_file = _load_intervals(out_dir)
    if not rows:
        raise ValueError(f"No interval rows found for {video_id}.")

    groups   = _group_by_object(rows)
    step     = _sample_step_from_cfg(cfg)
    objects  = _sorted_objects(groups, step)
    duration = max(
        float(iv["end_sec"])
        for _, _, ivs in objects
        for iv in ivs
    )

    fig = _build_figure(objects, video_id, duration, source_file, step=step)

    if output_path is None:
        output_path = out_dir / "visibility_track_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=_BG)

    import matplotlib.pyplot as plt
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--video", action="append", default=None,
        help="Video ID to visualize. May be repeated.",
    )
    parser.add_argument(
        "--participant", action="append", default=None,
        help="Visualize all videos for a participant. May be repeated.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output PNG path override (single video only).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config)

    video_ids: List[str] = list(args.video or [])

    if args.participant:
        for participant in args.participant:
            for v in cfg.videos_for_participant(participant):
                if v not in video_ids:
                    video_ids.append(v)

    if not video_ids:
        for participant in cfg.participants:
            for v in cfg.videos_for_participant(participant):
                if v not in video_ids:
                    video_ids.append(v)

    if not video_ids:
        video_ids = cfg.videos

    if not video_ids:
        raise ValueError(
            "No videos to process. "
            "Pass --video, --participant, or populate inputs in the config."
        )

    if args.output is not None and len(video_ids) > 1:
        raise ValueError("--output can only be used when processing a single video.")

    for video_id in video_ids:
        out_path = render_plot(
            cfg=cfg,
            video_id=video_id,
            output_path=args.output if len(video_ids) == 1 else None,
        )
        print(f"[plot] {video_id} -> {out_path}")


if __name__ == "__main__":
    main()
