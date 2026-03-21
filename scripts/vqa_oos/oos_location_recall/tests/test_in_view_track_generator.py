from __future__ import annotations

from pathlib import Path
import argparse
from io import BytesIO
import subprocess

from PIL import Image, ImageDraw, ImageOps

from in_view_determination import determine_in_view_objects, load_frame_context
from in_view_track_generator import generate_in_view_tracks


DEFAULT_VIDEO_ID = "P01-20240203-184045"
DEFAULT_VIDEO_REL_PATH = "HD-EPIC/Videos/P01/P01-20240203-184045.mp4"
DEFAULT_DATA_ROOT_REL = "../../../../data"
DEFAULT_ANN_ROOT_REL = "../../../../hd-epic-annotations"
DEFAULT_INTERMEDIATE_ROOT = "Intermediate_data"


def _select_evenly(values: list[float], n: int) -> list[float]:
    """Select up to n roughly even samples while preserving order."""
    if len(values) <= n:
        return values
    if n <= 1:
        return [values[len(values) // 2]]

    picks = []
    for i in range(n):
        idx = round(i * (len(values) - 1) / (n - 1))
        picks.append(values[idx])

    # Preserve order while removing accidental duplicates from rounding.
    out: list[float] = []
    seen = set()
    for v in picks:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _pick_example_object(tracks: dict, min_per_class: int = 10) -> str:
    """Pick an object with enough in-view and out-of-view samples."""
    candidates = []
    for assoc_id, tr in tracks.items():
        true_count = sum(1 for x in tr.visibility_samples if x)
        false_count = sum(1 for x in tr.visibility_samples if not x)
        if true_count >= min_per_class and false_count >= min_per_class:
            score = min(true_count, false_count)
            candidates.append((score, assoc_id))

    if not candidates:
        raise RuntimeError(
            "No object has at least 10 in-view and 10 out-of-view samples. "
            "Try increasing video duration window or lowering sampling_fps."
        )

    candidates.sort(reverse=True)
    return candidates[0][1]


def _read_frame_at_time(video_path: Path, t_sec: float) -> Image.Image:
    """Read one video frame at time t using ffmpeg and return RGB PIL image."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"Failed to read frame at t={t_sec:.3f}s: {proc.stderr.decode('utf-8', errors='ignore')}")
    return Image.open(BytesIO(proc.stdout)).convert("RGB")


def _build_grid(
    video_path: Path,
    times_sec: list[float],
    video_id: str,
    assoc_id: str,
    annotations_root: Path,
    fps_for_frame_lookup: float,
    title: str,
    calib_width: int,
    calib_height: int,
) -> Image.Image:
    """Build a labeled 2x5 frame grid for the selected timestamps."""

    cell_w, cell_h = 512, 288
    cells = []

    for t_sec in times_sec:
        frame = _read_frame_at_time(video_path, t_sec)
        states = determine_in_view_objects(
            video_id=video_id,
            time_sec=t_sec,
            annotations_root=annotations_root,
            fps=fps_for_frame_lookup,
            intermediate_root=DEFAULT_INTERMEDIATE_ROOT,
        )
        state = next((s for s in states if s.assoc_id == assoc_id), None)
        if state is None:
            raise RuntimeError(f"Object {assoc_id} not found at t={t_sec:.3f}")

        in_view = bool(state.status == "ok" and state.in_view)
        color = (0, 180, 0) if in_view else (200, 0, 0)
        draw = ImageDraw.Draw(frame)

        # Scale projected pixel from calibration resolution to actual video frame size.
        if in_view and state.projected_pixel is not None:
            w, h = frame.size
            px = int(round(state.projected_pixel[0] * (w / float(calib_width))))
            py = int(round(state.projected_pixel[1] * (h / float(calib_height))))
            draw.ellipse((px - 12, py - 12, px + 12, py + 12), fill=(255, 255, 0), outline=(0, 0, 0), width=2)

        draw.text((14, 14), f"t={t_sec:.2f}s | in_view={in_view}", fill=color)
        frame = frame.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        frame = ImageOps.expand(frame, border=3, fill=color)
        cells.append(frame)

    # Ensure exactly 10 cells for a 2x5 grid.
    while len(cells) < 10:
        blank = Image.new("RGB", (cell_w + 6, cell_h + 6), color=(255, 255, 255))
        ImageDraw.Draw(blank).text((cell_w // 2 - 20, cell_h // 2 - 8), "N/A", fill=(0, 0, 0))
        cells.append(blank)

    cells = cells[:10]
    row1 = Image.new("RGB", (sum(im.width for im in cells[:5]), max(im.height for im in cells[:5])), color=(255, 255, 255))
    x = 0
    for im in cells[:5]:
        row1.paste(im, (x, 0))
        x += im.width

    row2 = Image.new("RGB", (sum(im.width for im in cells[5:10]), max(im.height for im in cells[5:10])), color=(255, 255, 255))
    x = 0
    for im in cells[5:10]:
        row2.paste(im, (x, 0))
        x += im.width

    grid = Image.new("RGB", (row1.width, row1.height + row2.height), color=(255, 255, 255))
    grid.paste(row1, (0, 0))
    grid.paste(row2, (0, row1.height))

    banner = Image.new("RGB", (grid.width, 56), color=(255, 255, 255))
    ImageDraw.Draw(banner).text((16, 18), title, fill=(20, 20, 20))
    full = Image.new("RGB", (grid.width, grid.height + banner.height), color=(255, 255, 255))
    full.paste(banner, (0, 0))
    full.paste(grid, (0, banner.height))
    return full


def main() -> None:
    """Run integration sanity check and save in-view/out-of-view grids."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Real-data integration check for in_view_track_generator + in_view_determination")
    parser.add_argument("--video_id", type=str, default=DEFAULT_VIDEO_ID)
    parser.add_argument("--data_root", type=Path, default=(script_dir / DEFAULT_DATA_ROOT_REL).resolve())
    parser.add_argument("--annotations_root", type=Path, default=(script_dir / DEFAULT_ANN_ROOT_REL).resolve())
    parser.add_argument("--video_path", type=Path, default=None)
    parser.add_argument("--sampling_fps", type=float, default=2.0)
    parser.add_argument("--fps_for_frame_lookup", type=float, default=30.0)
    parser.add_argument("--output_dir", type=Path, default=(script_dir / "../../../outputs/oos_track_debug").resolve())
    args = parser.parse_args()

    video_path = args.video_path
    if video_path is None:
        video_path = args.data_root / DEFAULT_VIDEO_REL_PATH

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not args.annotations_root.exists():
        raise FileNotFoundError(f"Annotations root not found: {args.annotations_root}")
    if subprocess.run(["bash", "-lc", "command -v ffmpeg"], check=False, stdout=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("ffmpeg is required but was not found in PATH.")

    tracks = generate_in_view_tracks(
        video_id=args.video_id,
        annotations_root=args.annotations_root,
        sampling_fps=args.sampling_fps,
        fps_for_frame_lookup=args.fps_for_frame_lookup,
        intermediate_root=DEFAULT_INTERMEDIATE_ROOT,
    )
    if not tracks:
        raise RuntimeError("No visibility tracks generated.")

    assoc_id = _pick_example_object(tracks, min_per_class=10)
    track = tracks[assoc_id]

    in_view_times = [t for t, v in zip(track.sampled_times_sec, track.visibility_samples) if v]
    out_view_times = [t for t, v in zip(track.sampled_times_sec, track.visibility_samples) if not v]

    chosen_in = _select_evenly(in_view_times, 10)
    chosen_out = _select_evenly(out_view_times, 10)

    ctx = load_frame_context(
        video_id=args.video_id,
        time_sec=track.sampled_times_sec[0],
        annotations_root=args.annotations_root,
        fps=args.fps_for_frame_lookup,
        intermediate_root=DEFAULT_INTERMEDIATE_ROOT,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    in_grid = _build_grid(
        video_path=video_path,
        times_sec=chosen_in,
        video_id=args.video_id,
        assoc_id=assoc_id,
        annotations_root=args.annotations_root,
        fps_for_frame_lookup=args.fps_for_frame_lookup,
        title=f"IN-VIEW frames (object={track.name}, assoc_id={assoc_id})",
        calib_width=ctx.image_width,
        calib_height=ctx.image_height,
    )
    out_grid = _build_grid(
        video_path=video_path,
        times_sec=chosen_out,
        video_id=args.video_id,
        assoc_id=assoc_id,
        annotations_root=args.annotations_root,
        fps_for_frame_lookup=args.fps_for_frame_lookup,
        title=f"OUT-OF-VIEW frames (object={track.name}, assoc_id={assoc_id})",
        calib_width=ctx.image_width,
        calib_height=ctx.image_height,
    )

    in_path = args.output_dir / f"{args.video_id}_{assoc_id}_in_view_grid.png"
    out_path = args.output_dir / f"{args.video_id}_{assoc_id}_out_of_view_grid.png"
    in_grid.save(in_path)
    out_grid.save(out_path)

    print("Done.")
    print(f"Video: {video_path}")
    print(f"Chosen object: {track.name} ({assoc_id})")
    print(f"In-view frames shown: {len(chosen_in)}")
    print(f"Out-of-view frames shown: {len(chosen_out)}")
    print(f"Saved in-view grid: {in_path}")
    print(f"Saved out-of-view grid: {out_path}")


if __name__ == "__main__":
    main()
