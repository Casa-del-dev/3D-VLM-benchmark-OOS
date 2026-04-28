from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def format_time_hms_1dp(time_sec: float) -> str:
    t = max(0.0, float(time_sec))
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = t - (hours * 3600 + minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"


def build_video_path(data_root: Path, video_id: str) -> Path:
    participant = video_id.split("-")[0]
    return data_root / "HD-EPIC" / "Videos" / participant / f"{video_id}.mp4"


def run_ffmpeg_extract_frame(
    ffmpeg_path: Path,
    video_path: Path,
    time_sec: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-ss",
        str(time_sec),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path} at t={time_sec:.3f}s\n"
            f"{proc.stderr.decode('utf-8', errors='ignore')}"
        )


def _get_font(size: int = 24) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _resize_keep_aspect(img: Image.Image, target_h: int) -> Image.Image:
    scale = target_h / img.height
    target_w = int(round(img.width * scale))
    return img.resize((target_w, target_h))


def sample_frame_times(
    clip_start: float,
    query_time: float,
    num_frames: int,
    rng: random.Random,
    mode: str = "random",
) -> list[float]:
    """
    Always include:
      - first frame at clip_start
      - last frame at query_time

    Middle frames are sampled either uniformly or randomly.
    """
    if query_time <= clip_start:
        return [clip_start]

    if num_frames <= 1:
        return [clip_start]

    if num_frames == 2:
        return [clip_start, query_time]

    inner_n = num_frames - 2
    times: list[float] = [clip_start]

    if mode == "uniform":
        step = (query_time - clip_start) / (num_frames - 1)
        middle_times = [clip_start + i * step for i in range(1, num_frames - 1)]
        times.extend(middle_times)
    else:
        middle_times = sorted(
            rng.uniform(clip_start, query_time)
            for _ in range(inner_n)
        )
        times.extend(middle_times)

    times.append(query_time)

    deduped: list[float] = []
    for t in times:
        if not deduped or abs(deduped[-1] - t) > 1e-3:
            deduped.append(t)
    return deduped


def make_review_grid(
    image_paths: list[Path],
    output_path: Path,
    title_lines: list[str],
    labels: list[str],
) -> None:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    target_h = max(im.height for im in images)
    images = [_resize_keep_aspect(im, target_h) for im in images]

    gap = 16
    pad = 20
    header_h = 170
    label_h = 30

    total_w = sum(im.width for im in images) + gap * (len(images) - 1) + pad * 2
    total_h = header_h + label_h + target_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(24)
    small_font = _get_font(20)

    y = 10
    for i, line in enumerate(title_lines):
        draw.text((20, y), line, fill=(0, 0, 0), font=font if i == 0 else small_font)
        y += 30

    x = pad
    img_y = header_h + label_h
    label_y = header_h - 2

    for im, label in zip(images, labels):
        draw.text((x, label_y), label, fill=(0, 0, 0), font=small_font)
        canvas.paste(im, (x, img_y))
        x += im.width + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review camera_rotation questions with flexible multi-frame sampling")
    parser.add_argument("--questions_json", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--ffmpeg_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--frame_sampling", type=str, default="random", choices=["random", "uniform"])
    args = parser.parse_args()

    questions = load_json(args.questions_json)

    camera_items: list[tuple[str, dict[str, Any]]] = [
        (qid, q)
        for qid, q in questions.items()
        if q.get("question_class") == "camera_rotation"
    ]

    if not camera_items:
        raise ValueError("No camera_rotation questions found in input JSON")

    rng = random.Random(args.seed)
    rng.shuffle(camera_items)
    selected = camera_items[: min(args.num_samples, len(camera_items))]

    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    grids_dir = output_dir / "grids"
    manifest_path = output_dir / "review_manifest.json"

    manifest: list[dict[str, Any]] = []

    for idx, (qid, q) in enumerate(selected):
        video_id = str(q["video_id"])
        video_path = build_video_path(args.data_root, video_id)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        clip_start = float(q["clip_start_time_sec"])
        clip_end = float(q["clip_end_time_sec"])
        query_time = float(q["query_time_sec"])

        frame_times = sample_frame_times(
            clip_start=clip_start,
            query_time=query_time,
            num_frames=args.num_frames,
            rng=rng,
            mode=args.frame_sampling,
        )

        frame_labels = []
        frame_paths: list[Path] = []

        for j, t in enumerate(frame_times):
            if j == 0:
                label = f"start  {format_time_hms_1dp(t)}"
            elif j == len(frame_times) - 1:
                label = f"end    {format_time_hms_1dp(t)}"
            else:
                label = f"mid{j}   {format_time_hms_1dp(t)}"

            frame_labels.append(label)

            fp = frames_dir / f"{idx:02d}_{qid}_{j}.jpg"
            run_ffmpeg_extract_frame(args.ffmpeg_path, video_path, t, fp)
            frame_paths.append(fp)

        signed_deg = float(q.get("camera_signed_rotation_deg", 0.0))
        direction = str(q.get("camera_rotation_direction", ""))
        angle_bucket = str(q.get("camera_rotation_angle_bucket", ""))
        correct_idx = int(q["correct_idx"])
        choices = list(q["choices"])
        question = str(q["question"])

        grid_path = grids_dir / f"{idx:02d}_{qid}.jpg"

        title_lines = [
            f"[{idx+1:02d}] {qid}",
            question,
            f"GT: {direction} | signed rotation: {signed_deg:.2f} deg | angle bucket: {angle_bucket}",
            f"correct choice: {choices[correct_idx]} | video={video_id} | clip={format_time_hms_1dp(clip_start)} -> {format_time_hms_1dp(clip_end)}",
        ]

        make_review_grid(
            image_paths=frame_paths,
            output_path=grid_path,
            title_lines=title_lines,
            labels=frame_labels,
        )

        manifest.append(
            {
                "review_index": idx + 1,
                "question_id": qid,
                "video_id": video_id,
                "grid_path": str(grid_path),
                "frame_paths": [str(p) for p in frame_paths],
                "frame_times_sec": frame_times,
                "question": question,
                "choices": choices,
                "correct_idx": correct_idx,
                "camera_rotation_direction": direction,
                "camera_signed_rotation_deg": signed_deg,
                "camera_rotation_angle_bucket": angle_bucket,
                "human_judgment": "",
                "notes": "",
            }
        )

    save_json(manifest, manifest_path)

    print(f"Selected {len(selected)} camera_rotation questions")
    print(f"Review grids dir: {grids_dir}")
    print(f"Review manifest: {manifest_path}")


if __name__ == "__main__":
    main()