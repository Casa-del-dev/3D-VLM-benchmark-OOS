from __future__ import annotations

import argparse
import json
import math
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


def make_side_by_side_review_image(
    start_img_path: Path,
    end_img_path: Path,
    output_path: Path,
    title_lines: list[str],
) -> None:
    start_img = Image.open(start_img_path).convert("RGB")
    end_img = Image.open(end_img_path).convert("RGB")

    target_h = max(start_img.height, end_img.height)

    def resize_keep_aspect(img: Image.Image, target_h: int) -> Image.Image:
        scale = target_h / img.height
        target_w = int(round(img.width * scale))
        return img.resize((target_w, target_h))

    start_img = resize_keep_aspect(start_img, target_h)
    end_img = resize_keep_aspect(end_img, target_h)

    gap = 20
    pad = 20
    header_h = 140

    canvas_w = start_img.width + end_img.width + gap + pad * 2
    canvas_h = header_h + target_h + pad * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(24)
    small_font = _get_font(20)

    y = 10
    for i, line in enumerate(title_lines):
        draw.text((20, y), line, fill=(0, 0, 0), font=font if i == 0 else small_font)
        y += 30

    start_x = pad
    img_y = header_h
    end_x = pad + start_img.width + gap

    canvas.paste(start_img, (start_x, img_y))
    canvas.paste(end_img, (end_x, img_y))

    draw.text((start_x, img_y - 28), "clip start frame", fill=(0, 0, 0), font=small_font)
    draw.text((end_x, img_y - 28), "query frame", fill=(0, 0, 0), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_video_path(data_root: Path, video_id: str) -> Path:
    participant = video_id.split("-")[0]
    return data_root / "HD-EPIC" / "Videos" / participant / f"{video_id}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch review camera_rotation_direction questions")
    parser.add_argument(
        "--questions_json",
        type=Path,
        required=True,
        help="Path to generated questions JSON",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Repo data root, e.g. C:/3D-VLM-benchmark-OOS",
    )
    parser.add_argument(
        "--ffmpeg_path",
        type=Path,
        required=True,
        help="Working ffmpeg.exe path",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save review images and manifest",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="How many random camera_rotation_direction questions to review",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    questions = load_json(args.questions_json)

    camera_items: list[tuple[str, dict[str, Any]]] = [
        (qid, q)
        for qid, q in questions.items()
        if q.get("question_class") == "camera_rotation_direction"
    ]

    if not camera_items:
        raise ValueError("No camera_rotation_direction questions found in input JSON")

    rng = random.Random(args.seed)
    rng.shuffle(camera_items)
    selected = camera_items[: min(args.num_samples, len(camera_items))]

    review_dir = args.output_dir
    frames_dir = review_dir / "frames"
    grids_dir = review_dir / "grids"
    manifest_path = review_dir / "review_manifest.json"

    manifest: list[dict[str, Any]] = []

    for idx, (qid, q) in enumerate(selected):
        video_id = str(q["video_id"])
        video_path = build_video_path(args.data_root, video_id)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        clip_start = float(q["clip_start_time_sec"])
        query_time = float(q["query_time_sec"])
        signed_deg = float(q["camera_signed_rotation_deg"])
        direction = str(q["camera_rotation_direction"])
        correct_idx = int(q["correct_idx"])
        choices = list(q["choices"])
        question = str(q["question"])

        start_frame_path = frames_dir / f"{idx:02d}_{qid}_start.jpg"
        query_frame_path = frames_dir / f"{idx:02d}_{qid}_query.jpg"
        grid_path = grids_dir / f"{idx:02d}_{qid}.jpg"

        run_ffmpeg_extract_frame(args.ffmpeg_path, video_path, clip_start, start_frame_path)
        run_ffmpeg_extract_frame(args.ffmpeg_path, video_path, query_time, query_frame_path)

        title_lines = [
            f"[{idx+1:02d}] {qid}",
            question,
            f"GT: {direction} | signed rotation: {signed_deg:.2f} deg | correct choice: {choices[correct_idx]}",
            f"video={video_id} | clip_start={format_time_hms_1dp(clip_start)} | query={format_time_hms_1dp(query_time)}",
        ]

        make_side_by_side_review_image(
            start_img_path=start_frame_path,
            end_img_path=query_frame_path,
            output_path=grid_path,
            title_lines=title_lines,
        )

        manifest.append(
            {
                "review_index": idx + 1,
                "question_id": qid,
                "video_id": video_id,
                "grid_path": str(grid_path),
                "start_frame_path": str(start_frame_path),
                "query_frame_path": str(query_frame_path),
                "question": question,
                "choices": choices,
                "correct_idx": correct_idx,
                "camera_rotation_direction": direction,
                "camera_signed_rotation_deg": signed_deg,
                "human_judgment": "",
                "notes": "",
            }
        )

    save_json(manifest, manifest_path)

    print(f"Selected {len(selected)} camera_rotation_direction questions")
    print(f"Review grids dir: {grids_dir}")
    print(f"Review manifest: {manifest_path}")


if __name__ == "__main__":
    main()