from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow importing repo modules from the parent folder when run from tests/
MODULE_DIR = Path(__file__).resolve().parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from in_view_determination import load_frame_context  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load the review config. Install it with: pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_pillow_modules():
    try:
        from PIL import Image, ImageDraw, ImageOps, ImageFont
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required for review images. Install it with: pip install pillow"
        ) from exc
    return Image, ImageDraw, ImageOps


def _read_frame_at_index(video_path: Path, frame_index: int):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required to read frames. Install it with: pip install opencv-python"
        ) from exc

    Image, _, _ = _load_pillow_modules()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def _scale_pixel(pixel_xy: list[float] | None, frame_size: tuple[int, int], calib_size: tuple[int, int]) -> tuple[int, int] | None:
    if pixel_xy is None:
        return None
    frame_w, frame_h = frame_size
    calib_w, calib_h = calib_size
    if calib_w <= 0 or calib_h <= 0:
        return int(round(pixel_xy[0])), int(round(pixel_xy[1]))
    return (
        int(round(float(pixel_xy[0]) * (frame_w / float(calib_w)))),
        int(round(float(pixel_xy[1]) * (frame_h / float(calib_h)))),
    )


def _draw_marker(draw, xy: tuple[int, int], label: str, color=(255, 215, 0)) -> None:
    x, y = xy
    r = 20
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0), width=4)

    try:
        from PIL import ImageFont
        font_label = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 36)
    except:
        font_label = None

    draw.text((x + 24, y - 18), label, fill=(0, 0, 0), font=font_label)

def _wrap_text(text: str, width: int = 95) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if len(trial) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _format_secs(t: float | None) -> str:
    if t is None:
        return "N/A"
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - 3600 * h - 60 * m
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def _panelize(frame, title: str, body_lines: list[str]):
    Image, ImageDraw, _ = _load_pillow_modules()

    panel_h = 140 + 70 * len(body_lines)
    panel = Image.new("RGB", (frame.width, panel_h), color=(255, 255, 255))

    try:
        from PIL import ImageFont
        font_title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 60)
        font_body = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48)
    except:
        font_title = None
        font_body = None

    d = ImageDraw.Draw(panel)

    # big title
    d.text((14, 10), title, fill=(0, 0, 0), font=font_title)

    # body lines
    y = 80
    for line in body_lines:
        d.text((14, y), line, fill=(20, 20, 20), font=font_body)
        y += 60

    out = Image.new("RGB", (frame.width, panel_h + frame.height), color=(255, 255, 255))
    out.paste(panel, (0, 0))
    out.paste(frame, (0, panel_h))
    return out


def _compose_side_by_side(left, right, top_title: str):
    Image, ImageDraw, _ = _load_pillow_modules()

    content_w = left.width + right.width
    content_h = max(left.height, right.height)

    banner_h = 220
    canvas = Image.new("RGB", (content_w, content_h + banner_h), color=(255, 255, 255))
    banner = Image.new("RGB", (content_w, banner_h), color=(255, 255, 255))

    try:
        from PIL import ImageFont
        font_big = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 70)
    except:
        font_big = None

    d = ImageDraw.Draw(banner)

    y = 20
    for line in _wrap_text(top_title, width=120):
        d.text((14, y), line, fill=(0, 0, 0), font=font_big)
        y += 75

    canvas.paste(banner, (0, 0))
    canvas.paste(left, (0, banner_h))
    canvas.paste(right, (left.width, banner_h))
    return canvas


def _infer_video_path(video_id: str, cfg: dict[str, Any]) -> Path:
    if cfg.get("video_path_template"):
        participant = video_id.split("-")[0]
        return Path(str(cfg["video_path_template"]).format(participant=participant, video_id=video_id))
    data_root = cfg.get("data_root")
    rel_template = cfg.get("video_rel_template", "HD-EPIC/Videos/{participant}/{video_id}.mp4")
    if data_root is None:
        raise ValueError("Provide either data_root + video_rel_template, or video_path_template in the review config.")
    participant = video_id.split("-")[0]
    return Path(data_root) / rel_template.format(participant=participant, video_id=video_id)


def _answer_lines(qobj: dict[str, Any]) -> list[str]:
    qclass = str(qobj.get("question_class", ""))
    meta = qobj.get("answer_metadata", {}) or {}
    choices = qobj.get("choices", []) or []
    correct_idx = qobj.get("correct_idx")

    if qclass == "oos_step1_visibility":
        ans = choices[int(correct_idx)] if correct_idx is not None and choices else str(meta.get("in_view"))
        return [f"Correct answer: {ans}", f"status={meta.get('status')}"]

    if qclass == "oos_step2_last_visible":
        pix = meta.get("projected_pixel")
        pix_str = "N/A" if pix is None else f"({int(round(pix[0]))}, {int(round(pix[1]))})"
        return [
            f"Last visible (sampled): {_format_secs(meta.get('sampled_last_visible_time_in_clip_sec'))} in clip",
            f"Projected pixel: {pix_str}",
            f"Frame index: {meta.get('frame_index')}",
        ]

    if qclass == "oos_step3_fixture":
        ans = choices[int(correct_idx)] if correct_idx is not None and choices else meta.get("correct_fixture")
        return [f"Correct answer: {ans}"]

    if qclass == "oos_cam_relative":
        ans = choices[int(correct_idx)] if correct_idx is not None and choices else "N/A"
        return [f"Correct answer: {ans}", f"camera_coords={meta.get('camera_coordinates')}"]

    if qclass == "oos_step4_camera_pose_change":
        ans = choices[int(correct_idx)] if correct_idx is not None and choices else "N/A"
        yaw = meta.get("yaw_delta_deg")
        yaw_txt = "N/A" if yaw is None else f"{float(yaw):.1f} deg"
        return [f"Correct answer: {ans}", f"yaw delta: {yaw_txt}"]

    return ["Review answer not specialized for this question type yet."]


def _render_question_card(video_path: Path, qid: str, qobj: dict[str, Any], annotations_root: Path, fps: float, intermediate_root: str):
    Image, ImageDraw, ImageOps = _load_pillow_modules()

    video_id = str(qobj["video_id"])
    query_time_sec = float(qobj["query_time_sec"])
    query_ctx = load_frame_context(
        video_id=video_id,
        time_sec=query_time_sec,
        annotations_root=annotations_root,
        fps=fps,
        intermediate_root=intermediate_root,
    )
    query_frame = _read_frame_at_index(video_path, query_ctx.frame_index)
    qdraw = ImageDraw.Draw(query_frame)

    qmeta = qobj.get("answer_metadata", {}) or {}
    qclass = str(qobj.get("question_class", ""))

    query_pixel = None
    if qclass == "oos_step2_last_visible":
        # Step 2 answer belongs on the last visible frame, not the query frame.
        query_note = [
            f"Query time in clip: {_format_secs(qobj.get('query_time_in_clip_sec'))}",
            f"Frame index: {query_ctx.frame_index}",
            "Object is not visible at this query frame.",
        ]
    else:
        query_pixel = qmeta.get("projected_pixel")
        query_note = [
            f"Query time in clip: {_format_secs(qobj.get('query_time_in_clip_sec'))}",
            f"Frame index: {query_ctx.frame_index}",
        ]

    scaled_q = _scale_pixel(query_pixel, query_frame.size, (query_ctx.image_width, query_ctx.image_height))
    if scaled_q is not None:
        _draw_marker(qdraw, scaled_q, "GT")

    query_panel = _panelize(
        query_frame,
        title=f"LEFT: query frame | {qid}",
        body_lines=query_note,
    )

    if qclass == "oos_step2_last_visible":
        last_abs_time = qmeta.get("sampled_last_visible_time_sec")
        if last_abs_time is None:
            right_frame = Image.new("RGB", query_frame.size, color=(255, 255, 255))
            right_panel = _panelize(right_frame, "RIGHT: last visible frame", ["No sampled last-visible frame found."])
        else:
            last_ctx = load_frame_context(
                video_id=video_id,
                time_sec=float(last_abs_time),
                annotations_root=annotations_root,
                fps=fps,
                intermediate_root=intermediate_root,
            )
            last_frame = _read_frame_at_index(video_path, last_ctx.frame_index)
            ldraw = ImageDraw.Draw(last_frame)
            last_pixel = qmeta.get("projected_pixel")
            scaled_l = _scale_pixel(last_pixel, last_frame.size, (last_ctx.image_width, last_ctx.image_height))
            if scaled_l is not None:
                _draw_marker(ldraw, scaled_l, "GT")
            pix = qmeta.get("projected_pixel")
            pix_str = "N/A" if pix is None else f"({int(round(pix[0]))}, {int(round(pix[1]))})"
            right_panel = _panelize(
                last_frame,
                title="RIGHT: last visible frame",
                body_lines=[
                    f"Last visible in clip: {_format_secs(qmeta.get('sampled_last_visible_time_in_clip_sec'))}",
                    f"Frame index: {qmeta.get('frame_index')}",
                    f"Projected pixel: {pix_str}",
                ],
            )
    elif qclass == "oos_step4_camera_pose_change":

        ref_abs_time = (
            qmeta.get("reference_time_sec")
            or qmeta.get("previous_visible_time_sec")
            or qmeta.get("sampled_last_visible_time_sec")
        )

        if ref_abs_time is None:
            right_frame = Image.new("RGB", query_frame.size, color=(255, 255, 255))
            right_panel = _panelize(
                right_frame,
                "RIGHT: reference frame",
                _answer_lines(qobj) + ["Reference frame time not found in metadata."]
            )
        else:
            ref_ctx = load_frame_context(
                video_id=video_id,
                time_sec=float(ref_abs_time),
                annotations_root=annotations_root,
                fps=fps,
                intermediate_root=intermediate_root,
            )
            ref_frame = _read_frame_at_index(video_path, ref_ctx.frame_index)

            ref_draw = ImageDraw.Draw(ref_frame)
            ref_pixel = (
                qmeta.get("reference_projected_pixel")
                or qmeta.get("projected_pixel")
            )
            scaled_ref = _scale_pixel(
                ref_pixel,
                ref_frame.size,
                (ref_ctx.image_width, ref_ctx.image_height),
            )
            if scaled_ref is not None:
                _draw_marker(ref_draw, scaled_ref, "REF")

            right_panel = _panelize(
                ref_frame,
                title="RIGHT: reference frame",
                body_lines=[
                    f"Reference time in clip: {_format_secs(qmeta.get('reference_time_in_clip_sec') or qmeta.get('sampled_last_visible_time_in_clip_sec'))}",
                    f"Frame index: {ref_ctx.frame_index}",
                ] + _answer_lines(qobj),
            )
    else:
        answer_frame = Image.new("RGB", query_frame.size, color=(255, 255, 255))
        right_panel = _panelize(answer_frame, "RIGHT: answer", _answer_lines(qobj))

    top_title = f"{qobj.get('question', '')}"
    full = _compose_side_by_side(query_panel, right_panel, top_title)
    full = ImageOps.expand(full, border=3, fill=(220, 220, 220))
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description="Render question-review images for the staged OOS benchmark")
    parser.add_argument("--config", type=Path, required=True, help="Review config YAML")
    parser.add_argument("--benchmark_json", type=Path, default=None, help="Optional benchmark JSON override")
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg = _load_yaml(cfg_path)
    root = cfg_path.parent

    benchmark_json = args.benchmark_json.resolve() if args.benchmark_json is not None else (root / cfg["benchmark_json"]).resolve()
    output_dir = (root / cfg.get("review_output_dir", "../../outputs/staged_oos_review")).resolve()
    annotations_root = Path(str(cfg["annotations_root"])).resolve() if Path(str(cfg["annotations_root"])).is_absolute() else (root / str(cfg["annotations_root"])).resolve()
    fps = float(cfg.get("fps_for_frame_lookup", 30.0))
    intermediate_root = str(cfg.get("intermediate_root", "Intermediate_data"))

    if not benchmark_json.exists():
        raise FileNotFoundError(f"Benchmark JSON not found: {benchmark_json}")

    with benchmark_json.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"benchmark_json": str(benchmark_json), "images": {}}

    video_path_cache: dict[str, Path] = {}
    for qid, qobj in questions.items():
        video_id = str(qobj["video_id"])
        if video_id not in video_path_cache:
            path = _infer_video_path(video_id, {
                "data_root": (root / cfg["data_root"]).resolve() if cfg.get("data_root") and not Path(str(cfg["data_root"])).is_absolute() else cfg.get("data_root"),
                "video_rel_template": cfg.get("video_rel_template", "HD-EPIC/Videos/{participant}/{video_id}.mp4"),
                "video_path_template": cfg.get("video_path_template"),
            })
            video_path_cache[video_id] = Path(path)

        video_path = video_path_cache[video_id]
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found for {video_id}: {video_path}")

        card = _render_question_card(
            video_path=video_path,
            qid=qid,
            qobj=qobj,
            annotations_root=annotations_root,
            fps=fps,
            intermediate_root=intermediate_root,
        )
        out_path = output_dir / f"{qid}.png"
        card.save(out_path)
        manifest["images"][qid] = str(out_path)

    manifest_path = output_dir / "review_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Rendered {len(questions)} review images")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
