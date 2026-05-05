from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

input_dir = Path("/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/data/HD-EPIC/Videos/P01_raw")
output_dir = Path("/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/data/HD-EPIC/Videos/P01_preprocessed_with_watermark")
output_dir.mkdir(parents=True, exist_ok=True)

add_watermark = True
watermark_style = "plain"  # plain, labeled, token

target_fps = 1
target_size = (168, 168)

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"

def make_text(t):
    ts = format_time(t)
    if watermark_style == "labeled":
        return f"Time\n{ts}"
    if watermark_style == "token":
        return f"<TIME {ts} video 1>"
    return ts

for video_path in input_dir.glob("*.mp4"):
    output_path = output_dir / video_path.name
    print(f"Processing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, target_size)

    frame_idx = 0
    next_sample_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / src_fps

        if t + 1e-6 >= next_sample_time:
            frame = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            if add_watermark:
                draw = ImageDraw.Draw(img)
                text = make_text(t)

                font_size = max(4, int(target_size[0] * 0.035))
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()

                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                margin = 4
                x = target_size[0] - text_w - margin
                y = target_size[1] - text_h - margin

                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)

                overlay_draw.rectangle(
                    [x - 2, y - 1, x + text_w + 2, y + text_h + 1],
                    fill=(0, 0, 0, 100)
                )

                overlay_draw.text(
                    (x, y),
                    text,
                    fill=(255, 255, 255, 200),
                    font=font
                )

                img = Image.alpha_composite(
                    img.convert("RGBA"),
                    overlay
                ).convert("RGB")

            out_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            writer.write(out_frame)

            next_sample_time += 1.0 / target_fps

        frame_idx += 1

    cap.release()
    writer.release()

print("Done.")