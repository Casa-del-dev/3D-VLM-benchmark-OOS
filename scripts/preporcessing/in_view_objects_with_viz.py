#!/usr/bin/env python3
from pathlib import Path
import json
import argparse
import math
import cv2

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def choose_track_for_time(tracks, time_sec):
    """
    Priority:
    0. If time is inside a track -> object is in movement -> no stable coordinates
    1. Latest track whose end <= time_sec  -> use latest mask from that track
    2. Otherwise earliest future track     -> use first mask from that track
    """
    past_tracks = []
    future_tracks = []

    for tr in tracks:
        start_t, end_t = tr["time_segment"]

        if start_t <= time_sec <= end_t:
            return tr, "in_motion", None
        elif end_t < time_sec:
            past_tracks.append(tr)
        elif start_t > time_sec:
            future_tracks.append(tr)

    if past_tracks:
        chosen = max(past_tracks, key=lambda tr: tr["time_segment"][1])
        return chosen, "past", None

    if future_tracks:
        chosen = min(future_tracks, key=lambda tr: tr["time_segment"][0])
        return chosen, "future", chosen["time_segment"][0]

    return None, None, None


def get_mask_from_track(mask_info_video, track, pick="latest"):
    masks = []
    for mask_id in track.get("masks", []):
        if mask_id in mask_info_video:
            entry = mask_info_video[mask_id]
            masks.append({
                "mask_id": mask_id,
                "frame_number": entry["frame_number"],
                "3d_location": entry["3d_location"],
                "bbox": entry.get("bbox"),
                "fixture": entry.get("fixture"),
            })

    if not masks:
        return None

    if pick == "latest":
        return max(masks, key=lambda m: m["frame_number"])
    if pick == "first":
        return min(masks, key=lambda m: m["frame_number"])

    raise ValueError(f"Unknown pick mode: {pick}")


def yaml_quote_string(s):
    if s is None:
        return "null"
    s = str(s)
    if any(ch in s for ch in [":", "#", "[", "]", "{", "}", ",", "|", "-", '"', "'"]) or " " in s:
        return '"' + s.replace('"', '\\"') + '"'
    return s


def yaml_list_inline(values):
    if values is None:
        return "null"
    out = []
    for v in values:
        if isinstance(v, float):
            out.append(f"{v:.6f}")
        elif isinstance(v, int):
            out.append(str(v))
        elif isinstance(v, bool):
            out.append("true" if v else "false")
        elif v is None:
            out.append("null")
        else:
            out.append(yaml_quote_string(v))
    return "[" + ", ".join(out) + "]"



def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate which objects are visible in the RGB camera view at time t."
    )

    parser.add_argument("--video", type=str, required=True, help="Video ID, e.g. P01-20240202-110250")
    parser.add_argument("--t", type=float, required=True, help="Time in seconds, e.g. 101.0")
    parser.add_argument("--video_file", type=str, default=None, help="Path to input video (mp4). If provided, frame will be extracted automatically.")
    parser.add_argument("--output", type=str, default="in_view_output.yaml", help="Output YAML filename")
    parser.add_argument("--annotations_root", type=str, default="hd-epic-annotations", help="Root folder of HD-EPIC annotations")
    parser.add_argument("--intermediate_root", type=str, default="Intermediate_data", help="Root folder of intermediate data")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS used to map time_sec to frame_index")

    parser.add_argument(
        "--frame_image",
        type=str,
        default=None,
        help="Optional path to the RGB frame image to annotate."
    )
    parser.add_argument(
        "--viz_output",
        type=str,
        default="in_view_visualization.png",
        help="Path to save the annotated visualization image."
    )
    parser.add_argument(
        "--dot_radius",
        type=int,
        default=7,
        help="Radius of the dot drawn on each visible object."
    )
    parser.add_argument(
        "--draw_labels",
        action="store_true",
        help="If set, draw object names next to the dots."
    )
    return parser.parse_args()


# ----------------------------
# Matrix helpers
# ----------------------------

def to_homogeneous_4x4(T_3x4):
    return [
        [T_3x4[0][0], T_3x4[0][1], T_3x4[0][2], T_3x4[0][3]],
        [T_3x4[1][0], T_3x4[1][1], T_3x4[1][2], T_3x4[1][3]],
        [T_3x4[2][0], T_3x4[2][1], T_3x4[2][2], T_3x4[2][3]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_mul(A, B):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            s = 0.0
            for k in range(4):
                s += A[i][k] * B[k][j]
            out[i][j] = s
    return out


def invert_rigid_4x4(T):
    R = [
        [T[0][0], T[0][1], T[0][2]],
        [T[1][0], T[1][1], T[1][2]],
        [T[2][0], T[2][1], T[2][2]],
    ]
    t = [T[0][3], T[1][3], T[2][3]]

    Rt = [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]],
    ]

    t_inv = [
        -(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2]),
        -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2]),
        -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2]),
    ]

    return [
        [Rt[0][0], Rt[0][1], Rt[0][2], t_inv[0]],
        [Rt[1][0], Rt[1][1], Rt[1][2], t_inv[1]],
        [Rt[2][0], Rt[2][1], Rt[2][2], t_inv[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_point(T, p):
    x, y, z = p
    return [
        T[0][0] * x + T[0][1] * y + T[0][2] * z + T[0][3],
        T[1][0] * x + T[1][1] * y + T[1][2] * z + T[1][3],
        T[2][0] * x + T[2][1] * y + T[2][2] * z + T[2][3],
    ]


# ----------------------------
# RGB calibration only
# ----------------------------

def get_rgb_camera_entry(calibration):
    cameras = calibration.get("cameras")
    if not isinstance(cameras, dict):
        raise KeyError("device_calibration.json does not contain a valid 'cameras' dictionary")

    if "camera-rgb" not in cameras:
        raise KeyError("Could not find 'camera-rgb' in device_calibration.json['cameras']")

    return cameras["camera-rgb"]


def get_rgb_calibration(calibration):
    cam = get_rgb_camera_entry(calibration)

    model_name = cam["model_name"]
    image_size = cam["image_size"]
    T_device_camera_raw = cam["T_device_camera"]
    projection_params = cam["projection_params"]

    if len(T_device_camera_raw) == 3 and len(T_device_camera_raw[0]) == 4:
        T_device_camera = to_homogeneous_4x4(T_device_camera_raw)
    elif len(T_device_camera_raw) == 4 and len(T_device_camera_raw[0]) == 4:
        T_device_camera = T_device_camera_raw
    else:
        raise ValueError("camera-rgb T_device_camera is not 3x4 or 4x4")

    width, height = int(image_size[0]), int(image_size[1])

    return {
        "model_name": model_name,
        "width": width,
        "height": height,
        "T_device_camera": T_device_camera,
        "projection_params": projection_params,
    }


# ----------------------------
# FISHEYE624 projection
# ----------------------------

def project_fisheye624(point_cam, params):
    """
    Approximate implementation of the Aria FISHEYE624 model from the released
    parameter layout:
      [f, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3]

    Returns:
      pixel_xy, depth_forward, valid
    """
    x, y, z = point_cam

    # Assume optical axis is +Z in the camera frame used by the released RGB calibration.
    # If z <= 0, treat as behind camera.
    if z <= 1e-9:
        return None, z, False

    f = float(params[0])
    cu = float(params[1])
    cv = float(params[2])
    k0, k1, k2, k3, k4, k5 = [float(v) for v in params[3:9]]
    p0, p1 = [float(v) for v in params[9:11]]
    s0, s1, s2, s3 = [float(v) for v in params[11:15]]

    a = x / z
    b = y / z
    r = math.sqrt(a * a + b * b)

    if r < 1e-12:
        u = cu
        v = cv
        return [u, v], z, True

    theta = math.atan(r)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta10 = theta8 * theta2
    theta12 = theta6 * theta6

    theta_d = theta * (
        1.0
        + k0 * theta2
        + k1 * theta4
        + k2 * theta6
        + k3 * theta8
        + k4 * theta10
        + k5 * theta12
    )

    scale = theta_d / r
    xr = a * scale
    yr = b * scale

    rr = xr * xr + yr * yr
    rr2 = rr * rr

    x_tan = (2.0 * xr * xr + rr) * p0 + 2.0 * xr * yr * p1
    y_tan = (2.0 * yr * yr + rr) * p1 + 2.0 * xr * yr * p0

    x_prism = s0 * rr + s1 * rr2
    y_prism = s2 * rr + s3 * rr2

    xd = xr + x_tan + x_prism
    yd = yr + y_tan + y_prism

    u = f * xd + cu
    v = f * yd + cv
    return [u, v], z, True


def point_in_image(pixel_xy, width, height):
    if pixel_xy is None:
        return False
    u, v = pixel_xy
    return (0.0 <= u < width) and (0.0 <= v < height)


def find_closest_frame_entry(framewise_rows, time_sec, fps):
    target_frame = int(round(time_sec * fps))

    best = None
    best_dist = None

    for row in framewise_rows:
        frame_idx = row.get("frame_index")
        T_world_device = row.get("T_world_device")
        if frame_idx is None or T_world_device is None:
            continue

        dist = abs(frame_idx - target_frame)
        if best is None or dist < best_dist:
            best = row
            best_dist = dist

    return best, target_frame, best_dist


# ----------------------------
# Visualization helpers
# ----------------------------

def extract_frame_from_video(video_path: str, time_sec: float):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Invalid FPS from video")

    frame_idx = int(round(time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame at t={time_sec}s (frame {frame_idx})")

    # Convert BGR → RGB for consistency with PIL
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame, frame_idx, fps

def scale_point_between_sizes(pixel_xy, source_size, target_size):
    if pixel_xy is None:
        return None
    src_w, src_h = source_size
    dst_w, dst_h = target_size
    if src_w <= 0 or src_h <= 0:
        return pixel_xy
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    return [pixel_xy[0] * sx, pixel_xy[1] * sy]


def draw_text_with_background(draw, xy, text, font):
    if not text:
        return
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    bg_box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(bg_box, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def create_visualization(video_file, t, viz_output_path, object_entries, calib_width, calib_height, dot_radius=7, draw_labels=False):
    if Image is None:
        raise RuntimeError("Pillow is required for visualization. Install it with: pip install pillow")

    video_file = Path(video_file)
    viz_output_path = Path(viz_output_path)

    #image = Image.open(frame_image_path).convert("RGB")
    frame_np, frame_idx_used, fps = extract_frame_from_video(video_file, t)
    image = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    target_w, target_h = image.size
    if (target_w, target_h) != (calib_width, calib_height):
        print(
            f"Warning: frame image size {target_w}x{target_h} differs from calibration size "
            f"{calib_width}x{calib_height}. Projected pixels will be scaled to the image size."
        )

    visible_count = 0
    for obj in object_entries:
        if not obj.get("in_view"):
            continue
        pixel_xy = obj.get("projected_pixel")
        if pixel_xy is None:
            continue

        scaled_xy = scale_point_between_sizes(pixel_xy, (calib_width, calib_height), (target_w, target_h))
        x, y = scaled_xy
        visible_count += 1

        draw.ellipse(
            (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
            fill=(255, 0, 0),
            outline=(255, 255, 255),
            width=2,
        )

        if draw_labels:
            label = obj.get("name", "")
            draw_text_with_background(draw, (x + dot_radius + 4, y - dot_radius - 4), label, font)

    image.save(viz_output_path)
    print(f"Saved visualization to {viz_output_path} with {visible_count} visible object(s) marked.")


def main():
    args = parse_args()

    video_id = args.video
    time_sec = args.t
    output_yaml = Path(args.output)

    participant_id = video_id.split("-")[0]

    annotations_root = Path(args.annotations_root)
    intermediate_root = Path(args.intermediate_root)

    assoc_path = annotations_root / "scene-and-object-movements" / "assoc_info.json"
    mask_info_path = annotations_root / "scene-and-object-movements" / "mask_info.json"

    video_dir = annotations_root / intermediate_root / participant_id / video_id
    calib_path = video_dir / "device_calibration.json"
    framewise_path = video_dir / "framewise_info.jsonl"

    assoc_info = load_json(assoc_path)
    mask_info = load_json(mask_info_path)
    calibration = load_json(calib_path)
    framewise_rows = load_jsonl(framewise_path)

    if video_id not in assoc_info:
        msg = f"Video {video_id} not found in assoc_info.json"
        print(msg)
        output_yaml.write_text(f'error: {yaml_quote_string(msg)}\n', encoding="utf-8")
        return

    if video_id not in mask_info:
        msg = f"Video {video_id} not found in mask_info.json"
        print(msg)
        output_yaml.write_text(f'error: {yaml_quote_string(msg)}\n', encoding="utf-8")
        return

    frame_entry, target_frame, frame_dist = find_closest_frame_entry(framewise_rows, time_sec, args.fps)
    if frame_entry is None:
        msg = f"No valid frame entry with T_world_device found for video {video_id}."
        print(msg)
        output_yaml.write_text(f'error: {yaml_quote_string(msg)}\n', encoding="utf-8")
        return

    rgb_calib = get_rgb_calibration(calibration)
    width = rgb_calib["width"]
    height = rgb_calib["height"]
    model_name = rgb_calib["model_name"]
    T_device_camera = rgb_calib["T_device_camera"]
    projection_params = rgb_calib["projection_params"]

    T_world_device_raw = frame_entry["T_world_device"]
    if len(T_world_device_raw) == 3 and len(T_world_device_raw[0]) == 4:
        T_world_device = to_homogeneous_4x4(T_world_device_raw)
    elif len(T_world_device_raw) == 4 and len(T_world_device_raw[0]) == 4:
        T_world_device = T_world_device_raw
    else:
        msg = "T_world_device has unexpected shape."
        print(msg)
        output_yaml.write_text(f'error: {yaml_quote_string(msg)}\n', encoding="utf-8")
        return

    # Correct RGB camera pose in world:
    # T_world_camera = T_world_device @ T_device_camera
    T_world_camera = mat4_mul(T_world_device, T_device_camera)
    T_camera_world = invert_rigid_4x4(T_world_camera)

    video_objects = assoc_info[video_id]
    mask_info_video = mask_info[video_id]

    print("\n=== VIEW ESTIMATION INFO ===")
    print(f"Video: {video_id}")
    print(f"Time: {time_sec}")
    print(f"Participant: {participant_id}")
    print(f"Target frame from time_sec * fps: {target_frame}")
    print(f"Chosen frame_index: {frame_entry['frame_index']}")
    print(f"Frame distance: {frame_dist}")
    print(f"RGB model: {model_name}")
    print(f"Image size: {width} x {height}")

    object_entries = []

    print("\n=== ALL OBJECTS ===")

    for assoc_id, obj in video_objects.items():
        obj_name = obj["name"]
        tracks = obj["tracks"]

        chosen_track, mode, next_time = choose_track_for_time(tracks, time_sec)

        if chosen_track is None:
            print(f"{obj_name}: no track available around time {time_sec}")
            object_entries.append({
                "name": obj_name,
                "status": "no_track_available",
                "selection_mode": None,
                "track_id": None,
                "time_segment": None,
                "mask_id": None,
                "frame_number": None,
                "fixture": None,
                "world_coordinates": None,
                "camera_coordinates": None,
                "projected_pixel": None,
                "depth_in_camera": None,
                "in_view": None,
                "next_closest_possible_time": None,
                "comment": f"No track available for this object around time {time_sec}.",
            })
            continue

        if mode == "in_motion":
            print(f"{obj_name}: object is in movement at time {time_sec} | time_segment={chosen_track['time_segment']}")
            object_entries.append({
                "name": obj_name,
                "status": "in_motion",
                "selection_mode": mode,
                "track_id": chosen_track["track_id"],
                "time_segment": chosen_track["time_segment"],
                "mask_id": None,
                "frame_number": None,
                "fixture": None,
                "world_coordinates": None,
                "camera_coordinates": None,
                "projected_pixel": None,
                "depth_in_camera": None,
                "in_view": None,
                "next_closest_possible_time": None,
                "comment": "Time falls inside a movement track, so no stable location was used.",
            })
            continue

        pick = "latest" if mode == "past" else "first"
        chosen_mask = get_mask_from_track(mask_info_video, chosen_track, pick=pick)

        if chosen_mask is None:
            print(f"{obj_name}: chosen track has no valid mask in mask_info")
            object_entries.append({
                "name": obj_name,
                "status": "no_valid_mask",
                "selection_mode": mode,
                "track_id": chosen_track["track_id"],
                "time_segment": chosen_track["time_segment"],
                "mask_id": None,
                "frame_number": None,
                "fixture": None,
                "world_coordinates": None,
                "camera_coordinates": None,
                "projected_pixel": None,
                "depth_in_camera": None,
                "in_view": None,
                "next_closest_possible_time": next_time,
                "comment": "Chosen track exists, but no mask from that track was found in mask_info.",
            })
            continue

        xyz_world = chosen_mask["3d_location"]
        xyz_cam = transform_point(T_camera_world, xyz_world)

        pixel_xy = None
        depth = None
        valid_projection = False

        if model_name == "CameraModelType.FISHEYE624":
            pixel_xy, depth, valid_projection = project_fisheye624(xyz_cam, projection_params)
        else:
            # For now, only implement the RGB model that the released MP4 uses.
            valid_projection = False
            depth = xyz_cam[2]

        in_view = valid_projection and point_in_image(pixel_xy, width, height)

        if not valid_projection:
            comment = "Projection invalid or point is behind the RGB camera."
        elif in_view:
            comment = "Projected inside RGB image bounds using camera-rgb calibration."
        else:
            comment = "Projected outside RGB image bounds using camera-rgb calibration."

        print(
            f"{obj_name}: "
            f"world=({xyz_world[0]:.6f}, {xyz_world[1]:.6f}, {xyz_world[2]:.6f}) | "
            f"camera=({xyz_cam[0]:.6f}, {xyz_cam[1]:.6f}, {xyz_cam[2]:.6f}) | "
            f"pixel={pixel_xy if pixel_xy is not None else None} | "
            f"in_view={in_view}"
        )

        object_entries.append({
            "name": obj_name,
            "status": "ok",
            "selection_mode": mode,
            "track_id": chosen_track["track_id"],
            "time_segment": chosen_track["time_segment"],
            "mask_id": chosen_mask["mask_id"],
            "frame_number": chosen_mask["frame_number"],
            "fixture": chosen_mask["fixture"],
            "world_coordinates": xyz_world,
            "camera_coordinates": xyz_cam,
            "projected_pixel": pixel_xy,
            "depth_in_camera": depth,
            "in_view": in_view,
            "next_closest_possible_time": next_time,
            "comment": comment,
        })

    lines = []
    lines.append("# Estimated object visibility at chosen time")
    lines.append(f"video_id: {yaml_quote_string(video_id)}")
    lines.append(f"time_sec: {time_sec:.6f}")
    lines.append(f"participant_id: {yaml_quote_string(participant_id)}")
    lines.append(f"approx_fps_used: {args.fps:.6f}")
    lines.append("")

    lines.append("frame_info:")
    lines.append(f"  requested_frame_index: {target_frame}")
    lines.append(f"  chosen_frame_index: {frame_entry['frame_index']}")
    lines.append(f"  frame_distance: {frame_dist}")
    if args.frame_image is not None:
        lines.append(f"  visualization_frame_path: {yaml_quote_string(args.frame_image)}")
    else:
        lines.append("  visualization_frame_path: null")
    lines.append("")

    lines.append("camera_info:")
    lines.append("  camera_label: camera-rgb")
    lines.append(f"  model_name: {yaml_quote_string(model_name)}")
    lines.append(f"  image_size: {yaml_list_inline([width, height])}")
    lines.append('  projection_model_note: "Uses camera-rgb calibration and FISHEYE624 projection."')
    lines.append("")

    lines.append("objects:")

    for obj in object_entries:
        lines.append(f"  - name: {yaml_quote_string(obj['name'])}")
        lines.append(f"    status: {yaml_quote_string(obj['status'])}")
        lines.append(f"    selection_mode: {yaml_quote_string(obj['selection_mode'])}")
        lines.append(f"    track_id: {yaml_quote_string(obj['track_id'])}")
        lines.append(f"    time_segment: {yaml_list_inline(obj['time_segment'])}")
        lines.append(f"    mask_id: {yaml_quote_string(obj['mask_id'])}")
        lines.append(f"    frame_number: {obj['frame_number'] if obj['frame_number'] is not None else 'null'}")
        lines.append(f"    fixture: {yaml_quote_string(obj['fixture'])}")
        lines.append(f"    world_coordinates: {yaml_list_inline(obj['world_coordinates'])}")
        lines.append(f"    camera_coordinates: {yaml_list_inline(obj['camera_coordinates'])}")
        lines.append(f"    projected_pixel: {yaml_list_inline(obj['projected_pixel'])}")
        lines.append(f"    depth_in_camera: {obj['depth_in_camera']:.6f}" if obj["depth_in_camera"] is not None else "    depth_in_camera: null")
        lines.append(f"    in_view: {'true' if obj['in_view'] else 'false'}" if obj["in_view"] is not None else "    in_view: null")
        lines.append(f"    next_closest_possible_time: {obj['next_closest_possible_time']:.6f}" if obj["next_closest_possible_time"] is not None else "    next_closest_possible_time: null")
        lines.append(f"    comment: {yaml_quote_string(obj['comment'])}")

    if args.video_file is not None:
        lines.append("")
        lines.append("visualization:")
        lines.append(f"  output_image: {yaml_quote_string(str(Path(args.viz_output)))}")
        lines.append(f"  dot_radius: {args.dot_radius}")
        lines.append(f"  draw_labels: {'true' if args.draw_labels else 'false'}")
    else:
        lines.append("")
        lines.append("visualization:")
        lines.append("  output_image: null")
        lines.append("  dot_radius: null")
        lines.append("  draw_labels: null")

    output_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDone. Wrote YAML output to {output_yaml}")

    if args.video_file is not None:
        create_visualization(
            video_file=args.video_file,
            t= args.t,
            viz_output_path=args.viz_output,
            object_entries=object_entries,
            calib_width=width,
            calib_height=height,
            dot_radius=args.dot_radius,
            draw_labels=args.draw_labels,
        )
    else:
        print("Visualization skipped because --frame_image was not provided.")


if __name__ == "__main__":
    main()
