#!/usr/bin/env python3
from pathlib import Path
import json
import argparse
import math
import cv2

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


def find_object_entry(video_objects, object_name):
    for assoc_id, obj in video_objects.items():
        if obj["name"].strip().lower() == object_name.strip().lower():
            return assoc_id, obj
    return None, None


def choose_track_for_time(tracks, time_sec):
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


def get_mask_for_exact_frame(mask_info_video, obj, frame_number):
    """
    Return the object's mask if it has an observation exactly at the queried frame.
    Also returns the owning track metadata when found.
    """
    for tr in obj.get("tracks", []):
        for mask_id in tr.get("masks", []):
            if mask_id not in mask_info_video:
                continue
            entry = mask_info_video[mask_id]
            if int(entry["frame_number"]) == int(frame_number):
                return {
                    "track_id": tr["track_id"],
                    "time_segment": tr["time_segment"],
                    "mask": {
                        "mask_id": mask_id,
                        "frame_number": entry["frame_number"],
                        "3d_location": entry["3d_location"],
                        "bbox": entry.get("bbox"),
                        "fixture": entry.get("fixture"),
                    },
                }
    return None


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
        description=(
            "Express every candidate object A relative to anchor object B in an egocentric "
            "coordinate system whose origin is at B and whose axes are aligned with a chosen "
            "camera viewpoint from a frame where B is visible. Also report whether each A is visible "
            "in that same anchor frame, and optionally save a visualization of the anchored "
            "coordinate system overlaid on a video frame."
        )
    )

    parser.add_argument("--video", type=str, required=True, help="Video ID, e.g. P01-20240202-110250")
    parser.add_argument("--obj_b", type=str, required=True, help='Anchor object B name, e.g. "cutting board"')
    parser.add_argument("--t", type=float, required=True, help="Reference time in seconds")
    parser.add_argument("--output", type=str, default="relative_egocentric_output.yaml", help="Output YAML filename")
    parser.add_argument("--annotations_root", type=str, default="hd-epic-annotations", help="Root folder of HD-EPIC annotations")
    parser.add_argument("--intermediate_root", type=str, default="Intermediate_data", help="Root folder of intermediate data")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS used to map time_sec to frame_index when frame metadata is unavailable")

    parser.add_argument(
        "--camera_label",
        type=str,
        default="camera-rgb",
        help="Camera whose viewpoint/orientation defines the local axes, e.g. camera-rgb or camera-slam-left",
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Optional path to the actual video file. When provided, the script will save an image showing the local anchored coordinate system on the selected frame.",
    )
    parser.add_argument(
        "--vis_output",
        type=str,
        default="anchor_coordinate_visualization.jpg",
        help="Output image path for the visualization.",
    )
    parser.add_argument(
        "--axis_length_m",
        type=float,
        default=0.15,
        help="Axis length in meters for the X/Y/Z arrows drawn from the anchor object.",
    )
    parser.add_argument(
        "--axis_thickness",
        type=int,
        default=3,
        help="Arrow thickness for the visualization.",
    )
    return parser.parse_args()


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


def get_camera_entry(calibration, camera_label):
    cameras = calibration.get("cameras")
    if not isinstance(cameras, dict):
        raise KeyError("device_calibration.json does not contain a valid 'cameras' dictionary")

    if camera_label not in cameras:
        raise KeyError(f"Could not find {camera_label!r} in device_calibration.json['cameras']")

    return cameras[camera_label]


def get_camera_calibration(calibration, camera_label):
    cam = get_camera_entry(calibration, camera_label)

    model_name = cam["model_name"]
    image_size = cam["image_size"]
    T_device_camera_raw = cam["T_device_camera"]
    projection_params = cam["projection_params"]

    if len(T_device_camera_raw) == 3 and len(T_device_camera_raw[0]) == 4:
        T_device_camera = to_homogeneous_4x4(T_device_camera_raw)
    elif len(T_device_camera_raw) == 4 and len(T_device_camera_raw[0]) == 4:
        T_device_camera = T_device_camera_raw
    else:
        raise ValueError(f"{camera_label} T_device_camera is not 3x4 or 4x4")

    width, height = int(image_size[0]), int(image_size[1])

    return {
        "camera_label": camera_label,
        "model_name": model_name,
        "width": width,
        "height": height,
        "T_device_camera": T_device_camera,
        "projection_params": projection_params,
    }


def project_fisheye624(point_cam, params):
    x, y, z = point_cam

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
        return [cu, cv], z, True

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


def find_frame_entry_by_frame_index(framewise_rows, target_frame):
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

    return best, best_dist


def get_object_state(mask_info_video, obj, time_sec):
    chosen_track, mode, next_time = choose_track_for_time(obj["tracks"], time_sec)

    if chosen_track is None:
        return {
            "status": "no_track_available",
            "selection_mode": None,
            "track_id": None,
            "time_segment": None,
            "mask": None,
            "next_closest_possible_time": None,
            "comment": f"No track available for this object around time {time_sec}.",
        }

    if mode == "in_motion":
        return {
            "status": "in_motion",
            "selection_mode": mode,
            "track_id": chosen_track["track_id"],
            "time_segment": chosen_track["time_segment"],
            "mask": None,
            "next_closest_possible_time": None,
            "comment": "Time falls inside a movement track, so no stable coordinates are assigned.",
        }

    pick = "latest" if mode == "past" else "first"
    chosen_mask = get_mask_from_track(mask_info_video, chosen_track, pick=pick)

    if chosen_mask is None:
        return {
            "status": "no_valid_mask",
            "selection_mode": mode,
            "track_id": chosen_track["track_id"],
            "time_segment": chosen_track["time_segment"],
            "mask": None,
            "next_closest_possible_time": next_time,
            "comment": "Chosen track exists, but no mask from that track was found in mask_info.",
        }

    return {
        "status": "ok",
        "selection_mode": mode,
        "track_id": chosen_track["track_id"],
        "time_segment": chosen_track["time_segment"],
        "mask": chosen_mask,
        "next_closest_possible_time": next_time,
        "comment": (
            "Used latest mask from the latest track strictly before the chosen time."
            if mode == "past"
            else "No earlier track existed, so used first mask from the earliest future track."
        ),
    }


def write_error(output_yaml, msg):
    print(msg)
    output_yaml.write_text(f'error: {yaml_quote_string(msg)}\n', encoding="utf-8")


def read_video_frame(video_path, frame_index=None, time_sec=None, fallback_fps=30.0):
    if cv2 is None:
        raise RuntimeError("OpenCV is not available. Please install opencv-python to enable visualization.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps is None or actual_fps <= 1e-6:
        actual_fps = fallback_fps

    if frame_index is None:
        if time_sec is None:
            cap.release()
            raise ValueError("Either frame_index or time_sec must be provided to read a frame.")
        frame_index = int(round(time_sec * actual_fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()

    if not ok and time_sec is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
        ok, frame = cap.read()

    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from video: {video_path}")

    return frame, frame_index, actual_fps


def draw_text_block(image, lines, x=20, y=30, line_gap=26):
    if cv2 is None:
        return image
    for i, text in enumerate(lines):
        yy = y + i * line_gap
        cv2.putText(image, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def draw_arrow_if_valid(image, p0, p1, color, label, thickness):
    if cv2 is None or p0 is None or p1 is None:
        return
    p0i = (int(round(p0[0])), int(round(p0[1])))
    p1i = (int(round(p1[0])), int(round(p1[1])))
    cv2.arrowedLine(image, p0i, p1i, color, thickness, cv2.LINE_AA, tipLength=0.15)
    cv2.putText(image, label, (p1i[0] + 6, p1i[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, label, (p1i[0] + 6, p1i[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def create_anchor_visualization(
    video_path,
    vis_output,
    frame_index,
    time_sec,
    fallback_fps,
    anchor_name,
    origin_pixel,
    axis_pixels,
    object_pixels,
    axis_interpretation_line,
    axis_thickness,
):
    frame, used_frame_index, actual_fps = read_video_frame(
        video_path=video_path,
        frame_index=frame_index,
        time_sec=time_sec,
        fallback_fps=fallback_fps,
    )

    overlay = frame.copy()
    if origin_pixel is not None:
        origin_xy = (int(round(origin_pixel[0])), int(round(origin_pixel[1])))
        cv2.circle(overlay, origin_xy, 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, origin_xy, 12, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Origin: {anchor_name}", (origin_xy[0] + 12, origin_xy[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, f"Origin: {anchor_name}", (origin_xy[0] + 12, origin_xy[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    axis_colors = {
        "X": (0, 0, 255),
        "Y": (0, 255, 0),
        "Z": (255, 0, 0),
    }
    for axis_name, axis_pixel in axis_pixels.items():
        draw_arrow_if_valid(overlay, origin_pixel, axis_pixel, axis_colors[axis_name], axis_name, axis_thickness)

    for item in object_pixels:
        px = item.get("pixel")
        if px is None:
            continue
        pt = (int(round(px[0])), int(round(px[1])))
        cv2.circle(overlay, pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, pt, 8, (0, 0, 0), 2, cv2.LINE_AA)
        label = item["name"]
        cv2.putText(overlay, label, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, label, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    text_lines = [
        f"Frame: {used_frame_index}",
        f"Time: {time_sec:.3f}s",
        "Axes are camera-aligned at the anchor object",
        axis_interpretation_line,
    ]
    draw_text_block(overlay, text_lines)

    vis_output = Path(vis_output)
    vis_output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(vis_output), overlay)
    if not ok:
        raise RuntimeError(f"Failed to save visualization to: {vis_output}")

    return used_frame_index, actual_fps


def main():
    args = parse_args()

    video_id = args.video
    object_b_name = args.obj_b
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
        write_error(output_yaml, f"Video {video_id} not found in assoc_info.json")
        return

    if video_id not in mask_info:
        write_error(output_yaml, f"Video {video_id} not found in mask_info.json")
        return

    video_objects = assoc_info[video_id]
    mask_info_video = mask_info[video_id]

    _, obj_b = find_object_entry(video_objects, object_b_name)

    if obj_b is None:
        write_error(output_yaml, f"Object B '{object_b_name}' not found in video {video_id}.")
        return

    target_frame = int(round(time_sec * args.fps))
    frame_entry, frame_dist = find_frame_entry_by_frame_index(framewise_rows, target_frame)

    if frame_entry is None:
        write_error(output_yaml, f"No valid frame entry with T_world_device found near queried frame {target_frame}.")
        return

    queried_frame_index = int(frame_entry["frame_index"])

    state_b = get_object_state(mask_info_video, obj_b, time_sec)
    if state_b["status"] == "in_motion":
        write_error(
            output_yaml,
            (
                f"Object B '{object_b_name}' is in motion at queried time {time_sec}. "
                "Cannot define relative positions from a moving anchor."
            ),
        )
        return
    if state_b["status"] != "ok":
        write_error(
            output_yaml,
            (
                f"Object B '{object_b_name}' is not visible/stable at queried time {time_sec}: "
                f"{state_b['status']}."
            ),
        )
        return

    mask_b = state_b["mask"]
    xyz_b_world = mask_b["3d_location"]

    camera_label = args.camera_label
    camera_calib = get_camera_calibration(calibration, camera_label)
    width = camera_calib["width"]
    height = camera_calib["height"]
    model_name = camera_calib["model_name"]
    projection_params = camera_calib["projection_params"]
    T_device_camera = camera_calib["T_device_camera"]

    b_reference_frame = int(mask_b["frame_number"])

    T_world_device_raw = frame_entry["T_world_device"]
    if len(T_world_device_raw) == 3 and len(T_world_device_raw[0]) == 4:
        T_world_device = to_homogeneous_4x4(T_world_device_raw)
    elif len(T_world_device_raw) == 4 and len(T_world_device_raw[0]) == 4:
        T_world_device = T_world_device_raw
    else:
        write_error(output_yaml, "T_world_device has unexpected shape.")
        return

    T_world_camera = mat4_mul(T_world_device, T_device_camera)
    T_camera_world = invert_rigid_4x4(T_world_camera)

    b_cam = transform_point(T_camera_world, xyz_b_world)

    if model_name == "CameraModelType.FISHEYE624":
        b_pixel, _, b_valid = project_fisheye624(b_cam, projection_params)
    else:
        write_error(output_yaml, f"Unsupported camera model for projection: {model_name}")
        return

    b_in_view = b_valid and point_in_image(b_pixel, width, height)

    if not b_in_view:
        write_error(
            output_yaml,
            (
                f"Object B '{object_b_name}' is not visible in {camera_label} at queried time {time_sec}. "
                f"It projects outside the image for queried frame {frame_entry['frame_index']}."
            ),
        )
        return

    axis_length = float(args.axis_length_m)

    R_world_camera = [
        [T_world_camera[0][0], T_world_camera[0][1], T_world_camera[0][2]],
        [T_world_camera[1][0], T_world_camera[1][1], T_world_camera[1][2]],
        [T_world_camera[2][0], T_world_camera[2][1], T_world_camera[2][2]],
    ]
    x_axis_world = [R_world_camera[0][0], R_world_camera[1][0], R_world_camera[2][0]]
    y_axis_world = [R_world_camera[0][1], R_world_camera[1][1], R_world_camera[2][1]]
    z_axis_world = [R_world_camera[0][2], R_world_camera[1][2], R_world_camera[2][2]]

    axis_endpoints_world = {
        "X": [xyz_b_world[i] + axis_length * x_axis_world[i] for i in range(3)],
        "Y": [xyz_b_world[i] + axis_length * y_axis_world[i] for i in range(3)],
        "Z": [xyz_b_world[i] + axis_length * z_axis_world[i] for i in range(3)],
    }
    axis_points_cam = {name: transform_point(T_camera_world, point_world) for name, point_world in axis_endpoints_world.items()}
    axis_pixels = {}
    for axis_name, point_cam in axis_points_cam.items():
        px, _, valid = project_fisheye624(point_cam, projection_params)
        axis_pixels[axis_name] = px if valid and point_in_image(px, width, height) else px

    object_pixels_for_vis = []

    lines = []
    lines.append("# Relative position of every object A with respect to anchor object B")
    lines.append("# Egocentric frame: origin at B, axes aligned with the chosen camera viewpoint from a frame where B is visible")
    lines.append(f"video_id: {yaml_quote_string(video_id)}")
    lines.append(f"time_sec: {time_sec:.6f}")
    lines.append(f"participant_id: {yaml_quote_string(participant_id)}")
    lines.append("")
    lines.append("query:")
    lines.append(f"  anchor_object_b: {yaml_quote_string(object_b_name)}")
    lines.append("")
    lines.append("camera_anchor_frame:")
    lines.append(f"  camera_label: {yaml_quote_string(camera_label)}")
    lines.append(f"  model_name: {yaml_quote_string(model_name)}")
    lines.append(f"  image_size: {yaml_list_inline([width, height])}")
    lines.append(f"  queried_frame_index_from_time: {target_frame}")
    lines.append(f"  chosen_frame_index: {frame_entry['frame_index']}")
    lines.append(f"  queried_frame_index_used: {queried_frame_index}")
    lines.append(f"  reference_b_mask_frame_number: {b_reference_frame}")
    lines.append(f"  frame_distance_to_query: {frame_dist}")
    lines.append('  axis_definition: "Origin is at B; axes are aligned with the chosen camera axes of the queried frame."')
    lines.append('  axis_interpretation: "+X, +Y, +Z are the queried camera-frame axes after moving the origin to object B."')
    lines.append(f"  x_axis_world: {yaml_list_inline(x_axis_world)}")
    lines.append(f"  y_axis_world: {yaml_list_inline(y_axis_world)}")
    lines.append(f"  z_axis_world: {yaml_list_inline(z_axis_world)}")
    R_B = [
        [T_camera_world[0][0], T_camera_world[0][1], T_camera_world[0][2]],
        [T_camera_world[1][0], T_camera_world[1][1], T_camera_world[1][2]],
        [T_camera_world[2][0], T_camera_world[2][1], T_camera_world[2][2]],
    ]
    T_B = [
        [R_B[0][0], R_B[0][1], R_B[0][2], -(R_B[0][0] * xyz_b_world[0] + R_B[0][1] * xyz_b_world[1] + R_B[0][2] * xyz_b_world[2])],
        [R_B[1][0], R_B[1][1], R_B[1][2], -(R_B[1][0] * xyz_b_world[0] + R_B[1][1] * xyz_b_world[1] + R_B[1][2] * xyz_b_world[2])],
        [R_B[2][0], R_B[2][1], R_B[2][2], -(R_B[2][0] * xyz_b_world[0] + R_B[2][1] * xyz_b_world[1] + R_B[2][2] * xyz_b_world[2])],
    ]
    lines.append("  T_B_world_to_local:")
    lines.append(f"    - {yaml_list_inline(T_B[0])}")
    lines.append(f"    - {yaml_list_inline(T_B[1])}")
    lines.append(f"    - {yaml_list_inline(T_B[2])}")
    lines.append(f"  anchor_origin_pixel: {yaml_list_inline(b_pixel)}")
    lines.append(f"  anchor_axis_length_m: {axis_length:.6f}")
    lines.append(f"  anchor_axis_pixels:")
    lines.append(f"    X: {yaml_list_inline(axis_pixels['X'])}")
    lines.append(f"    Y: {yaml_list_inline(axis_pixels['Y'])}")
    lines.append(f"    Z: {yaml_list_inline(axis_pixels['Z'])}")
    lines.append("")
    lines.append("object_b_info:")
    lines.append(f"  selection_mode: {yaml_quote_string(state_b['selection_mode'])}")
    lines.append(f"  track_id: {yaml_quote_string(state_b['track_id'])}")
    lines.append(f"  time_segment: {yaml_list_inline(state_b['time_segment'])}")
    lines.append(f"  mask_id: {yaml_quote_string(mask_b['mask_id'])}")
    lines.append(f"  frame_number: {mask_b['frame_number']}")
    lines.append(f"  fixture: {yaml_quote_string(mask_b['fixture'])}")
    lines.append(f"  world_coordinates: {yaml_list_inline(xyz_b_world)}")
    lines.append(f"  camera_coordinates_in_anchor_view: {yaml_list_inline(b_cam)}")
    lines.append(f"  projected_pixel_in_anchor_view: {yaml_list_inline(b_pixel)}")
    lines.append("  in_view_in_anchor_frame: true")
    lines.append("")
    lines.append("objects_a:")

    for _, obj_a in video_objects.items():
        obj_a_name = obj_a["name"]
        if obj_a_name.strip().lower() == object_b_name.strip().lower():
            continue

        state_a = get_object_state(mask_info_video, obj_a, time_sec)

        lines.append(f"  - name: {yaml_quote_string(obj_a_name)}")
        lines.append(f"    status: {yaml_quote_string(state_a['status'])}")
        lines.append(f"    selection_mode: {yaml_quote_string(state_a['selection_mode'])}")
        lines.append(f"    track_id: {yaml_quote_string(state_a['track_id'])}")
        lines.append(f"    time_segment: {yaml_list_inline(state_a['time_segment'])}")
        lines.append(
            f"    next_closest_possible_time: {state_a['next_closest_possible_time']:.6f}"
            if state_a["next_closest_possible_time"] is not None else
            "    next_closest_possible_time: null"
        )

        if state_a["status"] != "ok":
            lines.append("    mask_id: null")
            lines.append("    frame_number: null")
            lines.append("    fixture: null")
            lines.append("    world_coordinates: null")
            lines.append("    camera_coordinates_in_anchor_view: null")
            lines.append("    projected_pixel_in_anchor_view: null")
            lines.append("    depth_in_anchor_camera: null")
            lines.append("    visible_in_anchor_frame: null")
            lines.append("    a_minus_b_world: null")
            lines.append("    a_relative_to_b_egocentric: null")
            lines.append(f"    comment: {yaml_quote_string(state_a['comment'])}")
            continue

        mask_a = state_a["mask"]
        xyz_a_world = mask_a["3d_location"]
        a_cam = transform_point(T_camera_world, xyz_a_world)
        a_pixel, a_depth, a_valid = project_fisheye624(a_cam, projection_params)
        a_in_view = a_valid and point_in_image(a_pixel, width, height)

        a_minus_b_world = [
            xyz_a_world[0] - xyz_b_world[0],
            xyz_a_world[1] - xyz_b_world[1],
            xyz_a_world[2] - xyz_b_world[2],
        ]
        a_relative_to_b_egocentric = [
            a_cam[0] - b_cam[0],
            a_cam[1] - b_cam[1],
            a_cam[2] - b_cam[2],
        ]

        lines.append(f"    mask_id: {yaml_quote_string(mask_a['mask_id'])}")
        lines.append(f"    frame_number: {mask_a['frame_number']}")
        lines.append(f"    fixture: {yaml_quote_string(mask_a['fixture'])}")
        lines.append(f"    world_coordinates: {yaml_list_inline(xyz_a_world)}")
        lines.append(f"    camera_coordinates_in_anchor_view: {yaml_list_inline(a_cam)}")
        lines.append(f"    projected_pixel_in_anchor_view: {yaml_list_inline(a_pixel)}")
        lines.append(f"    depth_in_anchor_camera: {a_depth:.6f}" if a_depth is not None else "    depth_in_anchor_camera: null")
        lines.append(f"    visible_in_anchor_frame: {'true' if a_in_view else 'false'}")
        lines.append(f"    a_minus_b_world: {yaml_list_inline(a_minus_b_world)}")
        lines.append(f"    a_relative_to_b_egocentric: {yaml_list_inline(a_relative_to_b_egocentric)}")
        lines.append(
            f"    comment: {yaml_quote_string(f'Projected inside {camera_label} image bounds in the anchor frame.' if a_in_view else f'Projected outside {camera_label} image bounds or behind camera in the anchor frame.') }"
        )

        if a_in_view:
            object_pixels_for_vis.append({"name": obj_a_name, "pixel": a_pixel})

    if args.video_path is not None:
        try:
            if camera_label != "camera-rgb":
                raise RuntimeError(
                    f"Visualization with --video_path currently assumes an RGB video, but camera_label={camera_label!r}. "
                    "Use --camera_label camera-rgb for overlay visualization or extend the script to read that camera stream."
                )
            used_frame_index, actual_fps = create_anchor_visualization(
                video_path=args.video_path,
                vis_output=args.vis_output,
                frame_index=int(frame_entry["frame_index"]),
                time_sec=time_sec,
                fallback_fps=args.fps,
                anchor_name=object_b_name,
                origin_pixel=b_pixel,
                axis_pixels=axis_pixels,
                object_pixels=object_pixels_for_vis,
                axis_interpretation_line="Axes follow the chosen camera frame",
                axis_thickness=args.axis_thickness,
            )
            lines.append("")
            lines.append("visualization:")
            lines.append(f"  video_path: {yaml_quote_string(args.video_path)}")
            lines.append(f"  output_image: {yaml_quote_string(args.vis_output)}")
            lines.append(f"  used_frame_index: {used_frame_index}")
            lines.append(f"  video_fps_used: {actual_fps:.6f}")
            lines.append('  note: "Saved image overlay showing the anchored local coordinate system and all objects visible in the anchor frame."')
        except Exception as e:
            lines.append("")
            lines.append("visualization:")
            lines.append(f"  video_path: {yaml_quote_string(args.video_path)}")
            lines.append(f"  output_image: {yaml_quote_string(args.vis_output)}")
            lines.append(f"  error: {yaml_quote_string(str(e))}")

    output_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
