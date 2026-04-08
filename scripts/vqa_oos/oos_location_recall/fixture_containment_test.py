import os
import json
import cv2


def normalize_fixture_name(name):
    if name is None:
        return None
    name = str(name).strip()
    return name if name else None


def load_fixture_meta(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def point_inside_aabb(point_xyz, bbox_min, bbox_max, margin=0.0):
    if point_xyz is None or bbox_min is None or bbox_max is None:
        return False

    x, y, z = point_xyz
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max

    return (
        (xmin - margin) <= x <= (xmax + margin) and
        (ymin - margin) <= y <= (ymax + margin) and
        (zmin - margin) <= z <= (zmax + margin)
    )


def point_inside_aabb_strict_and_loose(point_xyz, bbox_min, bbox_max, loose_margin=0.03):
    inside_strict = point_inside_aabb(point_xyz, bbox_min, bbox_max, margin=0.0)
    inside_loose = point_inside_aabb(point_xyz, bbox_min, bbox_max, margin=loose_margin)
    return inside_strict, inside_loose


def compute_normalized_distance_to_box_boundary(point_xyz, bbox_min, bbox_max):
    if point_xyz is None or bbox_min is None or bbox_max is None:
        return None

    x, y, z = point_xyz
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max

    sx = max(xmax - xmin, 1e-9)
    sy = max(ymax - ymin, 1e-9)
    sz = max(zmax - zmin, 1e-9)

    dx = min(x - xmin, xmax - x) / sx
    dy = min(y - ymin, ymax - y) / sy
    dz = min(z - zmin, zmax - z) / sz

    if point_inside_aabb(point_xyz, bbox_min, bbox_max, margin=0.0):
        return min(dx, dy, dz)

    ox = 0.0
    oy = 0.0
    oz = 0.0

    if x < xmin:
        ox = (xmin - x) / sx
    elif x > xmax:
        ox = (x - xmax) / sx

    if y < ymin:
        oy = (ymin - y) / sy
    elif y > ymax:
        oy = (y - ymax) / sy

    if z < zmin:
        oz = (zmin - z) / sz
    elif z > zmax:
        oz = (z - zmax) / sz

    return -max(ox, oy, oz)


def containment_state_from_flags(is_openable, inside_strict, inside_loose):
    if not is_openable:
        return "not_applicable"
    if inside_strict:
        return "inside_strict"
    if inside_loose:
        return "inside_loose"
    return "outside"


def check_containment_with_assigned_fixture(point_xyz, assigned_fixture, fixture_meta, loose_margin=0.03):
    assigned_fixture = normalize_fixture_name(assigned_fixture)

    result = {
        "candidate_fixture": assigned_fixture,
        "fixture_found_in_meta": False,
        "fixture_type": None,
        "fixture_is_openable": False,
        "bbox_min": None,
        "bbox_max": None,
        "inside_strict": False,
        "inside_loose": False,
        "boundary_score": None,
        "containment_state": "unknown",
        "containment_comment": "",
    }

    if assigned_fixture is None:
        result["containment_state"] = "unknown"
        result["containment_comment"] = "No assigned fixture."
        return result

    meta = fixture_meta.get(assigned_fixture)
    if meta is None:
        result["containment_state"] = "unknown"
        result["containment_comment"] = "Assigned fixture not found in fixture metadata."
        return result

    result["fixture_found_in_meta"] = True
    result["fixture_type"] = meta.get("fixture_type")
    result["fixture_is_openable"] = bool(meta.get("is_openable", False))

    bbox_min = meta.get("bbox_min")
    bbox_max = meta.get("bbox_max")

    result["bbox_min"] = bbox_min
    result["bbox_max"] = bbox_max

    if bbox_min is None or bbox_max is None:
        result["containment_state"] = "unknown"
        result["containment_comment"] = "Fixture has no bbox."
        return result

    inside_strict, inside_loose = point_inside_aabb_strict_and_loose(
        point_xyz=point_xyz,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        loose_margin=loose_margin,
    )

    result["inside_strict"] = inside_strict
    result["inside_loose"] = inside_loose
    result["boundary_score"] = compute_normalized_distance_to_box_boundary(
        point_xyz=point_xyz,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )
    result["containment_state"] = containment_state_from_flags(
        is_openable=result["fixture_is_openable"],
        inside_strict=inside_strict,
        inside_loose=inside_loose,
    )

    if not result["fixture_is_openable"]:
        result["containment_comment"] = "Fixture is not openable."
    elif inside_strict:
        result["containment_comment"] = "Point is strictly inside."
    elif inside_loose:
        result["containment_comment"] = "Point is inside with tolerance."
    else:
        result["containment_comment"] = "Point is outside."

    return result


def timestamp_to_frame_idx(video_path, timestamp_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None or fps <= 0:
        raise ValueError("Could not read FPS from video.")

    return int(round(timestamp_sec * fps))


def extract_frame_at_timestamp(video_path, timestamp_sec):
    frame_idx = timestamp_to_frame_idx(video_path, timestamp_sec)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise ValueError(f"Could not read frame at timestamp {timestamp_sec}s (frame {frame_idx}).")

    return frame_idx, frame


def draw_overlay(frame, result, timestamp_sec, object_id=None, point_uv=None):
    vis = frame.copy()

    lines = [
        f"timestamp_sec: {timestamp_sec}",
        f"frame_idx: {result.get('frame_idx')}",
        f"object_id: {object_id if object_id is not None else 'unknown'}",
        f"fixture: {result.get('candidate_fixture')}",
        f"fixture_type: {result.get('fixture_type')}",
        f"state: {result.get('containment_state')}",
        f"boundary_score: {result.get('boundary_score')}",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            vis,
            str(line),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30

    if point_uv is not None:
        u, v = int(point_uv[0]), int(point_uv[1])
        cv2.circle(vis, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(
            vis,
            "object point",
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return vis


def test_containment_at_timestamp(
    fixture_meta_path,
    video_path,
    timestamp_sec,
    point_xyz,
    assigned_fixture,
    output_image_path,
    object_id=None,
    point_uv=None,
    loose_margin=0.03,
    output_json_path=None,
):
    fixture_meta = load_fixture_meta(fixture_meta_path)

    result = check_containment_with_assigned_fixture(
        point_xyz=point_xyz,
        assigned_fixture=assigned_fixture,
        fixture_meta=fixture_meta,
        loose_margin=loose_margin,
    )

    frame_idx, frame = extract_frame_at_timestamp(video_path, timestamp_sec)
    result["timestamp_sec"] = timestamp_sec
    result["frame_idx"] = frame_idx
    result["point_xyz"] = point_xyz
    result["object_id"] = object_id

    vis = draw_overlay(
        frame=frame,
        result=result,
        timestamp_sec=timestamp_sec,
        object_id=object_id,
        point_uv=point_uv,
    )

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, vis)

    if output_json_path is not None:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    result = test_containment_at_timestamp(
        fixture_meta_path="/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/data/fixture_metadata/kitchen04_metadata.json",
        video_path="/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/data/HD-EPIC/Videos/P04/P04-20240413-142619.mp4",
        timestamp_sec=229.6,
        point_xyz=[-0.4425964290411465, -0.5415208860846654, 0.28208011928335747],
        assigned_fixture="cupboard.006",
        output_image_path="/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/outputs/containment_test/containment_t12_4.jpg",
        output_json_path="/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/outputs/containment_test/containment_t12_4.json",
        object_id="stack of three bowls",  # optional 2D point on image
        loose_margin=0.03,
    )

    print(json.dumps(result, indent=2))