import bpy
import json
from mathutils import Vector

OPENABLE_TYPES = {"cupboard", "drawer", "fridge", "microwave", "oven"}

# Optional: keep only object types you care about
VALID_TYPES = {
    "bin",
    "counter",
    "cupboard",
    "dishwasher",
    "drawer",
    "floor",
    "fridge",
    "hob",
    "hook",
    "microwave",
    "oven",
    "shelf",
    "sink",
    "storage",
    "top_microwave",
    "top_storage",
    "windowsill",
}

def get_world_bbox(obj):
    local_bbox = [Vector(corner) for corner in obj.bound_box]
    world_bbox = [obj.matrix_world @ v for v in local_bbox]

    min_corner = [
        min(v.x for v in world_bbox),
        min(v.y for v in world_bbox),
        min(v.z for v in world_bbox),
    ]
    max_corner = [
        max(v.x for v in world_bbox),
        max(v.y for v in world_bbox),
        max(v.z for v in world_bbox),
    ]
    return min_corner, max_corner

def parse_fixture_name(name):
    # Blender object names in your scene look like:
    # cupboard.001, drawer.003, dishwasher.001
    # and sometimes could be names without a numeric suffix.
    fixture_type = name.split(".")[0]

    if fixture_type == "":
        return None

    if fixture_type not in VALID_TYPES:
        return None

    return fixture_type

metadata = {}

for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue

    fixture_type = parse_fixture_name(obj.name)
    if fixture_type is None:
        print(f"Skipping: {obj.name}")
        continue

    bbox_min, bbox_max = get_world_bbox(obj)
    openable = fixture_type in OPENABLE_TYPES

    metadata[obj.name] = {
        "fixture_type": fixture_type,
        "is_openable": openable,
        "interior_bbox_min": bbox_min if openable else None,
        "interior_bbox_max": bbox_max if openable else None,
    }

output_path = "/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/data/fixture_metadata/kitchen04_metadata.json"

with open(output_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved metadata to: {output_path}")
print(f"Total fixtures saved: {len(metadata)}")