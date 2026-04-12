"""Load digital-twin OBJ meshes and perform ray-mesh occlusion tests.

The digital-twin meshes live at::

    <data_root>/HD-EPIC/Digital-Twin/blenders/meshes/<participant_id>/

Each ``.obj`` file represents a fixed fixture (counter, cupboard, shelf, …).
We combine them into a single :class:`trimesh.Trimesh` so that a single ray
intersection call can test occlusion against the entire scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def load_participant_mesh(data_root: Path, participant_id: str) -> trimesh.Trimesh:
    """Load and merge all OBJ meshes for *participant_id* into one Trimesh.

    Returns a single watertight-ish mesh whose vertices are in the same world
    coordinate frame used by HD-EPIC annotations (the Blender exports share
    that frame).
    """
    mesh_dir = data_root / "HD-EPIC" / "Digital-Twin" / "blenders" / "meshes" / participant_id
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

    obj_files = sorted(mesh_dir.glob("*.obj"))
    if not obj_files:
        raise FileNotFoundError(f"No .obj files in {mesh_dir}")

    meshes: list[trimesh.Trimesh] = []
    for obj_path in obj_files:
        loaded = trimesh.load(obj_path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Trimesh):
            meshes.append(loaded)
        elif isinstance(loaded, trimesh.Scene):
            for geom in loaded.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)

    if not meshes:
        raise RuntimeError(f"Could not load any mesh geometry from {mesh_dir}")

    combined = trimesh.util.concatenate(meshes)
    return combined


class RayOcclusionChecker:
    """Fast ray-mesh intersection tester backed by trimesh's ray module.

    Typical usage::

        checker = RayOcclusionChecker.from_data_root(data_root, "P01")
        occluded = checker.is_occluded(camera_xyz, object_xyz)
    """

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self._mesh = mesh
        # Prefer pyembree (much faster) but fall back to the pure-Python
        # intersector that ships with trimesh.
        try:
            self._intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        except (ImportError, AttributeError):
            self._intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    @classmethod
    def from_data_root(cls, data_root: Path, participant_id: str) -> "RayOcclusionChecker":
        mesh = load_participant_mesh(data_root, participant_id)
        return cls(mesh)

    def is_occluded(
        self,
        camera_world: np.ndarray | list[float],
        object_world: np.ndarray | list[float],
        tolerance: float = 0.05,
    ) -> bool:
        """Return True when the line of sight is blocked by scene geometry.

        Casts a ray from *camera_world* towards *object_world* and checks
        whether the closest hit is nearer than the object (minus *tolerance*
        metres to avoid self-intersection at the object surface).

        Parameters
        ----------
        camera_world : (3,) array
            Camera position in world coordinates.
        object_world : (3,) array
            Object position in world coordinates.
        tolerance : float
            Distance margin (metres) subtracted from the camera→object
            distance so that hits on the object's own surface are not counted
            as occlusion.
        """
        origin = np.asarray(camera_world, dtype=np.float64).reshape(1, 3)
        target = np.asarray(object_world, dtype=np.float64).reshape(1, 3)
        direction = target - origin
        dist_to_object = float(np.linalg.norm(direction))
        if dist_to_object < 1e-6:
            return False
        direction = direction / dist_to_object

        locations, ray_indices, _ = self._intersector.intersects_location(
            ray_origins=origin,
            ray_directions=direction,
            multiple_hits=False,
        )

        if len(locations) == 0:
            return False

        hit_dist = float(np.linalg.norm(locations[0] - origin[0]))
        return hit_dist < (dist_to_object - tolerance)
