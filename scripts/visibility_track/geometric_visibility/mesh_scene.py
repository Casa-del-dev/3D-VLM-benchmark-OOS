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

    def occlusion_fraction(
        self,
        camera_world: np.ndarray | list[float],
        target_points_world: np.ndarray | list[list[float]],
        tolerance: float = 0.05,
    ) -> float:
        """Fraction of rays from *camera_world* to *target_points_world*
        that are blocked by scene geometry.

        For each target point, casts a ray from the camera to that point and
        compares the nearest hit distance against the camera→target distance
        (minus *tolerance* to avoid self-intersection at the object surface).
        Returns blocked-count / N. Degenerate targets (distance < 1e-6) are
        skipped from both numerator and denominator; if every target is
        degenerate, returns 0.0.
        """
        origin = np.asarray(camera_world, dtype=np.float64).reshape(3)
        targets = np.asarray(target_points_world, dtype=np.float64).reshape(-1, 3)
        if targets.shape[0] == 0:
            return 0.0

        directions = targets - origin
        distances = np.linalg.norm(directions, axis=1)
        valid = distances >= 1e-6
        if not np.any(valid):
            return 0.0

        valid_dirs = directions[valid] / distances[valid, None]
        valid_dists = distances[valid]
        n_valid = int(valid_dirs.shape[0])
        origins = np.broadcast_to(origin, valid_dirs.shape).copy()

        locations, ray_indices, _ = self._intersector.intersects_location(
            ray_origins=origins,
            ray_directions=valid_dirs,
            multiple_hits=False,
        )

        if len(locations) == 0:
            return 0.0

        hit_dists = np.linalg.norm(locations - origins[ray_indices], axis=1)
        blocked = hit_dists < (valid_dists[ray_indices] - tolerance)
        return float(blocked.sum()) / float(n_valid)

    def is_occluded(
        self,
        camera_world: np.ndarray | list[float],
        object_world: np.ndarray | list[float],
        tolerance: float = 0.05,
    ) -> bool:
        """Single-ray occlusion check. Thin wrapper around occlusion_fraction.

        Returns True iff the line of sight from camera to *object_world* is
        blocked by scene geometry, with *tolerance* metres subtracted from
        the hit distance to avoid self-intersection at the object surface.
        """
        return self.occlusion_fraction(camera_world, [object_world], tolerance) >= 0.5
