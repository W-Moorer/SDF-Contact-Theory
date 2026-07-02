from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class TriangleMesh:
    """Simple triangle mesh container.

    vertices: (N, 3) float64
    faces: (M, 3) int64
    """

    vertices: np.ndarray
    faces: np.ndarray
    name: str = "mesh"

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be shaped (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be shaped (M, 3)")

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    def copy(self, name: str | None = None) -> "TriangleMesh":
        return TriangleMesh(self.vertices.copy(), self.faces.copy(), self.name if name is None else name)

    def transformed(self, R: np.ndarray | None = None, t: np.ndarray | None = None, name: str | None = None) -> "TriangleMesh":
        verts = self.vertices.copy()
        if R is not None:
            R = np.asarray(R, dtype=np.float64).reshape(3, 3)
            verts = verts @ R.T
        if t is not None:
            verts = verts + np.asarray(t, dtype=np.float64).reshape(1, 3)
        return TriangleMesh(verts, self.faces.copy(), self.name if name is None else name)

    def translated(self, t: np.ndarray, name: str | None = None) -> "TriangleMesh":
        return self.transformed(t=t, name=name)

    def scaled(self, s: float | np.ndarray, name: str | None = None) -> "TriangleMesh":
        scale = np.asarray(s, dtype=np.float64)
        verts = self.vertices * scale
        return TriangleMesh(verts, self.faces.copy(), self.name if name is None else name)

    def face_vertices(self) -> np.ndarray:
        return self.vertices[self.faces]

    def face_normals(self) -> np.ndarray:
        tri = self.face_vertices()
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        l = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(l, 1e-30)

    def face_areas(self) -> np.ndarray:
        tri = self.face_vertices()
        return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    def ensure_outward_orientation(self) -> "TriangleMesh":
        """Best-effort outward orientation using trimesh when available.

        This is primarily used for visualization/pseudo-normal sign. The parity
        ray-cast sign and analytic sign do not depend on winding.
        """
        try:
            import trimesh
            tm = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
            trimesh.repair.fix_normals(tm, multibody=True)
            return TriangleMesh(np.asarray(tm.vertices), np.asarray(tm.faces), self.name)
        except Exception:
            return self

    def to_trimesh(self):
        import trimesh
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)

    def export_obj(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# {self.name}\n")
            for v in self.vertices:
                f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
            for tri in self.faces:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def merge_duplicate_vertices(vertices: np.ndarray, faces: np.ndarray, tol: float = 1e-10) -> TriangleMesh:
    """Merge exact/near duplicate vertices after independently generated patches."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0:
        return TriangleMesh(vertices, faces)
    key = np.round(vertices / tol).astype(np.int64)
    mapping: Dict[Tuple[int, int, int], int] = {}
    new_vertices = []
    remap = np.empty(len(vertices), dtype=np.int64)
    for i, k in enumerate(map(tuple, key)):
        idx = mapping.get(k)
        if idx is None:
            idx = len(new_vertices)
            mapping[k] = idx
            new_vertices.append(vertices[i])
        remap[i] = idx
    new_faces = remap[faces]
    # Remove degenerate triangles created by merging.
    keep = (new_faces[:, 0] != new_faces[:, 1]) & (new_faces[:, 1] != new_faces[:, 2]) & (new_faces[:, 2] != new_faces[:, 0])
    return TriangleMesh(np.asarray(new_vertices, dtype=np.float64), new_faces[keep])


def rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_matrix_xyz(rx: float = 0.0, ry: float = 0.0, rz: float = 0.0) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx
