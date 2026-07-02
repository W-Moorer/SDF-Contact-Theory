from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from sdf_contact.geometry.mesh import TriangleMesh
from sdf_contact.sdf.grid import SDFGrid
from sdf_contact.sdf.interpolation import normalize_grad


@dataclass
class ContactTriangles:
    triangles: np.ndarray  # (K, 3, 3)
    phi: np.ndarray        # (K,) centroid phi
    depth: np.ndarray      # (K,) penetration depth
    normals: np.ndarray    # (K, 3)
    source_face: np.ndarray  # (K,)
    method: str = "linear"
    labels: np.ndarray | None = None

    @property
    def count(self) -> int:
        return int(self.triangles.shape[0])

    def centroids(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return self.triangles.mean(axis=1)

    def areas(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros(0, dtype=np.float64)
        return 0.5 * np.linalg.norm(np.cross(self.triangles[:, 1] - self.triangles[:, 0], self.triangles[:, 2] - self.triangles[:, 0]), axis=1)


def _clip_triangle_linear(vertices: np.ndarray, phis: np.ndarray, eps: float = 0.0) -> list[np.ndarray]:
    """Clip a triangle by phi<=eps, return 0..2 inside triangles."""
    poly = [(vertices[i], phis[i]) for i in range(3)]

    def inside(item):
        return item[1] <= eps

    out = []
    for i in range(len(poly)):
        curr = poly[i]
        prev = poly[i - 1]
        curr_in = inside(curr)
        prev_in = inside(prev)
        if curr_in != prev_in:
            p0, f0 = prev
            p1, f1 = curr
            denom = f1 - f0
            t = 0.5 if abs(denom) < 1e-14 else (eps - f0) / denom
            t = float(np.clip(t, 0.0, 1.0))
            p = (1.0 - t) * p0 + t * p1
            out.append((p, eps))
        if curr_in:
            out.append(curr)
    if len(out) < 3:
        return []
    pts = np.asarray([p for p, _ in out], dtype=np.float64)
    if len(pts) == 3:
        return [pts]
    # Fan triangulation for convex clipped polygon.
    tris = []
    for i in range(1, len(pts) - 1):
        tris.append(np.stack([pts[0], pts[i], pts[i + 1]], axis=0))
    return tris


def _subdivide_triangle(tri: np.ndarray) -> list[np.ndarray]:
    a, b, c = tri
    ab = 0.5 * (a + b)
    bc = 0.5 * (b + c)
    ca = 0.5 * (c + a)
    return [
        np.stack([a, ab, ca]),
        np.stack([ab, b, bc]),
        np.stack([ca, bc, c]),
        np.stack([ab, bc, ca]),
    ]


def _max_edge_length(tri: np.ndarray) -> float:
    return float(max(np.linalg.norm(tri[1] - tri[0]), np.linalg.norm(tri[2] - tri[1]), np.linalg.norm(tri[0] - tri[2])))


def _sample_phi(sdf: SDFGrid, pts: np.ndarray, method: str) -> np.ndarray:
    return np.asarray(sdf.sample(pts, method=method, return_grad=False), dtype=np.float64)


def extract_contact_region(
    active_mesh: TriangleMesh,
    passive_sdf: SDFGrid,
    method: str = "linear",
    max_depth: int = 5,
    min_edge_length: float | None = None,
    candidate_epsilon: float | None = None,
    base_clip_epsilon: float = 0.0,
    face_limit: int | None = None,
) -> ContactTriangles:
    """Extract active surface subset where passive_sdf(phi)<=0.

    A candidate triangle is recursively subdivided until the SDF sign pattern is
    resolved. Mixed leaf triangles are linearly clipped. This gives a geometric
    contact region rather than a mere point cloud.
    """
    h = float(np.max(passive_sdf.spacing))
    if min_edge_length is None:
        min_edge_length = 0.75 * h
    if candidate_epsilon is None:
        candidate_epsilon = 2.0 * h

    triangles_out: list[np.ndarray] = []
    source_faces: list[int] = []

    face_indices = np.arange(active_mesh.face_count, dtype=np.int64)
    if face_limit is not None:
        face_indices = face_indices[:face_limit]
    fv = active_mesh.face_vertices()

    # Vectorized candidate filtering: vertices + edge midpoints + centroid.
    # This avoids one Python-level SDF call per face and is important for the
    # high-face-count validation meshes.
    fvc = fv[face_indices]
    centroids0 = fvc.mean(axis=1)
    mids01 = 0.5 * (fvc[:, 0] + fvc[:, 1])
    mids12 = 0.5 * (fvc[:, 1] + fvc[:, 2])
    mids20 = 0.5 * (fvc[:, 2] + fvc[:, 0])
    probes = np.concatenate([fvc[:, 0], fvc[:, 1], fvc[:, 2], mids01, mids12, mids20, centroids0], axis=0)
    phip = _sample_phi(passive_sdf, probes, method).reshape(7, len(face_indices))
    candidate_mask = np.min(phip, axis=0) <= candidate_epsilon
    candidate_faces = face_indices[candidate_mask]

    for fi in candidate_faces:
        tri = fv[int(fi)]
        stack: list[tuple[np.ndarray, int]] = [(tri, 0)]
        while stack:
            sub, depth = stack.pop()
            phis = _sample_phi(passive_sdf, sub, method)
            if np.max(phis) <= base_clip_epsilon:
                triangles_out.append(sub)
                source_faces.append(int(fi))
            elif np.min(phis) > base_clip_epsilon:
                continue
            else:
                if depth >= max_depth or _max_edge_length(sub) <= min_edge_length:
                    clipped = _clip_triangle_linear(sub, phis, eps=base_clip_epsilon)
                    triangles_out.extend(clipped)
                    source_faces.extend([int(fi)] * len(clipped))
                else:
                    for child in _subdivide_triangle(sub):
                        stack.append((child, depth + 1))

    if len(triangles_out) == 0:
        return ContactTriangles(
            triangles=np.zeros((0, 3, 3), dtype=np.float64),
            phi=np.zeros(0, dtype=np.float64),
            depth=np.zeros(0, dtype=np.float64),
            normals=np.zeros((0, 3), dtype=np.float64),
            source_face=np.zeros(0, dtype=np.int64),
            method=method,
        )

    tris = np.asarray(triangles_out, dtype=np.float64)
    cent = tris.mean(axis=1)
    phi, grad = passive_sdf.sample(cent, method=method, return_grad=True)
    phi = np.asarray(phi, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    normals = np.asarray(normalize_grad(grad), dtype=np.float64)
    depth = np.maximum(0.0, -phi)
    return ContactTriangles(tris, phi, depth, normals, np.asarray(source_faces, dtype=np.int64), method=method)


def label_connected_components(contact: ContactTriangles, threshold: float) -> Tuple[np.ndarray, List[Dict]]:
    """Spatially cluster contact micro-triangles by centroid distance."""
    n = contact.count
    if n == 0:
        contact.labels = np.zeros(0, dtype=np.int64)
        return contact.labels, []
    cent = contact.centroids()
    tree = cKDTree(cent)
    pairs = tree.query_pairs(r=threshold)
    parent = np.arange(n, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)
    roots = np.array([find(i) for i in range(n)])
    unique, labels = np.unique(roots, return_inverse=True)
    contact.labels = labels.astype(np.int64)

    areas = contact.areas()
    components = []
    for lab in range(len(unique)):
        mask = labels == lab
        area = float(np.sum(areas[mask]))
        if area > 0:
            centroid = np.average(cent[mask], axis=0, weights=areas[mask])
        else:
            centroid = cent[mask].mean(axis=0)
        components.append(
            {
                "id": int(lab),
                "triangles": int(np.sum(mask)),
                "area": area,
                "centroid": centroid.tolist(),
                "mean_depth": float(np.average(contact.depth[mask], weights=np.maximum(areas[mask], 1e-30))),
                "max_depth": float(np.max(contact.depth[mask])),
                "mean_normal": np.average(contact.normals[mask], axis=0, weights=np.maximum(areas[mask], 1e-30)).tolist(),
            }
        )
    components.sort(key=lambda x: x["area"], reverse=True)
    return contact.labels, components
