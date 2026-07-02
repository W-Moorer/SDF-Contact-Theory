from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from sdf_contact.geometry.mesh import TriangleMesh
from .grid import SDFGrid


def _grid_points(bounds, resolution):
    mn, mx = np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)
    if isinstance(resolution, int):
        shape = np.array([resolution, resolution, resolution], dtype=np.int64)
    else:
        shape = np.asarray(resolution, dtype=np.int64)
    spacing = (mx - mn) / (shape - 1)
    xs = np.linspace(mn[0], mx[0], shape[0])
    ys = np.linspace(mn[1], mx[1], shape[1])
    zs = np.linspace(mn[2], mx[2], shape[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return pts, mn, spacing, tuple(shape)


def mesh_to_sdf_warp(
    mesh: TriangleMesh,
    bounds,
    resolution: int | tuple[int, int, int] = 128,
    device: str | None = "cuda:0",
    max_query_distance: float | None = None,
    chunk_size: int = 1_000_000,
    analytic_sign_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    name: str | None = None,
) -> SDFGrid:
    """Generate SDF grid with NVIDIA Warp Mesh BVH.

    Warp's mesh_query_point_sign_normal is used when available. If query.sign
    is unreliable for a problematic mesh, pass analytic_sign_fn to override sign.
    """
    try:
        import warp as wp
    except Exception as exc:
        raise RuntimeError("warp-lang is not installed") from exc

    wp.init()

    @wp.kernel
    def _query_kernel(points: wp.array(dtype=wp.vec3), mesh_id: wp.uint64, max_dist: float, out: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        x = points[tid]
        q = wp.mesh_query_point_sign_normal(mesh_id, x, max_dist)
        if q.result:
            p = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
            delta = x - p
            out[tid] = wp.length(delta) * q.sign
        else:
            out[tid] = max_dist

    pts, origin, spacing, shape = _grid_points(bounds, resolution)
    if max_query_distance is None:
        max_query_distance = float(np.linalg.norm(np.asarray(bounds[1]) - np.asarray(bounds[0])))

    # Warp expects flattened triangle vertex indices.
    points_wp = wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3, device=device)
    indices_wp = wp.array(mesh.faces.reshape(-1).astype(np.int32), dtype=wp.int32, device=device)
    wmesh = wp.Mesh(points=points_wp, indices=indices_wp)

    out = np.empty(len(pts), dtype=np.float32)
    with wp.ScopedDevice(device):
        for start in range(0, len(pts), chunk_size):
            end = min(start + chunk_size, len(pts))
            p_wp = wp.array(pts[start:end].astype(np.float32), dtype=wp.vec3, device=device)
            d_wp = wp.empty(end - start, dtype=wp.float32, device=device)
            wp.launch(_query_kernel, dim=end - start, inputs=[p_wp, wmesh.id, float(max_query_distance), d_wp], device=device)
            out[start:end] = d_wp.numpy()
    if analytic_sign_fn is not None:
        sign = np.where(analytic_sign_fn(pts) < 0.0, -1.0, 1.0).astype(np.float32)
        out = np.abs(out) * sign
    return SDFGrid(out.reshape(shape), origin, spacing, name=name or f"{mesh.name}_mesh_sdf_warp")
