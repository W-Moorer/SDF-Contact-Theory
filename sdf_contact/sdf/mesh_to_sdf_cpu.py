from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from sdf_contact.geometry.mesh import TriangleMesh
from .grid import SDFGrid

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    njit = None
    prange = range


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @njit(cache=True)
    def _sub(a, b):
        return np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])

    @njit(cache=True)
    def _point_triangle_distance2(p, a, b, c):
        # Christer Ericson, Real-Time Collision Detection.
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = _dot(ab, ap)
        d2 = _dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            d = p - a
            return _dot(d, d)

        bp = p - b
        d3 = _dot(ab, bp)
        d4 = _dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            d = p - b
            return _dot(d, d)

        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            q = a + v * ab
            d = p - q
            return _dot(d, d)

        cp = p - c
        d5 = _dot(ab, cp)
        d6 = _dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            d = p - c
            return _dot(d, d)

        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            q = a + w * ac
            d = p - q
            return _dot(d, d)

        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            q = b + w * (c - b)
            d = p - q
            return _dot(d, d)

        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        q = a + ab * v + ac * w
        d = p - q
        return _dot(d, d)

    @njit(cache=True)
    def _cross(a, b):
        return np.array([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])

    @njit(cache=True)
    def _ray_triangle_intersect(orig, direction, a, b, c):
        eps = 1e-10
        e1 = b - a
        e2 = c - a
        h = _cross(direction, e2)
        det = _dot(e1, h)
        if det > -eps and det < eps:
            return False
        inv_det = 1.0 / det
        s = orig - a
        u = inv_det * _dot(s, h)
        if u < eps or u > 1.0 - eps:
            return False
        q = _cross(s, e1)
        v = inv_det * _dot(direction, q)
        if v < eps or u + v > 1.0 - eps:
            return False
        t = inv_det * _dot(e2, q)
        return t > eps

    @njit(parallel=True, cache=True)
    def _unsigned_distances(points, triangles):
        n = points.shape[0]
        out = np.empty(n, dtype=np.float64)
        for ip in prange(n):
            p = points[ip]
            best = 1e100
            for it in range(triangles.shape[0]):
                d2 = _point_triangle_distance2(p, triangles[it, 0], triangles[it, 1], triangles[it, 2])
                if d2 < best:
                    best = d2
            out[ip] = np.sqrt(best)
        return out

    @njit(parallel=True, cache=True)
    def _parity_signs(points, triangles):
        n = points.shape[0]
        out = np.empty(n, dtype=np.float64)
        direction = np.array([1.0, 0.3713906763541037, 0.193452115410])
        direction = direction / np.sqrt(_dot(direction, direction))
        for ip in prange(n):
            p = points[ip]
            cnt = 0
            for it in range(triangles.shape[0]):
                if _ray_triangle_intersect(p, direction, triangles[it, 0], triangles[it, 1], triangles[it, 2]):
                    cnt += 1
            out[ip] = -1.0 if (cnt % 2 == 1) else 1.0
        return out


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


def mesh_to_sdf_cpu(
    mesh: TriangleMesh,
    bounds,
    resolution: int | tuple[int, int, int] = 64,
    analytic_sign_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    chunk_size: int = 16384,
    sign_method: str = "analytic_or_parity",
    name: str | None = None,
) -> SDFGrid:
    """Generate a signed distance grid from a triangle mesh using Numba.

    This backend is intentionally simple and robust rather than fast. For
    production-resolution grids use the Warp backend.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required for CPU mesh_to_sdf backend")
    pts, origin, spacing, shape = _grid_points(bounds, resolution)
    triangles = np.asarray(mesh.face_vertices(), dtype=np.float64)
    out = np.empty(pts.shape[0], dtype=np.float32)
    for start in range(0, len(pts), chunk_size):
        end = min(start + chunk_size, len(pts))
        pchunk = pts[start:end]
        dist = _unsigned_distances(pchunk, triangles)
        if analytic_sign_fn is not None:
            sign = np.where(analytic_sign_fn(pchunk) < 0.0, -1.0, 1.0)
        else:
            if sign_method not in {"parity", "analytic_or_parity"}:
                raise ValueError("CPU backend needs analytic_sign_fn or sign_method='parity'")
            sign = _parity_signs(pchunk, triangles)
        out[start:end] = (dist * sign).astype(np.float32)
    return SDFGrid(out.reshape(shape), origin, spacing, name=name or f"{mesh.name}_mesh_sdf_cpu")
