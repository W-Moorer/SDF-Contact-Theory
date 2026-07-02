from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


class AnalyticSDF:
    name: str = "analytic_sdf"

    def __call__(self, points: Array) -> Array:
        raise NotImplementedError

    def gradient(self, points: Array, eps: float = 1e-5) -> Array:
        points = np.asarray(points, dtype=np.float64)
        grad = np.zeros_like(points)
        for k in range(3):
            d = np.zeros(3)
            d[k] = eps
            grad[:, k] = (self(points + d) - self(points - d)) / (2.0 * eps)
        return grad


@dataclass
class FunctionSDF(AnalyticSDF):
    fn: Callable[[Array], Array]
    name: str = "function_sdf"

    def __call__(self, points: Array) -> Array:
        return np.asarray(self.fn(np.asarray(points, dtype=np.float64)), dtype=np.float64)


@dataclass
class BoxSDF(AnalyticSDF):
    extents: tuple[float, float, float]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "box_sdf"

    def __call__(self, points: Array) -> Array:
        p = np.asarray(points, dtype=np.float64) - np.asarray(self.center, dtype=np.float64)
        h = np.asarray(self.extents, dtype=np.float64) / 2.0
        q = np.abs(p) - h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def gradient(self, points: Array, eps: float = 1e-6) -> Array:
        # Finite difference is safer near box edges/corners.
        return super().gradient(points, eps=eps)


@dataclass
class PlaneSDF(AnalyticSDF):
    z0: float = 0.0
    normal_sign: float = 1.0
    name: str = "plane_sdf"

    def __call__(self, points: Array) -> Array:
        p = np.asarray(points, dtype=np.float64)
        return self.normal_sign * (p[:, 2] - self.z0)

    def gradient(self, points: Array, eps: float = 1e-5) -> Array:
        g = np.zeros_like(np.asarray(points, dtype=np.float64))
        g[:, 2] = self.normal_sign
        return g


@dataclass
class TorusSDF(AnalyticSDF):
    R: float = 1.0
    r: float = 0.25
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "torus_sdf"

    def __call__(self, points: Array) -> Array:
        p = np.asarray(points, dtype=np.float64) - np.asarray(self.center, dtype=np.float64)
        rho = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        q0 = rho - self.R
        q1 = p[:, 2]
        return np.sqrt(q0 * q0 + q1 * q1) - self.r

    def gradient(self, points: Array, eps: float = 1e-8) -> Array:
        p = np.asarray(points, dtype=np.float64) - np.asarray(self.center, dtype=np.float64)
        rho = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        q0 = rho - self.R
        q1 = p[:, 2]
        qn = np.sqrt(q0 * q0 + q1 * q1)
        grad = np.zeros_like(p)
        safe_rho = np.maximum(rho, eps)
        safe_qn = np.maximum(qn, eps)
        grad[:, 0] = (q0 / safe_qn) * (p[:, 0] / safe_rho)
        grad[:, 1] = (q0 / safe_qn) * (p[:, 1] / safe_rho)
        grad[:, 2] = q1 / safe_qn
        return grad


@dataclass
class CappedCylinderSDF(AnalyticSDF):
    radius: float = 1.0
    height: float = 1.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "capped_cylinder_sdf"

    def __call__(self, points: Array) -> Array:
        p = np.asarray(points, dtype=np.float64) - np.asarray(self.center, dtype=np.float64)
        d0 = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - self.radius
        d1 = np.abs(p[:, 2]) - self.height / 2.0
        d = np.stack([d0, d1], axis=1)
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside = np.minimum(np.maximum(d0, d1), 0.0)
        return outside + inside


@dataclass
class HollowCylinderSDF(AnalyticSDF):
    outer_radius: float = 1.0
    inner_radius: float = 0.7
    height: float = 1.2
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "hollow_cylinder_sdf"

    def __call__(self, points: Array) -> Array:
        p = np.asarray(points, dtype=np.float64) - np.asarray(self.center, dtype=np.float64)
        outer = CappedCylinderSDF(self.outer_radius, self.height, (0.0, 0.0, 0.0))(p)
        inner_inf = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - self.inner_radius
        # Solid annulus = outer capped cylinder minus infinite inner cylinder.
        return np.maximum(outer, -inner_inf)


@dataclass
class ConeApproxSDF(AnalyticSDF):
    """Approximate signed distance for a vertical finite cone.

    Negative inside the cone. This is used for quick sanity checks; the mesh
    backend should be used for exact cone SDF validation.
    """

    radius: float = 0.8
    height: float = 1.2
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "cone_approx_sdf"

    def __call__(self, points: Array) -> Array:
        p = np.asarray(points, dtype=np.float64) - np.asarray(self.center, dtype=np.float64)
        z_base = -self.height / 2.0
        z_apex = self.height / 2.0
        rho = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        s = (p[:, 2] - z_base) / self.height
        r_at_z = self.radius * (1.0 - s)
        radial = rho - r_at_z
        below = z_base - p[:, 2]
        above = p[:, 2] - z_apex
        outside = np.maximum.reduce([radial, below, above])
        # Not exact Euclidean distance but sign-consistent for clean points.
        return outside


def gradient_fd(fn: Callable[[Array], Array], points: Array, eps: float = 1e-5) -> Array:
    points = np.asarray(points, dtype=np.float64)
    grad = np.zeros_like(points)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        grad[:, k] = (fn(points + d) - fn(points - d)) / (2.0 * eps)
    return grad
