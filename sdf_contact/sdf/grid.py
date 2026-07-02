from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .interpolation import normalize_grad, sample_tricubic, sample_trilinear


@dataclass
class SDFGrid:
    values: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    name: str = "sdf_grid"

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float32)
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        self.spacing = np.asarray(self.spacing, dtype=np.float64).reshape(3)
        if self.values.ndim != 3:
            raise ValueError("values must be a 3D array indexed [ix, iy, iz]")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(map(int, self.values.shape))

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        hi = self.origin + self.spacing * (np.asarray(self.shape) - 1)
        return self.origin.copy(), hi

    @classmethod
    def from_analytic(cls, sdf_fn, bounds, resolution: int | tuple[int, int, int], name: str = "analytic_grid") -> "SDFGrid":
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
        vals = sdf_fn(pts).reshape(tuple(shape)).astype(np.float32)
        return cls(vals, mn, spacing, name=name)

    def sample(self, points: np.ndarray, method: str = "linear", return_grad: bool = True):
        method = method.lower()
        if method in {"linear", "trilinear", "lin"}:
            return sample_trilinear(self.values, self.origin, self.spacing, points, return_grad=return_grad)
        if method in {"cubic", "tricubic", "cub"}:
            return sample_tricubic(self.values, self.origin, self.spacing, points, return_grad=return_grad)
        raise ValueError(f"Unknown interpolation method: {method}")

    def normal(self, points: np.ndarray, method: str = "linear") -> np.ndarray:
        phi, grad = self.sample(points, method=method, return_grad=True)
        return normalize_grad(grad)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, values=self.values, origin=self.origin, spacing=self.spacing, name=self.name)

    @classmethod
    def load_npz(cls, path: str | Path) -> "SDFGrid":
        data = np.load(path)
        return cls(data["values"], data["origin"], data["spacing"], str(data.get("name", "sdf_grid")))


def padded_bounds_from_meshes(meshes, padding: float = 0.25):
    mins, maxs = [], []
    for m in meshes:
        mn, mx = m.bounds
        mins.append(mn)
        maxs.append(mx)
    mn = np.min(np.stack(mins), axis=0)
    mx = np.max(np.stack(maxs), axis=0)
    pad = np.full(3, padding, dtype=np.float64)
    return mn - pad, mx + pad
