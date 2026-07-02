from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from sdf_contact.geometry.mesh import TriangleMesh
from .grid import SDFGrid


def mesh_to_sdf_grid(
    mesh: TriangleMesh,
    bounds,
    resolution: int | tuple[int, int, int] = 64,
    backend: str = "auto",
    device: str | None = "cuda:0",
    analytic_sign_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    name: str | None = None,
) -> SDFGrid:
    backend = backend.lower()
    if backend == "auto":
        try:
            import warp  # noqa: F401
            backend = "warp"
        except Exception:
            try:
                import cupy  # noqa: F401
                if analytic_sign_fn is not None:
                    backend = "cupy"
                else:
                    backend = "cpu"
            except Exception:
                backend = "cpu"

    if backend == "warp":
        from .mesh_to_sdf_warp import mesh_to_sdf_warp
        return mesh_to_sdf_warp(mesh, bounds, resolution, device=device, analytic_sign_fn=analytic_sign_fn, name=name)
    if backend == "cupy":
        from .mesh_to_sdf_cupy import mesh_to_sdf_cupy
        return mesh_to_sdf_cupy(mesh, bounds, resolution, analytic_sign_fn=analytic_sign_fn, name=name)
    if backend in {"cpu", "numba"}:
        from .mesh_to_sdf_cpu import mesh_to_sdf_cpu
        return mesh_to_sdf_cpu(mesh, bounds, resolution, analytic_sign_fn=analytic_sign_fn, name=name)
    raise ValueError(f"Unknown Mesh→SDF backend: {backend}")
