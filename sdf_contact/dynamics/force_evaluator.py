from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sdf_contact.geometry.mesh import TriangleMesh
from sdf_contact.sdf.grid import SDFGrid
from sdf_contact.sdf.interpolation import normalize_grad

try:  # Optional acceleration; the code works without CUDA.
    import numba as nb
    from numba import cuda

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    nb = None
    cuda = None
    NUMBA_AVAILABLE = False


@dataclass
class SurfaceContactResult:
    points: np.ndarray
    forces: np.ndarray
    normals: np.ndarray
    phis: np.ndarray
    depths: np.ndarray
    pressures: np.ndarray
    areas: np.ndarray
    total_force: np.ndarray
    total_torque: np.ndarray
    elastic_energy: float
    contact_area: float
    contact_count: int
    backend_used: str
    method: str

    @property
    def max_depth(self) -> float:
        return float(np.max(self.depths)) if self.depths.size else 0.0

    @property
    def force_norm(self) -> float:
        return float(np.linalg.norm(self.total_force))

    def to_dict(self) -> dict:
        return {
            "total_force": self.total_force.tolist(),
            "total_force_norm": self.force_norm,
            "total_torque": self.total_torque.tolist(),
            "total_torque_norm": float(np.linalg.norm(self.total_torque)),
            "elastic_energy": float(self.elastic_energy),
            "contact_area": float(self.contact_area),
            "contact_count": int(self.contact_count),
            "max_depth": self.max_depth,
            "pressure_max": float(np.max(self.pressures)) if self.pressures.size else 0.0,
            "backend_used": self.backend_used,
            "method": self.method,
        }


# -----------------------------------------------------------------------------
# Numba CPU kernels
# -----------------------------------------------------------------------------
if NUMBA_AVAILABLE:

    @nb.njit(cache=True)
    def _cr_w(t, a):
        t2 = t * t
        t3 = t2 * t
        if a == 0:
            return -0.5 * t + t2 - 0.5 * t3
        if a == 1:
            return 1.0 - 2.5 * t2 + 1.5 * t3
        if a == 2:
            return 0.5 * t + 2.0 * t2 - 1.5 * t3
        return -0.5 * t2 + 0.5 * t3

    @nb.njit(cache=True)
    def _cr_dw(t, a):
        t2 = t * t
        if a == 0:
            return -0.5 + 2.0 * t - 1.5 * t2
        if a == 1:
            return -5.0 * t + 4.5 * t2
        if a == 2:
            return 0.5 + 4.0 * t - 4.5 * t2
        return -t + 1.5 * t2

    @nb.njit(cache=True)
    def _linear_sample(values, origin, spacing, x, y, z):
        nx, ny, nz = values.shape
        ux = (x - origin[0]) / spacing[0]
        uy = (y - origin[1]) / spacing[1]
        uz = (z - origin[2]) / spacing[2]
        ix = int(np.floor(ux)); iy = int(np.floor(uy)); iz = int(np.floor(uz))
        ax = ux - ix; ay = uy - iy; az = uz - iz
        if ix < 0:
            ix = 0; ax = 0.0
        elif ix > nx - 2:
            ix = nx - 2; ax = 1.0
        if iy < 0:
            iy = 0; ay = 0.0
        elif iy > ny - 2:
            iy = ny - 2; ay = 1.0
        if iz < 0:
            iz = 0; az = 0.0
        elif iz > nz - 2:
            iz = nz - 2; az = 1.0

        c000 = float(values[ix, iy, iz])
        c100 = float(values[ix + 1, iy, iz])
        c010 = float(values[ix, iy + 1, iz])
        c110 = float(values[ix + 1, iy + 1, iz])
        c001 = float(values[ix, iy, iz + 1])
        c101 = float(values[ix + 1, iy, iz + 1])
        c011 = float(values[ix, iy + 1, iz + 1])
        c111 = float(values[ix + 1, iy + 1, iz + 1])

        c00 = c000 * (1.0 - ax) + c100 * ax
        c10 = c010 * (1.0 - ax) + c110 * ax
        c01 = c001 * (1.0 - ax) + c101 * ax
        c11 = c011 * (1.0 - ax) + c111 * ax
        c0 = c00 * (1.0 - ay) + c10 * ay
        c1 = c01 * (1.0 - ay) + c11 * ay
        phi = c0 * (1.0 - az) + c1 * az

        gx = (((c100 - c000) * (1.0 - ay) + (c110 - c010) * ay) * (1.0 - az) + ((c101 - c001) * (1.0 - ay) + (c111 - c011) * ay) * az) / spacing[0]
        gy = (((c010 - c000) * (1.0 - ax) + (c110 - c100) * ax) * (1.0 - az) + ((c011 - c001) * (1.0 - ax) + (c111 - c101) * ax) * az) / spacing[1]
        gz = (c1 - c0) / spacing[2]
        return phi, gx, gy, gz

    @nb.njit(cache=True)
    def _cubic_sample(values, origin, spacing, x, y, z):
        nx, ny, nz = values.shape
        ux = (x - origin[0]) / spacing[0]
        uy = (y - origin[1]) / spacing[1]
        uz = (z - origin[2]) / spacing[2]
        bx = int(np.floor(ux)); by = int(np.floor(uy)); bz = int(np.floor(uz))
        tx = ux - bx; ty = uy - by; tz = uz - bz
        if bx < 1:
            bx = 1; tx = 0.0
        elif bx > nx - 3:
            bx = nx - 3; tx = 1.0
        if by < 1:
            by = 1; ty = 0.0
        elif by > ny - 3:
            by = ny - 3; ty = 1.0
        if bz < 1:
            bz = 1; tz = 0.0
        elif bz > nz - 3:
            bz = nz - 3; tz = 1.0

        phi = 0.0; gx = 0.0; gy = 0.0; gz = 0.0
        local_min = 1.0e30; local_max = -1.0e30
        for a in range(4):
            ix = bx + a - 1
            wx = _cr_w(tx, a); dwx = _cr_dw(tx, a)
            for b in range(4):
                iy = by + b - 1
                wy = _cr_w(ty, b); dwy = _cr_dw(ty, b)
                for c in range(4):
                    iz = bz + c - 1
                    wz = _cr_w(tz, c); dwz = _cr_dw(tz, c)
                    val = float(values[ix, iy, iz])
                    w = wx * wy * wz
                    phi += w * val
                    gx += dwx * wy * wz * val
                    gy += wx * dwy * wz * val
                    gz += wx * wy * dwz * val
                    if val < local_min:
                        local_min = val
                    if val > local_max:
                        local_max = val
        if phi < local_min:
            phi = local_min
        elif phi > local_max:
            phi = local_max
        return phi, gx / spacing[0], gy / spacing[1], gz / spacing[2]

    @nb.njit(parallel=True, cache=True)
    def _surface_forces_cpu_kernel(centroids_local, areas_local, values, origin, spacing, translation, velocity, kn, alpha, cn, method_flag, points, forces, normals, phis, depths, pressures, energies):
        n = centroids_local.shape[0]
        for i in nb.prange(n):
            x = centroids_local[i, 0] + translation[0]
            y = centroids_local[i, 1] + translation[1]
            z = centroids_local[i, 2] + translation[2]
            if method_flag == 0:
                phi, gx, gy, gz = _linear_sample(values, origin, spacing, x, y, z)
            else:
                phi, gx, gy, gz = _cubic_sample(values, origin, spacing, x, y, z)
            gl = np.sqrt(gx * gx + gy * gy + gz * gz)
            if gl < 1.0e-12:
                nx_ = 0.0; ny_ = 0.0; nz_ = 1.0
            else:
                nx_ = gx / gl; ny_ = gy / gl; nz_ = gz / gl
            depth = -phi if phi < 0.0 else 0.0
            pressure = 0.0
            energy = 0.0
            if depth > 0.0:
                pressure = kn * (depth ** alpha)
                vn = velocity[0] * nx_ + velocity[1] * ny_ + velocity[2] * nz_
                if vn < 0.0 and cn > 0.0:
                    pressure += cn * (-vn)
                energy = kn / (alpha + 1.0) * (depth ** (alpha + 1.0)) * areas_local[i]
            fx = pressure * nx_ * areas_local[i]
            fy = pressure * ny_ * areas_local[i]
            fz = pressure * nz_ * areas_local[i]
            points[i, 0] = x; points[i, 1] = y; points[i, 2] = z
            forces[i, 0] = fx; forces[i, 1] = fy; forces[i, 2] = fz
            normals[i, 0] = nx_; normals[i, 1] = ny_; normals[i, 2] = nz_
            phis[i] = phi; depths[i] = depth; pressures[i] = pressure; energies[i] = energy

# -----------------------------------------------------------------------------
# CUDA device/kernel definitions.  They are intentionally simple: one face
# quadrature point per thread, host-side reduction.  This is robust on consumer
# RTX cards and avoids atomics for the first validation version.
# -----------------------------------------------------------------------------
if NUMBA_AVAILABLE:

    @cuda.jit(device=True)
    def _cr_w_dev(t, a):
        t2 = t * t
        t3 = t2 * t
        if a == 0:
            return -0.5 * t + t2 - 0.5 * t3
        if a == 1:
            return 1.0 - 2.5 * t2 + 1.5 * t3
        if a == 2:
            return 0.5 * t + 2.0 * t2 - 1.5 * t3
        return -0.5 * t2 + 0.5 * t3

    @cuda.jit(device=True)
    def _cr_dw_dev(t, a):
        t2 = t * t
        if a == 0:
            return -0.5 + 2.0 * t - 1.5 * t2
        if a == 1:
            return -5.0 * t + 4.5 * t2
        if a == 2:
            return 0.5 + 4.0 * t - 4.5 * t2
        return -t + 1.5 * t2

    @cuda.jit(device=True)
    def _linear_sample_dev(values, origin, spacing, x, y, z):
        nx = values.shape[0]; ny = values.shape[1]; nz = values.shape[2]
        ux = (x - origin[0]) / spacing[0]
        uy = (y - origin[1]) / spacing[1]
        uz = (z - origin[2]) / spacing[2]
        ix = int(np.floor(ux)); iy = int(np.floor(uy)); iz = int(np.floor(uz))
        ax = ux - ix; ay = uy - iy; az = uz - iz
        if ix < 0:
            ix = 0; ax = 0.0
        elif ix > nx - 2:
            ix = nx - 2; ax = 1.0
        if iy < 0:
            iy = 0; ay = 0.0
        elif iy > ny - 2:
            iy = ny - 2; ay = 1.0
        if iz < 0:
            iz = 0; az = 0.0
        elif iz > nz - 2:
            iz = nz - 2; az = 1.0
        c000 = values[ix, iy, iz]
        c100 = values[ix + 1, iy, iz]
        c010 = values[ix, iy + 1, iz]
        c110 = values[ix + 1, iy + 1, iz]
        c001 = values[ix, iy, iz + 1]
        c101 = values[ix + 1, iy, iz + 1]
        c011 = values[ix, iy + 1, iz + 1]
        c111 = values[ix + 1, iy + 1, iz + 1]
        c00 = c000 * (1.0 - ax) + c100 * ax
        c10 = c010 * (1.0 - ax) + c110 * ax
        c01 = c001 * (1.0 - ax) + c101 * ax
        c11 = c011 * (1.0 - ax) + c111 * ax
        c0 = c00 * (1.0 - ay) + c10 * ay
        c1 = c01 * (1.0 - ay) + c11 * ay
        phi = c0 * (1.0 - az) + c1 * az
        gx = (((c100 - c000) * (1.0 - ay) + (c110 - c010) * ay) * (1.0 - az) + ((c101 - c001) * (1.0 - ay) + (c111 - c011) * ay) * az) / spacing[0]
        gy = (((c010 - c000) * (1.0 - ax) + (c110 - c100) * ax) * (1.0 - az) + ((c011 - c001) * (1.0 - ax) + (c111 - c101) * ax) * az) / spacing[1]
        gz = (c1 - c0) / spacing[2]
        return phi, gx, gy, gz

    @cuda.jit(device=True)
    def _cubic_sample_dev(values, origin, spacing, x, y, z):
        nx = values.shape[0]; ny = values.shape[1]; nz = values.shape[2]
        ux = (x - origin[0]) / spacing[0]
        uy = (y - origin[1]) / spacing[1]
        uz = (z - origin[2]) / spacing[2]
        bx = int(np.floor(ux)); by = int(np.floor(uy)); bz = int(np.floor(uz))
        tx = ux - bx; ty = uy - by; tz = uz - bz
        if bx < 1:
            bx = 1; tx = 0.0
        elif bx > nx - 3:
            bx = nx - 3; tx = 1.0
        if by < 1:
            by = 1; ty = 0.0
        elif by > ny - 3:
            by = ny - 3; ty = 1.0
        if bz < 1:
            bz = 1; tz = 0.0
        elif bz > nz - 3:
            bz = nz - 3; tz = 1.0
        phi = 0.0; gx = 0.0; gy = 0.0; gz = 0.0
        local_min = 1.0e30; local_max = -1.0e30
        for a in range(4):
            ix = bx + a - 1
            wx = _cr_w_dev(tx, a); dwx = _cr_dw_dev(tx, a)
            for b in range(4):
                iy = by + b - 1
                wy = _cr_w_dev(ty, b); dwy = _cr_dw_dev(ty, b)
                for c in range(4):
                    iz = bz + c - 1
                    wz = _cr_w_dev(tz, c); dwz = _cr_dw_dev(tz, c)
                    val = values[ix, iy, iz]
                    phi += wx * wy * wz * val
                    gx += dwx * wy * wz * val
                    gy += wx * dwy * wz * val
                    gz += wx * wy * dwz * val
                    if val < local_min:
                        local_min = val
                    if val > local_max:
                        local_max = val
        if phi < local_min:
            phi = local_min
        elif phi > local_max:
            phi = local_max
        return phi, gx / spacing[0], gy / spacing[1], gz / spacing[2]

    @cuda.jit
    def _surface_forces_cuda_kernel(centroids_local, areas_local, values, origin, spacing, translation, velocity, kn, alpha, cn, method_flag, points, forces, normals, phis, depths, pressures, energies):
        i = cuda.grid(1)
        if i >= centroids_local.shape[0]:
            return
        x = centroids_local[i, 0] + translation[0]
        y = centroids_local[i, 1] + translation[1]
        z = centroids_local[i, 2] + translation[2]
        if method_flag == 0:
            phi, gx, gy, gz = _linear_sample_dev(values, origin, spacing, x, y, z)
        else:
            phi, gx, gy, gz = _cubic_sample_dev(values, origin, spacing, x, y, z)
        gl = np.sqrt(gx * gx + gy * gy + gz * gz)
        if gl < 1.0e-12:
            nx_ = 0.0; ny_ = 0.0; nz_ = 1.0
        else:
            nx_ = gx / gl; ny_ = gy / gl; nz_ = gz / gl
        depth = -phi if phi < 0.0 else 0.0
        pressure = 0.0
        energy = 0.0
        if depth > 0.0:
            pressure = kn * (depth ** alpha)
            vn = velocity[0] * nx_ + velocity[1] * ny_ + velocity[2] * nz_
            if vn < 0.0 and cn > 0.0:
                pressure += cn * (-vn)
            energy = kn / (alpha + 1.0) * (depth ** (alpha + 1.0)) * areas_local[i]
        fx = pressure * nx_ * areas_local[i]
        fy = pressure * ny_ * areas_local[i]
        fz = pressure * nz_ * areas_local[i]
        points[i, 0] = x; points[i, 1] = y; points[i, 2] = z
        forces[i, 0] = fx; forces[i, 1] = fy; forces[i, 2] = fz
        normals[i, 0] = nx_; normals[i, 1] = ny_; normals[i, 2] = nz_
        phis[i] = phi; depths[i] = depth; pressures[i] = pressure; energies[i] = energy


class SurfaceContactEvaluator:
    """Batched face-centroid contact force evaluator for explicit dynamics.

    It preserves the previous SDF interpolation and penalty pressure law, but uses
    four quadrature samples per active face for speed.  The exact clipped contact
    manifold extractor is still used for static validation and can be sampled at
    selected dynamic frames if needed.
    """

    def __init__(self, active_mesh: TriangleMesh, passive_grid: SDFGrid, method: str = "linear", backend: str = "auto") -> None:
        self.active_mesh = active_mesh
        self.grid = passive_grid
        self.method = "cubic" if method.lower().startswith("cub") else "linear"
        self.backend_request = backend.lower()
        fv = active_mesh.face_vertices()
        face_centroids = fv.mean(axis=1).astype(np.float64)
        face_areas = active_mesh.face_areas().astype(np.float64)
        # Four-point symmetric surface quadrature per triangle: three vertices
        # plus centroid.  This catches sharp first-contact events, e.g. cone apex
        # and wavy valleys, much earlier than centroid-only sampling while keeping
        # the same SDF interpolation and pressure law.
        self.centroids_local = np.concatenate([fv[:, 0], fv[:, 1], fv[:, 2], face_centroids], axis=0).astype(np.float64)
        self.areas = np.concatenate([face_areas, face_areas, face_areas, face_areas], axis=0).astype(np.float64) * 0.25
        self._cuda_ready = False
        self._cuda_arrays = None
        if self.backend_request in {"auto", "cuda"} and NUMBA_AVAILABLE:
            try:
                self._cuda_ready = bool(cuda.is_available())
            except Exception:
                self._cuda_ready = False
        if self.backend_request == "cuda" and not self._cuda_ready:
            raise RuntimeError("CUDA backend requested, but numba.cuda reports no available CUDA device.")
        if self._cuda_ready and self.backend_request in {"auto", "cuda"}:
            self._init_cuda_arrays()

    @property
    def backend_used(self) -> str:
        if self._cuda_ready and self.backend_request in {"auto", "cuda"}:
            return "cuda"
        if NUMBA_AVAILABLE and self.backend_request in {"auto", "numba", "numba_cpu"}:
            return "numba_cpu"
        return "numpy"

    def _init_cuda_arrays(self) -> None:  # pragma: no cover - requires GPU
        n = self.centroids_local.shape[0]
        self._cuda_arrays = {
            "centroids": cuda.to_device(self.centroids_local),
            "areas": cuda.to_device(self.areas),
            "values": cuda.to_device(np.asarray(self.grid.values, dtype=np.float32)),
            "origin": cuda.to_device(np.asarray(self.grid.origin, dtype=np.float64)),
            "spacing": cuda.to_device(np.asarray(self.grid.spacing, dtype=np.float64)),
            "translation": cuda.device_array(3, dtype=np.float64),
            "velocity": cuda.device_array(3, dtype=np.float64),
            "points": cuda.device_array((n, 3), dtype=np.float64),
            "forces": cuda.device_array((n, 3), dtype=np.float64),
            "normals": cuda.device_array((n, 3), dtype=np.float64),
            "phis": cuda.device_array(n, dtype=np.float64),
            "depths": cuda.device_array(n, dtype=np.float64),
            "pressures": cuda.device_array(n, dtype=np.float64),
            "energies": cuda.device_array(n, dtype=np.float64),
        }

    def evaluate(self, translation: np.ndarray, velocity: np.ndarray, kn: float, alpha: float = 1.0, cn: float = 0.0, torque_center: np.ndarray | None = None) -> SurfaceContactResult:
        translation = np.asarray(translation, dtype=np.float64).reshape(3)
        velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
        if torque_center is None:
            torque_center = np.zeros(3, dtype=np.float64)
        torque_center = np.asarray(torque_center, dtype=np.float64).reshape(3)
        method_flag = 1 if self.method == "cubic" else 0
        backend = self.backend_used
        n = self.centroids_local.shape[0]

        if backend == "cuda":  # pragma: no cover - requires GPU
            arr = self._cuda_arrays
            arr["translation"].copy_to_device(translation)
            arr["velocity"].copy_to_device(velocity)
            threads = 128
            blocks = (n + threads - 1) // threads
            _surface_forces_cuda_kernel[blocks, threads](
                arr["centroids"], arr["areas"], arr["values"], arr["origin"], arr["spacing"], arr["translation"], arr["velocity"],
                float(kn), float(alpha), float(cn), method_flag,
                arr["points"], arr["forces"], arr["normals"], arr["phis"], arr["depths"], arr["pressures"], arr["energies"],
            )
            points = arr["points"].copy_to_host()
            forces = arr["forces"].copy_to_host()
            normals = arr["normals"].copy_to_host()
            phis = arr["phis"].copy_to_host()
            depths = arr["depths"].copy_to_host()
            pressures = arr["pressures"].copy_to_host()
            energies = arr["energies"].copy_to_host()
        elif backend == "numba_cpu":
            points = np.empty((n, 3), dtype=np.float64)
            forces = np.empty((n, 3), dtype=np.float64)
            normals = np.empty((n, 3), dtype=np.float64)
            phis = np.empty(n, dtype=np.float64)
            depths = np.empty(n, dtype=np.float64)
            pressures = np.empty(n, dtype=np.float64)
            energies = np.empty(n, dtype=np.float64)
            _surface_forces_cpu_kernel(
                self.centroids_local,
                self.areas,
                np.asarray(self.grid.values, dtype=np.float32),
                np.asarray(self.grid.origin, dtype=np.float64),
                np.asarray(self.grid.spacing, dtype=np.float64),
                translation,
                velocity,
                float(kn),
                float(alpha),
                float(cn),
                method_flag,
                points,
                forces,
                normals,
                phis,
                depths,
                pressures,
                energies,
            )
        else:
            points = self.centroids_local + translation[None, :]
            phi, grad = self.grid.sample(points, method=self.method, return_grad=True)
            phi = np.asarray(phi, dtype=np.float64)
            normals = np.asarray(normalize_grad(np.asarray(grad, dtype=np.float64)), dtype=np.float64)
            depths = np.maximum(0.0, -phi)
            pressures = np.zeros_like(depths)
            active = depths > 0.0
            if np.any(active):
                pressures[active] = kn * np.power(depths[active], alpha)
                active_idx = np.flatnonzero(active)
                vn = normals[active_idx] @ velocity
                approaching = vn < 0.0
                pressures[active_idx[approaching]] += cn * (-vn[approaching])
            forces = pressures[:, None] * normals * self.areas[:, None]
            phis = phi
            energies = kn / (alpha + 1.0) * np.power(depths, alpha + 1.0) * self.areas

        active_mask = depths > 0.0
        total_force = np.sum(forces, axis=0)
        total_torque = np.sum(np.cross(points - torque_center[None, :], forces), axis=0)
        return SurfaceContactResult(
            points=points,
            forces=forces,
            normals=normals,
            phis=phis,
            depths=depths,
            pressures=pressures,
            areas=self.areas.copy(),
            total_force=total_force,
            total_torque=total_torque,
            elastic_energy=float(np.sum(energies)),
            contact_area=float(np.sum(self.areas[active_mask])),
            contact_count=int(np.sum(active_mask)),
            backend_used=backend,
            method=self.method,
        )
