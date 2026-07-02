from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sdf_contact.geometry.mesh import TriangleMesh
from sdf_contact.sdf.grid import SDFGrid
from sdf_contact.sdf.interpolation import normalize_grad

try:
    import numba as nb
    from numba import cuda

    NUMBA_AVAILABLE = True
except Exception:
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
# Quaternion helpers (pure numpy, no torch)
# -----------------------------------------------------------------------------
def _quat_mul(q, r):
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return np.array([
        qw * rw - qx * rx - qy * ry - qz * rz,
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw,
    ], dtype=np.float64)


def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_rotate(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return _quat_mul(_quat_mul(q, qv), _quat_conjugate(q))[1:4]


def _quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-30 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _quat_derivative(q, omega):
    qw, qx, qy, qz = q
    wx, wy, wz = omega
    return 0.5 * np.array([
        -qx * wx - qy * wy - qz * wz,
         qw * wx + qy * wz - qz * wy,
         qw * wy - qx * wz + qz * wx,
         qw * wz + qx * wy - qy * wx,
    ], dtype=np.float64)


def _rotation_matrix_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return _quat_normalize(np.array([
            0.25 / s,
            (R[2, 1] - R[1, 2]) * s,
            (R[0, 2] - R[2, 0]) * s,
            (R[1, 0] - R[0, 1]) * s,
        ]))
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return _quat_normalize(np.array([
            (R[2, 1] - R[1, 2]) / s, 0.25 * s,
            (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
        ]))
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return _quat_normalize(np.array([
            (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
            0.25 * s, (R[1, 2] + R[2, 1]) / s,
        ]))
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return _quat_normalize(np.array([
        (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
        (R[1, 2] + R[2, 1]) / s, 0.25 * s,
    ]))


def _compute_inertia_tensor(vertices, faces, mass):
    face_verts = vertices[faces]
    e01 = face_verts[:, 1] - face_verts[:, 0]
    e02 = face_verts[:, 2] - face_verts[:, 0]
    cross = np.cross(e01, e02)
    tet_vols = np.abs(np.sum(face_verts[:, 0] * cross, axis=1)) / 6.0
    total_vol = float(np.sum(tet_vols))
    if total_vol < 1e-30:
        return np.eye(3, dtype=np.float64) * (mass / 6.0)
    origin = np.zeros(3, dtype=np.float64)
    I = np.zeros((3, 3), dtype=np.float64)
    for k in range(len(faces)):
        v0, v1, v2 = face_verts[k]
        w = tet_vols[k] / total_vol
        tet = np.stack([v0, v1, v2, origin], axis=0)
        for s in range(4):
            for t in range(4):
                I[0, 0] += w * (tet[s, 1] * tet[t, 1] + tet[s, 2] * tet[t, 2])
                I[1, 1] += w * (tet[s, 0] * tet[t, 0] + tet[s, 2] * tet[t, 2])
                I[2, 2] += w * (tet[s, 0] * tet[t, 0] + tet[s, 1] * tet[t, 1])
                I[0, 1] -= w * tet[s, 0] * tet[t, 1]
                I[0, 2] -= w * tet[s, 0] * tet[t, 2]
                I[1, 2] -= w * tet[s, 1] * tet[t, 2]
    I[1, 0] = I[0, 1]; I[2, 0] = I[0, 2]; I[2, 1] = I[1, 2]
    return mass * I


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

    @nb.njit(cache=True)
    def _rotate_point_batch_cpu(pts, quat, com_local):
        n = pts.shape[0]
        out = np.empty_like(pts)
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        for i in range(n):
            dx = pts[i, 0] - com_local[0]
            dy = pts[i, 1] - com_local[1]
            dz = pts[i, 2] - com_local[2]
            tw = -qx * dx - qy * dy - qz * dz
            tx = qw * dx + qy * dz - qz * dy
            ty = qw * dy - qx * dz + qz * dx
            tz = qw * dz + qx * dy - qy * dx
            rx = tw * (-qx) + tx * qw + ty * (-qz) - tz * (-qy)
            ry = tw * (-qy) - tx * (-qz) + ty * qw + tz * (-qx)
            rz = tw * (-qz) + tx * (-qy) - ty * (-qx) + tz * qw
            out[i, 0] = rx + com_local[0]
            out[i, 1] = ry + com_local[1]
            out[i, 2] = rz + com_local[2]
        return out

    @nb.njit(parallel=True, cache=True)
    def _surface_forces_cpu_kernel(
        centroids_local, areas_local, values, origin, spacing,
        translation, velocity, kn, alpha, cn,
        mu, ct, method_flag,
        points, forces, normals, phis, depths, pressures, energies,
        tangential_forces,
    ):
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
            fn = pressure * areas_local[i]
            fx = fn * nx_; fy = fn * ny_; fz = fn * nz_
            vtx = 0.0; vty = 0.0; vtz = 0.0
            if mu > 0.0 and depth > 0.0:
                vtx = velocity[0] - (velocity[0] * nx_ + velocity[1] * ny_ + velocity[2] * nz_) * nx_
                vty = velocity[1] - (velocity[0] * nx_ + velocity[1] * ny_ + velocity[2] * nz_) * ny_
                vtz = velocity[2] - (velocity[0] * nx_ + velocity[1] * ny_ + velocity[2] * nz_) * nz_
                vt_len = np.sqrt(vtx * vtx + vty * vty + vtz * vtz)
                ft_mag = mu * fn
                if ct > 0.0 and vt_len > 1.0e-14:
                    ft_damp = ct * vt_len
                    if ft_damp > ft_mag:
                        ft_damp = ft_mag
                    ft_scale = -ft_damp / vt_len
                    vtx *= ft_scale
                    vty *= ft_scale
                    vtz *= ft_scale
                elif vt_len > 1.0e-14:
                    ft_scale = -ft_mag / vt_len
                    vtx *= ft_scale
                    vty *= ft_scale
                    vtz *= ft_scale
                else:
                    vtx = 0.0; vty = 0.0; vtz = 0.0
                fx += vtx; fy += vty; fz += vtz
            points[i, 0] = x; points[i, 1] = y; points[i, 2] = z
            forces[i, 0] = fx; forces[i, 1] = fy; forces[i, 2] = fz
            normals[i, 0] = nx_; normals[i, 1] = ny_; normals[i, 2] = nz_
            phis[i] = phi; depths[i] = depth; pressures[i] = pressure; energies[i] = energy
            tangential_forces[i, 0] = vtx; tangential_forces[i, 1] = vty; tangential_forces[i, 2] = vtz

# -----------------------------------------------------------------------------
# CUDA kernels
# -----------------------------------------------------------------------------
if NUMBA_AVAILABLE:

    @cuda.jit(device=True)
    def _cr_w_dev(t, a):
        t2 = t * t; t3 = t2 * t
        if a == 0: return -0.5 * t + t2 - 0.5 * t3
        if a == 1: return 1.0 - 2.5 * t2 + 1.5 * t3
        if a == 2: return 0.5 * t + 2.0 * t2 - 1.5 * t3
        return -0.5 * t2 + 0.5 * t3

    @cuda.jit(device=True)
    def _cr_dw_dev(t, a):
        t2 = t * t
        if a == 0: return -0.5 + 2.0 * t - 1.5 * t2
        if a == 1: return -5.0 * t + 4.5 * t2
        if a == 2: return 0.5 + 4.0 * t - 4.5 * t2
        return -t + 1.5 * t2

    @cuda.jit(device=True)
    def _linear_sample_dev(values, origin, spacing, x, y, z):
        nx = values.shape[0]; ny = values.shape[1]; nz = values.shape[2]
        ux = (x - origin[0]) / spacing[0]
        uy = (y - origin[1]) / spacing[1]
        uz = (z - origin[2]) / spacing[2]
        ix = int(np.floor(ux)); iy = int(np.floor(uy)); iz = int(np.floor(uz))
        ax = ux - ix; ay = uy - iy; az = uz - iz
        if ix < 0: ix = 0; ax = 0.0
        elif ix > nx - 2: ix = nx - 2; ax = 1.0
        if iy < 0: iy = 0; ay = 0.0
        elif iy > ny - 2: iy = ny - 2; ay = 1.0
        if iz < 0: iz = 0; az = 0.0
        elif iz > nz - 2: iz = nz - 2; az = 1.0
        c000 = values[ix, iy, iz]; c100 = values[ix + 1, iy, iz]
        c010 = values[ix, iy + 1, iz]; c110 = values[ix + 1, iy + 1, iz]
        c001 = values[ix, iy, iz + 1]; c101 = values[ix + 1, iy, iz + 1]
        c011 = values[ix, iy + 1, iz + 1]; c111 = values[ix + 1, iy + 1, iz + 1]
        c00 = c000 * (1.0 - ax) + c100 * ax; c10 = c010 * (1.0 - ax) + c110 * ax
        c01 = c001 * (1.0 - ax) + c101 * ax; c11 = c011 * (1.0 - ax) + c111 * ax
        c0 = c00 * (1.0 - ay) + c10 * ay; c1 = c01 * (1.0 - ay) + c11 * ay
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
        if bx < 1: bx = 1; tx = 0.0
        elif bx > nx - 3: bx = nx - 3; tx = 1.0
        if by < 1: by = 1; ty = 0.0
        elif by > ny - 3: by = ny - 3; ty = 1.0
        if bz < 1: bz = 1; tz = 0.0
        elif bz > nz - 3: bz = nz - 3; tz = 1.0
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
                    w = wx * wy * wz; phi += w * val
                    gx += dwx * wy * wz * val; gy += wx * dwy * wz * val; gz += wx * wy * dwz * val
                    if val < local_min: local_min = val
                    if val > local_max: local_max = val
        if phi < local_min: phi = local_min
        elif phi > local_max: phi = local_max
        return phi, gx / spacing[0], gy / spacing[1], gz / spacing[2]

    CUDA_BLOCK_SIZE = 128

    @cuda.jit
    def _surface_forces_cuda_kernel(
        centroids_local, areas_local, values, origin, spacing,
        translation, velocity, kn, alpha, cn, mu, ct, method_flag,
        points, forces, normals, phis, depths, pressures, energies,
        block_force_x, block_force_y, block_force_z,
        block_torque_x, block_torque_y, block_torque_z,
        block_energy,
    ):
        shared_fx = cuda.shared.array(CUDA_BLOCK_SIZE, dtype=np.float64)
        shared_fy = cuda.shared.array(CUDA_BLOCK_SIZE, dtype=np.float64)
        shared_fz = cuda.shared.array(CUDA_BLOCK_SIZE, dtype=np.float64)
        shared_tx = cuda.shared.array(CUDA_BLOCK_SIZE, dtype=np.float64)
        shared_ty = cuda.shared.array(CUDA_BLOCK_SIZE, dtype=np.float64)
        shared_tz = cuda.shared.array(CUDA_BLOCK_SIZE, dtype=np.float64)
        shared_e = cuda.shared.array(CUDA_BLOCK_SIZE, dtype=np.float64)
        tid = cuda.threadIdx.x
        i = cuda.grid(1)
        s_fx = 0.0; s_fy = 0.0; s_fz = 0.0
        s_tx = 0.0; s_ty = 0.0; s_tz = 0.0
        s_e = 0.0
        if i < centroids_local.shape[0]:
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
            pressure = 0.0; energy = 0.0
            if depth > 0.0:
                pressure = kn * (depth ** alpha)
                vn = velocity[0] * nx_ + velocity[1] * ny_ + velocity[2] * nz_
                if vn < 0.0 and cn > 0.0:
                    pressure += cn * (-vn)
                energy = kn / (alpha + 1.0) * (depth ** (alpha + 1.0)) * areas_local[i]
            fn = pressure * areas_local[i]
            fx = fn * nx_; fy = fn * ny_; fz = fn * nz_
            vtx = 0.0; vty = 0.0; vtz = 0.0
            if mu > 0.0 and depth > 0.0:
                vn = velocity[0] * nx_ + velocity[1] * ny_ + velocity[2] * nz_
                vtx = velocity[0] - vn * nx_
                vty = velocity[1] - vn * ny_
                vtz = velocity[2] - vn * nz_
                vt_len = np.sqrt(vtx * vtx + vty * vty + vtz * vtz)
                ft_mag = mu * fn
                if ct > 0.0 and vt_len > 1.0e-14:
                    ft_damp = ct * vt_len
                    if ft_damp > ft_mag:
                        ft_damp = ft_mag
                    ft_scale = -ft_damp / vt_len
                    vtx *= ft_scale; vty *= ft_scale; vtz *= ft_scale
                elif vt_len > 1.0e-14:
                    ft_scale = -ft_mag / vt_len
                    vtx *= ft_scale; vty *= ft_scale; vtz *= ft_scale
                else:
                    vtx = 0.0; vty = 0.0; vtz = 0.0
                fx += vtx; fy += vty; fz += vtz
            points[i, 0] = x; points[i, 1] = y; points[i, 2] = z
            forces[i, 0] = fx; forces[i, 1] = fy; forces[i, 2] = fz
            normals[i, 0] = nx_; normals[i, 1] = ny_; normals[i, 2] = nz_
            phis[i] = phi; depths[i] = depth; pressures[i] = pressure; energies[i] = energy
            s_fx = fx; s_fy = fy; s_fz = fz
            rx = x - translation[0]; ry = y - translation[1]; rz = z - translation[2]
            s_tx = ry * fz - rz * fy
            s_ty = rz * fx - rx * fz
            s_tz = rx * fy - ry * fx
            s_e = energy
        shared_fx[tid] = s_fx; shared_fy[tid] = s_fy; shared_fz[tid] = s_fz
        shared_tx[tid] = s_tx; shared_ty[tid] = s_ty; shared_tz[tid] = s_tz
        shared_e[tid] = s_e
        cuda.syncthreads()
        stride = CUDA_BLOCK_SIZE // 2
        while stride > 0:
            if tid < stride:
                shared_fx[tid] += shared_fx[tid + stride]
                shared_fy[tid] += shared_fy[tid + stride]
                shared_fz[tid] += shared_fz[tid + stride]
                shared_tx[tid] += shared_tx[tid + stride]
                shared_ty[tid] += shared_ty[tid + stride]
                shared_tz[tid] += shared_tz[tid + stride]
                shared_e[tid] += shared_e[tid + stride]
            cuda.syncthreads()
            stride //= 2
        if tid == 0:
            bid = cuda.blockIdx.x
            block_force_x[bid] = shared_fx[0]
            block_force_y[bid] = shared_fy[0]
            block_force_z[bid] = shared_fz[0]
            block_torque_x[bid] = shared_tx[0]
            block_torque_y[bid] = shared_ty[0]
            block_torque_z[bid] = shared_tz[0]
            block_energy[bid] = shared_e[0]

    @cuda.jit
    def _reduce_blocks_kernel(block_vals, result, n_blocks):
        tid = cuda.threadIdx.x
        if tid < n_blocks:
            cuda.atomic.add(result, 0, block_vals[tid])

    @cuda.jit
    def _reduce_blocks_kernel3(block_x, block_y, block_z, result, n_blocks):
        tid = cuda.threadIdx.x
        if tid < n_blocks:
            cuda.atomic.add(result, 0, block_x[tid])
            cuda.atomic.add(result, 1, block_y[tid])
            cuda.atomic.add(result, 2, block_z[tid])


class SurfaceContactEvaluator:
    """Batched face-quadrature contact force evaluator for explicit dynamics.

    Supports translation-only or full 6-DOF (translation + rotation via quaternion).
    CUDA path uses shared-memory block reduction for force/torque/energy accumulation.
    Coulomb friction with viscous tangential damping is available.
    """

    def __init__(
        self,
        active_mesh: TriangleMesh,
        passive_grid: SDFGrid,
        method: str = "linear",
        backend: str = "auto",
        mu: float = 0.0,
        ct: float = 0.0,
    ) -> None:
        self.active_mesh = active_mesh
        self.grid = passive_grid
        self.method = "cubic" if method.lower().startswith("cub") else "linear"
        self.backend_request = backend.lower()
        self.mu = mu
        self.ct = ct

        fv = active_mesh.face_vertices()
        face_centroids = fv.mean(axis=1).astype(np.float64)
        face_areas = active_mesh.face_areas().astype(np.float64)
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

    def _init_cuda_arrays(self) -> None:
        n = self.centroids_local.shape[0]
        n_blocks = (n + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE
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
            "block_fx": cuda.device_array(n_blocks, dtype=np.float64),
            "block_fy": cuda.device_array(n_blocks, dtype=np.float64),
            "block_fz": cuda.device_array(n_blocks, dtype=np.float64),
            "block_tx": cuda.device_array(n_blocks, dtype=np.float64),
            "block_ty": cuda.device_array(n_blocks, dtype=np.float64),
            "block_tz": cuda.device_array(n_blocks, dtype=np.float64),
            "block_e": cuda.device_array(n_blocks, dtype=np.float64),
            "total_force": cuda.device_array(3, dtype=np.float64),
            "total_torque": cuda.device_array(3, dtype=np.float64),
            "total_energy": cuda.device_array(1, dtype=np.float64),
        }

    def evaluate(
        self,
        translation: np.ndarray,
        velocity: np.ndarray,
        kn: float,
        alpha: float = 1.0,
        cn: float = 0.0,
        torque_center: np.ndarray | None = None,
        quaternion: np.ndarray | None = None,
    ) -> SurfaceContactResult:
        translation = np.asarray(translation, dtype=np.float64).reshape(3)
        velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
        if torque_center is None:
            torque_center = np.zeros(3, dtype=np.float64)
        torque_center = np.asarray(torque_center, dtype=np.float64).reshape(3)

        use_rotation = quaternion is not None and np.linalg.norm(quaternion[1:4]) > 1e-10
        if use_rotation:
            q = _quat_normalize(np.asarray(quaternion, dtype=np.float64))
            com_local = torque_center - translation
            rotated = _rotate_point_batch_cpu(self.centroids_local, q, com_local)
            world_points = rotated + translation[None, :]
        else:
            world_points = self.centroids_local + translation[None, :]

        method_flag = 1 if self.method == "cubic" else 0
        backend = self.backend_used
        n = self.centroids_local.shape[0]

        if backend == "cuda":
            arr = self._cuda_arrays
            arr["translation"].copy_to_device(translation)
            arr["velocity"].copy_to_device(velocity)
            n_blocks = (n + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE
            _surface_forces_cuda_kernel[n_blocks, CUDA_BLOCK_SIZE](
                arr["centroids"], arr["areas"], arr["values"], arr["origin"], arr["spacing"],
                arr["translation"], arr["velocity"],
                float(kn), float(alpha), float(cn), float(self.mu), float(self.ct), method_flag,
                arr["points"], arr["forces"], arr["normals"],
                arr["phis"], arr["depths"], arr["pressures"], arr["energies"],
                arr["block_fx"], arr["block_fy"], arr["block_fz"],
                arr["block_tx"], arr["block_ty"], arr["block_tz"],
                arr["block_e"],
            )
            arr["total_force"].copy_to_device(np.zeros(3, dtype=np.float64))
            arr["total_torque"].copy_to_device(np.zeros(3, dtype=np.float64))
            arr["total_energy"].copy_to_device(np.zeros(1, dtype=np.float64))
            _reduce_blocks_kernel3[n_blocks, min(n_blocks, 32)](
                arr["block_fx"], arr["block_fy"], arr["block_fz"],
                arr["total_force"], np.int32(n_blocks),
            )
            _reduce_blocks_kernel3[n_blocks, min(n_blocks, 32)](
                arr["block_tx"], arr["block_ty"], arr["block_tz"],
                arr["total_torque"], np.int32(n_blocks),
            )
            _reduce_blocks_kernel[n_blocks, min(n_blocks, 32)](
                arr["block_e"], arr["total_energy"], np.int32(n_blocks),
            )
            points = arr["points"].copy_to_host()
            forces = arr["forces"].copy_to_host()
            normals = arr["normals"].copy_to_host()
            phis = arr["phis"].copy_to_host()
            depths = arr["depths"].copy_to_host()
            pressures = arr["pressures"].copy_to_host()
            energies = arr["energies"].copy_to_host()
            total_force = arr["total_force"].copy_to_host()
            total_torque = arr["total_torque"].copy_to_host()
            total_energy = float(arr["total_energy"].copy_to_host()[0])
        elif backend == "numba_cpu":
            points = np.empty((n, 3), dtype=np.float64)
            forces = np.empty((n, 3), dtype=np.float64)
            normals = np.empty((n, 3), dtype=np.float64)
            phis = np.empty(n, dtype=np.float64)
            depths = np.empty(n, dtype=np.float64)
            pressures = np.empty(n, dtype=np.float64)
            energies = np.empty(n, dtype=np.float64)
            tangential_forces = np.empty((n, 3), dtype=np.float64)
            _surface_forces_cpu_kernel(
                self.centroids_local, self.areas,
                np.asarray(self.grid.values, dtype=np.float32),
                np.asarray(self.grid.origin, dtype=np.float64),
                np.asarray(self.grid.spacing, dtype=np.float64),
                translation, velocity,
                float(kn), float(alpha), float(cn),
                float(self.mu), float(self.ct), method_flag,
                points, forces, normals, phis, depths, pressures, energies,
                tangential_forces,
            )
            total_force = np.sum(forces, axis=0)
            total_torque = np.sum(np.cross(points - torque_center[None, :], forces), axis=0)
            total_energy = float(np.sum(energies))
        else:
            points = world_points
            phi, grad = self.grid.sample(points, method=self.method, return_grad=True)
            phi = np.asarray(phi, dtype=np.float64)
            phis = phi.copy()
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
            fn = pressures * self.areas
            forces = fn[:, None] * normals
            if self.mu > 0.0 and np.any(active):
                active_idx = np.flatnonzero(active)
                vn_all = normals[active_idx] @ velocity
                vt = velocity[None, :] - vn_all[:, None] * normals[active_idx]
                vt_len = np.linalg.norm(vt, axis=1)
                ft_mag = self.mu * fn[active_idx]
                ft = np.zeros_like(vt)
                moving = vt_len > 1e-14
                if np.any(moving):
                    mi = active_idx[moving]
                    if self.ct > 0.0:
                        ft_damp = np.minimum(self.ct * vt_len[moving], ft_mag[moving])
                        ft[moving] = -ft_damp[:, None] * vt[moving] / vt_len[moving, None]
                    else:
                        ft[moving] = -ft_mag[moving, None] * vt[moving] / vt_len[moving, None]
                forces[active_idx] += ft
            energies = kn / (alpha + 1.0) * np.power(depths, alpha + 1.0) * self.areas
            total_force = np.sum(forces, axis=0)
            total_torque = np.sum(np.cross(points - torque_center[None, :], forces), axis=0)
            total_energy = float(np.sum(energies))

        active_mask = depths > 0.0
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
            elastic_energy=total_energy if backend != "cuda" else total_energy,
            contact_area=float(np.sum(self.areas[active_mask])),
            contact_count=int(np.sum(active_mask)),
            backend_used=backend,
            method=self.method,
        )
