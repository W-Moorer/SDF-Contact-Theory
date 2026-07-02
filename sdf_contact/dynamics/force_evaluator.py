from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sdf_contact.geometry.mesh import TriangleMesh
from sdf_contact.sdf.grid import SDFGrid
from sdf_contact.sdf.interpolation import normalize_grad

try:
    import numba as nb
    NUMBA_AVAILABLE = True
except Exception:
    nb = None
    NUMBA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except Exception:
    torch = None
    TORCH_AVAILABLE = False


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


# ── Quaternion helpers (pure numpy) ──────────────────────────────────────────
def _quat_mul(q, r):
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return np.array([
        qw*rw - qx*rx - qy*ry - qz*rz,
        qw*rx + qx*rw + qy*rz - qz*ry,
        qw*ry - qx*rz + qy*rw + qz*rx,
        qw*rz + qx*ry - qy*rx + qz*rw,
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
        -qx*wx - qy*wy - qz*wz,
         qw*wx + qy*wz - qz*wy,
         qw*wy - qx*wz + qz*wx,
         qw*wz + qx*wy - qy*wx,
    ], dtype=np.float64)

def _rotation_matrix_to_quat(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return _quat_normalize(np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s]))
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return _quat_normalize(np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s]))
    if R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return _quat_normalize(np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s]))
    s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
    return _quat_normalize(np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s]))

def _compute_inertia_tensor_tet(vertices, faces, mass):
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
                I[0,0] += w * (tet[s,1]*tet[t,1] + tet[s,2]*tet[t,2])
                I[1,1] += w * (tet[s,0]*tet[t,0] + tet[s,2]*tet[t,2])
                I[2,2] += w * (tet[s,0]*tet[t,0] + tet[s,1]*tet[t,1])
                I[0,1] -= w * tet[s,0]*tet[t,1]
                I[0,2] -= w * tet[s,0]*tet[t,2]
                I[1,2] -= w * tet[s,1]*tet[t,2]
    I[1,0] = I[0,1]; I[2,0] = I[0,2]; I[2,1] = I[1,2]
    return mass * I


# ── Numba CPU kernels ────────────────────────────────────────────────────────
if NUMBA_AVAILABLE:
    @nb.njit(cache=True)
    def _cr_w(t, a):
        t2 = t*t; t3 = t2*t
        if a==0: return -0.5*t + t2 - 0.5*t3
        if a==1: return 1.0 - 2.5*t2 + 1.5*t3
        if a==2: return 0.5*t + 2.0*t2 - 1.5*t3
        return -0.5*t2 + 0.5*t3

    @nb.njit(cache=True)
    def _cr_dw(t, a):
        t2 = t*t
        if a==0: return -0.5 + 2.0*t - 1.5*t2
        if a==1: return -5.0*t + 4.5*t2
        if a==2: return 0.5 + 4.0*t - 4.5*t2
        return -t + 1.5*t2

    @nb.njit(cache=True)
    def _linear_sample(values, origin, spacing, x, y, z):
        nx, ny, nz = values.shape
        ux = (x-origin[0])/spacing[0]; uy = (y-origin[1])/spacing[1]; uz = (z-origin[2])/spacing[2]
        ix = int(np.floor(ux)); iy = int(np.floor(uy)); iz = int(np.floor(uz))
        ax = ux-ix; ay = uy-iy; az = uz-iz
        if ix<0: ix=0; ax=0.0
        elif ix>nx-2: ix=nx-2; ax=1.0
        if iy<0: iy=0; ay=0.0
        elif iy>ny-2: iy=ny-2; ay=1.0
        if iz<0: iz=0; az=0.0
        elif iz>nz-2: iz=nz-2; az=1.0
        c000=float(values[ix,iy,iz]); c100=float(values[ix+1,iy,iz])
        c010=float(values[ix,iy+1,iz]); c110=float(values[ix+1,iy+1,iz])
        c001=float(values[ix,iy,iz+1]); c101=float(values[ix+1,iy,iz+1])
        c011=float(values[ix,iy+1,iz+1]); c111=float(values[ix+1,iy+1,iz+1])
        c00=c000*(1-ax)+c100*ax; c10=c010*(1-ax)+c110*ax
        c01=c001*(1-ax)+c101*ax; c11=c011*(1-ax)+c111*ax
        c0=c00*(1-ay)+c10*ay; c1=c01*(1-ay)+c11*ay
        phi=c0*(1-az)+c1*az
        gx=(((c100-c000)*(1-ay)+(c110-c010)*ay)*(1-az)+((c101-c001)*(1-ay)+(c111-c011)*ay)*az)/spacing[0]
        gy=(((c010-c000)*(1-ax)+(c110-c100)*ax)*(1-az)+((c011-c001)*(1-ax)+(c111-c101)*ax)*az)/spacing[1]
        gz=(c1-c0)/spacing[2]
        return phi, gx, gy, gz

    @nb.njit(cache=True)
    def _cubic_sample(values, origin, spacing, x, y, z):
        nx, ny, nz = values.shape
        ux=(x-origin[0])/spacing[0]; uy=(y-origin[1])/spacing[1]; uz=(z-origin[2])/spacing[2]
        bx=int(np.floor(ux)); by=int(np.floor(uy)); bz=int(np.floor(uz))
        tx=ux-bx; ty=uy-by; tz=uz-bz
        if bx<1: bx=1; tx=0.0
        elif bx>nx-3: bx=nx-3; tx=1.0
        if by<1: by=1; ty=0.0
        elif by>ny-3: by=ny-3; ty=1.0
        if bz<1: bz=1; tz=0.0
        elif bz>nz-3: bz=nz-3; tz=1.0
        phi=0.0; gx=0.0; gy=0.0; gz=0.0
        lmin=1e30; lmax=-1e30
        for a in range(4):
            ix=bx+a-1; wx=_cr_w(tx,a); dwx=_cr_dw(tx,a)
            for b in range(4):
                iy=by+b-1; wy=_cr_w(ty,b); dwy=_cr_dw(ty,b)
                for c in range(4):
                    iz=bz+c-1; wz=_cr_w(tz,c); dwz=_cr_dw(tz,c)
                    val=float(values[ix,iy,iz]); w=wx*wy*wz
                    phi+=w*val; gx+=dwx*wy*wz*val; gy+=wx*dwy*wz*val; gz+=wx*wy*dwz*val
                    if val<lmin: lmin=val
                    if val>lmax: lmax=val
        if phi<lmin: phi=lmin
        elif phi>lmax: phi=lmax
        return phi, gx/spacing[0], gy/spacing[1], gz/spacing[2]

    @nb.njit(cache=True)
    def _rotate_point_batch_cpu(pts, quat, com_local):
        n = pts.shape[0]; out = np.empty_like(pts)
        qw,qx,qy,qz = quat[0],quat[1],quat[2],quat[3]
        for i in range(n):
            dx=pts[i,0]-com_local[0]; dy=pts[i,1]-com_local[1]; dz=pts[i,2]-com_local[2]
            tw=-qx*dx-qy*dy-qz*dz; tx=qw*dx+qy*dz-qz*dy
            ty=qw*dy-qx*dz+qz*dx; tz=qw*dz+qx*dy-qy*dx
            rx=tw*(-qx)+tx*qw+ty*(-qz)-tz*(-qy)
            ry=tw*(-qy)-tx*(-qz)+ty*qw+tz*(-qx)
            rz=tw*(-qz)+tx*(-qy)-ty*(-qx)+tz*qw
            out[i,0]=rx+com_local[0]; out[i,1]=ry+com_local[1]; out[i,2]=rz+com_local[2]
        return out

    @nb.njit(parallel=True, cache=True)
    def _surface_forces_cpu_kernel(
        centroids_local, areas_local, values, origin, spacing,
        translation, velocity, kn, alpha, cn, mu, ct, method_flag,
        points, forces, normals, phis, depths, pressures, energies,
        tangential_forces,
    ):
        n = centroids_local.shape[0]
        for i in nb.prange(n):
            x=centroids_local[i,0]+translation[0]; y=centroids_local[i,1]+translation[1]; z=centroids_local[i,2]+translation[2]
            if method_flag==0: phi,gx,gy,gz=_linear_sample(values,origin,spacing,x,y,z)
            else: phi,gx,gy,gz=_cubic_sample(values,origin,spacing,x,y,z)
            gl=np.sqrt(gx*gx+gy*gy+gz*gz)
            if gl<1e-12: nx_=0.0; ny_=0.0; nz_=1.0
            else: nx_=gx/gl; ny_=gy/gl; nz_=gz/gl
            depth=-phi if phi<0.0 else 0.0; pressure=0.0; energy=0.0
            if depth>0.0:
                pressure=kn*(depth**alpha)
                vn=velocity[0]*nx_+velocity[1]*ny_+velocity[2]*nz_
                if vn<0.0 and cn>0.0: pressure+=cn*(-vn)
                energy=kn/(alpha+1.0)*(depth**(alpha+1.0))*areas_local[i]
            fn=pressure*areas_local[i]; fx=fn*nx_; fy=fn*ny_; fz=fn*nz_
            vtx=0.0; vty=0.0; vtz=0.0
            if mu>0.0 and depth>0.0:
                vn=velocity[0]*nx_+velocity[1]*ny_+velocity[2]*nz_
                vtx=velocity[0]-vn*nx_; vty=velocity[1]-vn*ny_; vtz=velocity[2]-vn*nz_
                vt_len=np.sqrt(vtx*vtx+vty*vty+vtz*vtz)
                ft_mag=mu*fn
                if ct>0.0 and vt_len>1e-14:
                    ft_damp=min(ct*vt_len, ft_mag); ft_scale=-ft_damp/vt_len
                    vtx*=ft_scale; vty*=ft_scale; vtz*=ft_scale
                elif vt_len>1e-14:
                    ft_scale=-ft_mag/vt_len; vtx*=ft_scale; vty*=ft_scale; vtz*=ft_scale
                else: vtx=0.0; vty=0.0; vtz=0.0
                fx+=vtx; fy+=vty; fz+=vtz
            points[i,0]=x; points[i,1]=y; points[i,2]=z
            forces[i,0]=fx; forces[i,1]=fy; forces[i,2]=fz
            normals[i,0]=nx_; normals[i,1]=ny_; normals[i,2]=nz_
            phis[i]=phi; depths[i]=depth; pressures[i]=pressure; energies[i]=energy
            tangential_forces[i,0]=vtx; tangential_forces[i,1]=vty; tangential_forces[i,2]=vtz


# ── PyTorch CUDA kernels ─────────────────────────────────────────────────────
if TORCH_AVAILABLE:

    def _torch_trilinear_sample(values_np, origin_np, spacing_np, points_np):
        """Vectorised trilinear SDF sampling on GPU via PyTorch."""
        dev = torch.device("cuda")
        values = torch.from_numpy(values_np).to(device=dev, dtype=torch.float32)
        origin = torch.from_numpy(origin_np).to(device=dev, dtype=torch.float64)
        spacing = torch.from_numpy(spacing_np).to(device=dev, dtype=torch.float64)
        pts = torch.from_numpy(points_np).to(device=dev, dtype=torch.float64)

        N = pts.shape[0]
        nx, ny, nz = values.shape
        flat = values.reshape(-1)

        ux = (pts[:, 0] - origin[0]) / spacing[0]
        uy = (pts[:, 1] - origin[1]) / spacing[1]
        uz = (pts[:, 2] - origin[2]) / spacing[2]

        ix = torch.floor(ux).long().clamp(0, nx - 2)
        iy = torch.floor(uy).long().clamp(0, ny - 2)
        iz = torch.floor(uz).long().clamp(0, nz - 2)
        ax = (ux - ix.double()).clamp(0.0, 1.0)
        ay = (uy - iy.double()).clamp(0.0, 1.0)
        az = (uz - iz.double()).clamp(0.0, 1.0)

        # Single batched gather: 8 corners
        offsets = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                                [0,0,1],[1,0,1],[0,1,1],[1,1,1]], device=dev, dtype=torch.long)
        ox = (ix.unsqueeze(1) + offsets[:, 0]).clamp(0, nx - 1)
        oy = (iy.unsqueeze(1) + offsets[:, 1]).clamp(0, ny - 1)
        oz = (iz.unsqueeze(1) + offsets[:, 2]).clamp(0, nz - 1)
        c = flat[ox * (ny * nz) + oy * nz + oz].double()  # (N, 8)

        # Trilinear interpolation: split into x-axis pairs
        c00 = c[:, 0]*(1-ax) + c[:, 1]*ax  # y0z0
        c10 = c[:, 2]*(1-ax) + c[:, 3]*ax  # y1z0
        c01 = c[:, 4]*(1-ax) + c[:, 5]*ax  # y0z1
        c11 = c[:, 6]*(1-ax) + c[:, 7]*ax  # y1z1
        c0 = c00*(1-ay) + c10*ay
        c1 = c01*(1-ay) + c11*ay
        phi = c0*(1-az) + c1*az

        gx = (((c[:, 1]-c[:, 0])*(1-ay) + (c[:, 3]-c[:, 2])*ay) * (1-az)
               + ((c[:, 5]-c[:, 4])*(1-ay) + (c[:, 7]-c[:, 6])*ay) * az) / spacing[0]
        gy = (((c[:, 2]-c[:, 0])*(1-ax) + (c[:, 3]-c[:, 1])*ax) * (1-az)
               + ((c[:, 6]-c[:, 4])*(1-ax) + (c[:, 7]-c[:, 5])*ax) * az) / spacing[1]
        gz = (c1 - c0) / spacing[2]

        return phi.cpu().numpy(), torch.stack([gx, gy, gz], dim=1).cpu().numpy()

    def _torch_tricubic_sample(values_np, origin_np, spacing_np, points_np):
        """Fully-vectorised Catmull-Rom tricubic SDF sampling on GPU.

        Key optimisation: the 4×4×4 = 64 gather offsets and weights are
        computed as a single batched operation (no Python loop), reducing
        CUDA kernel-launch overhead from 64 iterations to ~3-4.
        """
        dev = torch.device("cuda")
        values = torch.from_numpy(values_np).to(device=dev, dtype=torch.float32)
        origin = torch.from_numpy(origin_np).to(device=dev, dtype=torch.float64)
        spacing = torch.from_numpy(spacing_np).to(device=dev, dtype=torch.float64)
        pts = torch.from_numpy(points_np).to(device=dev, dtype=torch.float64)

        N = pts.shape[0]
        nx, ny, nz = values.shape
        flat = values.reshape(-1)

        ux = (pts[:, 0] - origin[0]) / spacing[0]
        uy = (pts[:, 1] - origin[1]) / spacing[1]
        uz = (pts[:, 2] - origin[2]) / spacing[2]

        bx = torch.floor(ux).long()
        by = torch.floor(uy).long()
        bz = torch.floor(uz).long()
        bx = bx.clamp(1, nx - 3)
        by = by.clamp(1, ny - 3)
        bz = bz.clamp(1, nz - 3)
        tx = (ux - bx.double()).clamp(0.0, 1.0)
        ty = (uy - by.double()).clamp(0.0, 1.0)
        tz = (uz - bz.double()).clamp(0.0, 1.0)

        # ── Catmull-Rom basis weights (all 4 at once, no branching) ──
        def cr_w_vec(t):
            t2 = t * t; t3 = t2 * t
            return torch.stack([
                -0.5*t + t2 - 0.5*t3,
                 1.0 - 2.5*t2 + 1.5*t3,
                 0.5*t + 2.0*t2 - 1.5*t3,
                -0.5*t2 + 0.5*t3,
            ], dim=1)  # (N, 4)

        def cr_dw_vec(t):
            t2 = t * t
            return torch.stack([
                -0.5 + 2.0*t - 1.5*t2,
                -5.0*t + 4.5*t2,
                 0.5 + 4.0*t - 4.5*t2,
                -t + 1.5*t2,
            ], dim=1)  # (N, 4)

        wx = cr_w_vec(tx)   # (N, 4)
        wy = cr_w_vec(ty)   # (N, 4)
        wz = cr_w_vec(tz)   # (N, 4)
        dwx = cr_dw_vec(tx)  # (N, 4)
        dwy = cr_dw_vec(ty)  # (N, 4)
        dwz = cr_dw_vec(tz)  # (N, 4)

        # ── Build all 64 integer offsets: (bx+a-1, by+b-1, bz+c-1) ──
        # offsets_x[a,b,c] = bx + a - 1, etc.
        a_idx = torch.arange(4, device=dev, dtype=torch.long)  # [0,1,2,3]
        b_idx = a_idx
        c_idx = a_idx

        # Outer-product index grids: shape (4,4,4) → flatten to 64
        # offset_x[i] = bx + a_vals[i] - 1
        a_vals, b_vals, c_vals = torch.meshgrid(a_idx, b_idx, c_idx, indexing='ij')
        a_flat = a_vals.reshape(-1)  # (64,)
        b_flat = b_vals.reshape(-1)
        c_flat = c_vals.reshape(-1)

        # Compute all 64 index offsets per point: (N, 64)
        ox = (bx.unsqueeze(1) + a_flat.unsqueeze(0) - 1).clamp(0, nx - 1)
        oy = (by.unsqueeze(1) + b_flat.unsqueeze(0) - 1).clamp(0, ny - 1)
        oz = (bz.unsqueeze(1) + c_flat.unsqueeze(0) - 1).clamp(0, nz - 1)
        flat_idx = ox * (ny * nz) + oy * nz + oz  # (N, 64)

        # Single batched gather: (N, 64)
        vals_64 = flat[flat_idx].double()  # (N, 64)

        # ── Compute all 64 tensor-product weights at once ──
        # wx[:, a] * wy[:, b] * wz[:, c] for all 64 (a,b,c) combos
        # Reshape wx/wy/wz to (N, 4, 1, 1), (N, 1, 4, 1), (N, 1, 1, 4)
        # then broadcast multiply → (N, 4, 4, 4) → reshape to (N, 64)
        w_3d = wx.unsqueeze(2).unsqueeze(3) * wy.unsqueeze(1).unsqueeze(3) * wz.unsqueeze(1).unsqueeze(2)  # (N,4,4,4)
        w_64 = w_3d.reshape(N, 64)

        # Value weighted sum: phi = Σ w_i * val_i
        phi = (w_64 * vals_64).sum(dim=1)

        # Gradient: gx = Σ (dwx_a * wy_b * wz_c / spacing_x) * val_i
        dwx_3d = dwx.unsqueeze(2).unsqueeze(3) * wy.unsqueeze(1).unsqueeze(3) * wz.unsqueeze(1).unsqueeze(2)
        gx = (dwx_3d.reshape(N, 64) * vals_64).sum(dim=1) / spacing[0]

        dwy_3d = wx.unsqueeze(2).unsqueeze(3) * dwy.unsqueeze(1).unsqueeze(3) * wz.unsqueeze(1).unsqueeze(2)
        gy = (dwy_3d.reshape(N, 64) * vals_64).sum(dim=1) / spacing[1]

        dwz_3d = wx.unsqueeze(2).unsqueeze(3) * wy.unsqueeze(1).unsqueeze(3) * dwz.unsqueeze(1).unsqueeze(2)
        gz = (dwz_3d.reshape(N, 64) * vals_64).sum(dim=1) / spacing[2]

        # Clamp phi to local min/max (single reduction)
        local_min = vals_64.min(dim=1).values
        local_max = vals_64.max(dim=1).values
        phi = torch.clamp(phi, min=local_min, max=local_max)

        return phi.cpu().numpy(), torch.stack([gx, gy, gz], dim=1).cpu().numpy()

    def _torch_rotate_points(pts_np, quat_np, com_local_np):
        dev = torch.device("cuda")
        pts = torch.from_numpy(pts_np).to(device=dev, dtype=torch.float64)
        q = torch.from_numpy(quat_np).to(device=dev, dtype=torch.float64)
        com = torch.from_numpy(com_local_np).to(device=dev, dtype=torch.float64)

        d = pts - com.unsqueeze(0)
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        tw = -qx*d[:, 0] - qy*d[:, 1] - qz*d[:, 2]
        tx_ = qw*d[:, 0] + qy*d[:, 2] - qz*d[:, 1]
        ty_ = qw*d[:, 1] - qx*d[:, 2] + qz*d[:, 0]
        tz_ = qw*d[:, 2] + qx*d[:, 1] - qy*d[:, 0]

        rx = tw*(-qx) + tx_*qw + ty_*(-qz) - tz_*(-qy)
        ry = tw*(-qy) - tx_*(-qz) + ty_*qw + tz_*(-qx)
        rz = tw*(-qz) + tx_*(-qy) - ty_*(-qx) + tz_*qw

        out = torch.stack([rx, ry, rz], dim=1) + com.unsqueeze(0)
        return out.cpu().numpy()

    def _torch_contact_forces(
        centroids_local_np, areas_local_np, values_np, origin_np, spacing_np,
        translation_np, velocity_np, kn, alpha, cn, mu, ct, method_flag,
    ):
        """Vectorised contact force evaluation on GPU via PyTorch.

        Returns (points, forces, normals, phis, depths, pressures, energies,
                 total_force, total_torque, contact_area, contact_count).
        """
        dev = torch.device("cuda")

        centroids = torch.from_numpy(centroids_local_np).to(device=dev, dtype=torch.float64)
        areas = torch.from_numpy(areas_local_np).to(device=dev, dtype=torch.float64)
        translation = torch.from_numpy(translation_np).to(device=dev, dtype=torch.float64)
        velocity = torch.from_numpy(velocity_np).to(device=dev, dtype=torch.float64)

        points = centroids + translation.unsqueeze(0)

        if method_flag == 0:
            phi_np, grad_np = _torch_trilinear_sample(values_np, origin_np, spacing_np, points.cpu().numpy())
        else:
            phi_np, grad_np = _torch_tricubic_sample(values_np, origin_np, spacing_np, points.cpu().numpy())

        phi = torch.from_numpy(phi_np).to(device=dev, dtype=torch.float64)
        grad = torch.from_numpy(grad_np).to(device=dev, dtype=torch.float64)

        gl = torch.norm(grad, dim=1)
        normals = torch.zeros_like(grad)
        mask_valid = gl > 1e-12
        normals[mask_valid] = grad[mask_valid] / gl[mask_valid, None]
        normals[~mask_valid] = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=torch.float64)

        depths = torch.clamp(-phi, min=0.0)
        pressures = torch.zeros_like(phi)
        active = depths > 0.0

        if active.any():
            pressures[active] = kn * depths[active].pow(alpha)
            vn = (normals[active] * velocity.unsqueeze(0)).sum(dim=1)
            approaching = vn < 0.0
            if approaching.any() and cn > 0.0:
                active_idx = active.nonzero(as_tuple=True)[0]
                app_in_active = approaching.nonzero(as_tuple=True)[0]
                global_app = active_idx[app_in_active]
                pressures[global_app] += cn * (-vn[app_in_active])

        fn = pressures * areas
        forces = fn.unsqueeze(1) * normals

        if mu > 0.0 and active.any():
            active_idx = active.nonzero(as_tuple=True)[0]
            vn_all = (normals[active_idx] * velocity.unsqueeze(0)).sum(dim=1)
            vt = velocity.unsqueeze(0) - vn_all.unsqueeze(1) * normals[active_idx]
            vt_len = torch.norm(vt, dim=1)
            ft_mag = mu * fn[active_idx]
            moving = vt_len > 1e-14
            ft = torch.zeros_like(vt)
            if moving.any():
                mi = active_idx[moving]
                if ct > 0.0:
                    ft_damp = torch.minimum(ct * vt_len[moving], ft_mag[moving])
                    ft[moving] = -ft_damp.unsqueeze(1) * vt[moving] / vt_len[moving].unsqueeze(1)
                else:
                    ft[moving] = -ft_mag[moving].unsqueeze(1) * vt[moving] / vt_len[moving].unsqueeze(1)
            forces[active_idx] += ft

        energies = kn / (alpha + 1.0) * depths.pow(alpha + 1.0) * areas
        total_force = forces.sum(dim=0)
        total_torque = torch.cross(points - translation.unsqueeze(0), forces, dim=1).sum(dim=0)

        return (
            points.cpu().numpy(),
            forces.cpu().numpy(),
            normals.cpu().numpy(),
            phi.cpu().numpy(),
            depths.cpu().numpy(),
            pressures.cpu().numpy(),
            energies.cpu().numpy(),
            total_force.cpu().numpy(),
            total_torque.cpu().numpy(),
            float(depths.gt(0.0).sum().item()),
        )


class SurfaceContactEvaluator:
    """Batched face-quadrature contact force evaluator for explicit dynamics.

    Supports translation-only or full 6-DOF (translation + rotation via quaternion).
    Backends: numpy (vectorised), numba_cpu (parallel JIT), torch (GPU).
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
        self._values_np = np.asarray(self.grid.values, dtype=np.float32)
        self._origin_np = np.asarray(self.grid.origin, dtype=np.float64)
        self._spacing_np = np.asarray(self.grid.spacing, dtype=np.float64)

    @property
    def backend_used(self) -> str:
        req = self.backend_request
        if req == "torch" and TORCH_AVAILABLE:
            return "torch"
        if req == "auto" and TORCH_AVAILABLE:
            return "torch"
        if req in {"auto", "numba", "numba_cpu"} and NUMBA_AVAILABLE:
            return "numba_cpu"
        if req == "torch" and not TORCH_AVAILABLE:
            raise RuntimeError("torch backend requested but PyTorch CUDA is not available.")
        return "numpy"

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

        method_flag = 1 if self.method == "cubic" else 0
        backend = self.backend_used
        n = self.centroids_local.shape[0]

        if backend == "torch":
            if use_rotation:
                if TORCH_AVAILABLE:
                    rotated = _torch_rotate_points(self.centroids_local, q, com_local)
                else:
                    rotated = _rotate_point_batch_cpu(self.centroids_local, q, com_local)
                world_pts_for_force = rotated + translation[None, :]
            else:
                world_pts_for_force = self.centroids_local + translation[None, :]

            points, forces, normals, phis, depths, pressures, energies, total_force, total_torque, contact_count = \
                _torch_contact_forces(
                    self.centroids_local, self.areas,
                    self._values_np, self._origin_np, self._spacing_np,
                    translation, velocity, kn, alpha, cn, self.mu, self.ct, method_flag,
                )
            total_energy = float(np.sum(energies))

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
                self._values_np, self._origin_np, self._spacing_np,
                translation, velocity,
                float(kn), float(alpha), float(cn),
                float(self.mu), float(self.ct), method_flag,
                points, forces, normals, phis, depths, pressures, energies,
                tangential_forces,
            )
            total_force = np.sum(forces, axis=0)
            total_torque = np.sum(np.cross(points - torque_center[None, :], forces), axis=0)
            total_energy = float(np.sum(energies))
            contact_count = int(np.sum(depths > 0.0))

        else:  # numpy
            if use_rotation:
                rotated = _rotate_point_batch_cpu(self.centroids_local, q, com_local)
                points = rotated + translation[None, :]
            else:
                points = self.centroids_local + translation[None, :]

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
            contact_count = int(np.sum(active))

        active_mask = depths > 0.0
        return SurfaceContactResult(
            points=points, forces=forces, normals=normals,
            phis=phis if 'phis' in dir() else phis if backend != "torch" else phis,
            depths=depths, pressures=pressures, areas=self.areas.copy(),
            total_force=total_force, total_torque=total_torque,
            elastic_energy=total_energy,
            contact_area=float(np.sum(self.areas[active_mask])),
            contact_count=contact_count,
            backend_used=backend, method=self.method,
        )
