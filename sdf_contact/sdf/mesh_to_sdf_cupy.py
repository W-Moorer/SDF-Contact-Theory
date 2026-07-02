from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from sdf_contact.geometry.mesh import TriangleMesh
from .grid import SDFGrid


def _kernel_source() -> str:
    return r'''
extern "C" __global__
void point_triangle_unsigned_distance(
    const float* __restrict__ points,
    const float* __restrict__ tris,
    const int n_points,
    const int n_tris,
    float* __restrict__ out)
{
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= n_points) return;
    float px = points[3*ip+0];
    float py = points[3*ip+1];
    float pz = points[3*ip+2];
    float best = 3.402823e38f;

    for (int it=0; it<n_tris; ++it) {
        const float* t = tris + 9*it;
        float ax=t[0], ay=t[1], az=t[2];
        float bx=t[3], by=t[4], bz=t[5];
        float cx=t[6], cy=t[7], cz=t[8];
        float abx=bx-ax, aby=by-ay, abz=bz-az;
        float acx=cx-ax, acy=cy-ay, acz=cz-az;
        float apx=px-ax, apy=py-ay, apz=pz-az;
        float d1=abx*apx+aby*apy+abz*apz;
        float d2=acx*apx+acy*apy+acz*apz;
        float qx,qy,qz;
        if (d1 <= 0.0f && d2 <= 0.0f) { qx=ax; qy=ay; qz=az; }
        else {
            float bpx=px-bx, bpy=py-by, bpz=pz-bz;
            float d3=abx*bpx+aby*bpy+abz*bpz;
            float d4=acx*bpx+acy*bpy+acz*bpz;
            if (d3 >= 0.0f && d4 <= d3) { qx=bx; qy=by; qz=bz; }
            else {
                float vc=d1*d4-d3*d2;
                if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                    float v=d1/(d1-d3); qx=ax+v*abx; qy=ay+v*aby; qz=az+v*abz;
                } else {
                    float cpx=px-cx, cpy=py-cy, cpz=pz-cz;
                    float d5=abx*cpx+aby*cpy+abz*cpz;
                    float d6=acx*cpx+acy*cpy+acz*cpz;
                    if (d6 >= 0.0f && d5 <= d6) { qx=cx; qy=cy; qz=cz; }
                    else {
                        float vb=d5*d2-d1*d6;
                        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                            float w=d2/(d2-d6); qx=ax+w*acx; qy=ay+w*acy; qz=az+w*acz;
                        } else {
                            float va=d3*d6-d5*d4;
                            if (va <= 0.0f && (d4-d3) >= 0.0f && (d5-d6) >= 0.0f) {
                                float w=(d4-d3)/((d4-d3)+(d5-d6));
                                qx=bx+w*(cx-bx); qy=by+w*(cy-by); qz=bz+w*(cz-bz);
                            } else {
                                float denom=1.0f/(va+vb+vc);
                                float v=vb*denom; float w=vc*denom;
                                qx=ax+abx*v+acx*w; qy=ay+aby*v+acy*w; qz=az+abz*v+acz*w;
                            }
                        }
                    }
                }
            }
        }
        float dx=px-qx, dy=py-qy, dz=pz-qz;
        float d2p=dx*dx+dy*dy+dz*dz;
        if (d2p < best) best = d2p;
    }
    out[ip] = sqrtf(best);
}
'''


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


def mesh_to_sdf_cupy(
    mesh: TriangleMesh,
    bounds,
    resolution: int | tuple[int, int, int] = 64,
    analytic_sign_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    chunk_size: int = 65536,
    name: str | None = None,
) -> SDFGrid:
    """CuPy RawKernel brute-force Mesh→SDF backend.

    It computes unsigned distance on the GPU. Sign is supplied by analytic_sign_fn.
    For closed arbitrary meshes without analytic sign, use Warp or CPU parity.
    """
    try:
        import cupy as cp
    except Exception as exc:
        raise RuntimeError("cupy is not installed") from exc
    if analytic_sign_fn is None:
        raise ValueError("CuPy backend currently requires analytic_sign_fn for robust sign assignment")

    pts, origin, spacing, shape = _grid_points(bounds, resolution)
    tris = np.asarray(mesh.face_vertices(), dtype=np.float32).reshape(-1, 9)
    tris_gpu = cp.asarray(tris.reshape(-1))
    kernel = cp.RawKernel(_kernel_source(), "point_triangle_unsigned_distance")
    out = np.empty(len(pts), dtype=np.float32)
    threads = 128
    for start in range(0, len(pts), chunk_size):
        end = min(start + chunk_size, len(pts))
        pchunk = pts[start:end].astype(np.float32)
        p_gpu = cp.asarray(pchunk.reshape(-1))
        d_gpu = cp.empty(end - start, dtype=cp.float32)
        blocks = ((end - start + threads - 1) // threads,)
        kernel(blocks, (threads,), (p_gpu, tris_gpu, end - start, tris.shape[0], d_gpu))
        dist = cp.asnumpy(d_gpu)
        sign = np.where(analytic_sign_fn(pchunk.astype(np.float64)) < 0.0, -1.0, 1.0).astype(np.float32)
        out[start:end] = dist * sign
    return SDFGrid(out.reshape(shape), origin, spacing, name=name or f"{mesh.name}_mesh_sdf_cupy")
