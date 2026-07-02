from __future__ import annotations

from typing import Tuple

import numpy as np


def _get_xp(arr):
    try:
        import cupy as cp
        return cp.get_array_module(arr)
    except Exception:
        return np


def _to_xp(points, xp):
    if xp is np:
        return np.asarray(points, dtype=np.float64)
    return xp.asarray(points, dtype=xp.float32)


def _catmull_rom_weights(t):
    xp = _get_xp(t)
    t2 = t * t
    t3 = t2 * t
    w0 = -0.5 * t + t2 - 0.5 * t3
    w1 = 1.0 - 2.5 * t2 + 1.5 * t3
    w2 = 0.5 * t + 2.0 * t2 - 1.5 * t3
    w3 = -0.5 * t2 + 0.5 * t3
    return [w0, w1, w2, w3]


def _catmull_rom_dweights(t):
    xp = _get_xp(t)
    t2 = t * t
    dw0 = -0.5 + 2.0 * t - 1.5 * t2
    dw1 = -5.0 * t + 4.5 * t2
    dw2 = 0.5 + 4.0 * t - 4.5 * t2
    dw3 = -1.0 * t + 1.5 * t2
    return [dw0, dw1, dw2, dw3]


def sample_trilinear(values, origin, spacing, points, return_grad: bool = True):
    """Vectorized trilinear SDF sampling.

    values are indexed as values[ix, iy, iz].
    """
    xp = _get_xp(values)
    values = xp.asarray(values)
    pts = _to_xp(points, xp)
    origin = xp.asarray(origin, dtype=pts.dtype)
    spacing = xp.asarray(spacing, dtype=pts.dtype)
    shape = values.shape

    u = (pts - origin[None, :]) / spacing[None, :]
    i0 = xp.floor(u).astype(xp.int64)
    frac = u - i0.astype(pts.dtype)
    for ax in range(3):
        i0[:, ax] = xp.clip(i0[:, ax], 0, shape[ax] - 2)
        frac[:, ax] = xp.clip(frac[:, ax], 0.0, 1.0)
    ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]
    ax, ay, az = frac[:, 0], frac[:, 1], frac[:, 2]

    c000 = values[ix, iy, iz]
    c100 = values[ix + 1, iy, iz]
    c010 = values[ix, iy + 1, iz]
    c110 = values[ix + 1, iy + 1, iz]
    c001 = values[ix, iy, iz + 1]
    c101 = values[ix + 1, iy, iz + 1]
    c011 = values[ix, iy + 1, iz + 1]
    c111 = values[ix + 1, iy + 1, iz + 1]

    c00 = c000 * (1 - ax) + c100 * ax
    c10 = c010 * (1 - ax) + c110 * ax
    c01 = c001 * (1 - ax) + c101 * ax
    c11 = c011 * (1 - ax) + c111 * ax
    c0 = c00 * (1 - ay) + c10 * ay
    c1 = c01 * (1 - ay) + c11 * ay
    phi = c0 * (1 - az) + c1 * az

    if not return_grad:
        return phi

    dphidx = (
        ((c100 - c000) * (1 - ay) + (c110 - c010) * ay) * (1 - az)
        + ((c101 - c001) * (1 - ay) + (c111 - c011) * ay) * az
    ) / spacing[0]
    dphidy = (
        ((c010 - c000) * (1 - ax) + (c110 - c100) * ax) * (1 - az)
        + ((c011 - c001) * (1 - ax) + (c111 - c101) * ax) * az
    ) / spacing[1]
    dphidz = (c1 - c0) / spacing[2]
    grad = xp.stack([dphidx, dphidy, dphidz], axis=1)
    return phi, grad


def sample_tricubic(values, origin, spacing, points, return_grad: bool = True, clamp_value: bool = True):
    """Separable Catmull-Rom tricubic convolution with analytic derivative."""
    xp = _get_xp(values)
    values = xp.asarray(values)
    pts = _to_xp(points, xp)
    origin = xp.asarray(origin, dtype=pts.dtype)
    spacing = xp.asarray(spacing, dtype=pts.dtype)
    shape = values.shape

    u = (pts - origin[None, :]) / spacing[None, :]
    base = xp.floor(u).astype(xp.int64)
    t = u - base.astype(pts.dtype)
    # Need i-1..i+2 available.
    for ax in range(3):
        base[:, ax] = xp.clip(base[:, ax], 1, shape[ax] - 3)
        t[:, ax] = xp.clip(t[:, ax], 0.0, 1.0)

    wx = _catmull_rom_weights(t[:, 0])
    wy = _catmull_rom_weights(t[:, 1])
    wz = _catmull_rom_weights(t[:, 2])
    dwx = _catmull_rom_dweights(t[:, 0])
    dwy = _catmull_rom_dweights(t[:, 1])
    dwz = _catmull_rom_dweights(t[:, 2])

    bx, by, bz = base[:, 0], base[:, 1], base[:, 2]
    phi = xp.zeros(pts.shape[0], dtype=values.dtype)
    gx = xp.zeros_like(phi)
    gy = xp.zeros_like(phi)
    gz = xp.zeros_like(phi)
    if clamp_value:
        local_min = xp.full_like(phi, xp.inf)
        local_max = xp.full_like(phi, -xp.inf)
    for a in range(4):
        ix = bx + (a - 1)
        for b in range(4):
            iy = by + (b - 1)
            for c in range(4):
                iz = bz + (c - 1)
                vals = values[ix, iy, iz]
                w = wx[a] * wy[b] * wz[c]
                phi = phi + w * vals
                if return_grad:
                    gx = gx + dwx[a] * wy[b] * wz[c] * vals
                    gy = gy + wx[a] * dwy[b] * wz[c] * vals
                    gz = gz + wx[a] * wy[b] * dwz[c] * vals
                if clamp_value:
                    local_min = xp.minimum(local_min, vals)
                    local_max = xp.maximum(local_max, vals)
    if clamp_value:
        phi = xp.clip(phi, local_min, local_max)
    if not return_grad:
        return phi
    grad = xp.stack([gx / spacing[0], gy / spacing[1], gz / spacing[2]], axis=1)
    return phi, grad


def normalize_grad(grad, eps: float = 1e-12):
    xp = _get_xp(grad)
    n = xp.linalg.norm(grad, axis=1, keepdims=True)
    return grad / xp.maximum(n, eps)
