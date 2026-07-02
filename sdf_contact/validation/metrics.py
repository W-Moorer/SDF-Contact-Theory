from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from sdf_contact.sdf.grid import SDFGrid


def basic_sdf_metrics(grid: SDFGrid) -> Dict:
    vals = grid.values.astype(np.float64)
    h = float(np.max(grid.spacing))
    return {
        "name": grid.name,
        "shape": list(grid.shape),
        "spacing": grid.spacing.tolist(),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "near_zero_fraction_abs_phi_lt_2h": float(np.mean(np.abs(vals) < 2.0 * h)),
    }


def compare_grid_to_analytic(grid: SDFGrid, analytic_fn: Callable[[np.ndarray], np.ndarray], sample_count: int = 20000, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    mn, mx = grid.bounds
    pts = rng.uniform(mn, mx, size=(sample_count, 3))
    truth = analytic_fn(pts)
    out: Dict[str, Dict] = {}
    for method in ["linear", "cubic"]:
        pred = np.asarray(grid.sample(pts, method=method, return_grad=False), dtype=np.float64)
        err = pred - truth
        out[method] = {
            "rms": float(np.sqrt(np.mean(err * err))),
            "l1": float(np.mean(np.abs(err))),
            "linf": float(np.max(np.abs(err))),
            "sign_flip_fraction": float(np.mean(np.signbit(pred) != np.signbit(truth))),
        }
    return out


def eikonal_metrics(grid: SDFGrid, sample_count: int = 20000, seed: int = 1, band: float | None = None) -> Dict:
    rng = np.random.default_rng(seed)
    mn, mx = grid.bounds
    pts = rng.uniform(mn, mx, size=(sample_count, 3))
    if band is not None:
        # Resample a few times to focus near zero level set.
        keep = []
        for _ in range(8):
            cand = rng.uniform(mn, mx, size=(sample_count, 3))
            phi = np.asarray(grid.sample(cand, method="linear", return_grad=False), dtype=np.float64)
            keep.append(cand[np.abs(phi) < band])
            if sum(len(k) for k in keep) >= sample_count:
                break
        if keep:
            pts = np.concatenate(keep, axis=0)[:sample_count]
    out: Dict[str, Dict] = {}
    for method in ["linear", "cubic"]:
        phi, grad = grid.sample(pts, method=method, return_grad=True)
        norm = np.linalg.norm(np.asarray(grad, dtype=np.float64), axis=1)
        e = np.abs(norm - 1.0)
        out[method] = {
            "samples": int(len(e)),
            "mean_abs_norm_minus_1": float(np.mean(e)) if len(e) else 0.0,
            "rms_norm_minus_1": float(np.sqrt(np.mean(e * e))) if len(e) else 0.0,
            "linf_norm_minus_1": float(np.max(e)) if len(e) else 0.0,
        }
    return out


def contact_summary(contact, components) -> Dict:
    areas = contact.areas()
    return {
        "method": contact.method,
        "triangles": int(contact.count),
        "area": float(np.sum(areas)),
        "mean_depth_area_weighted": float(np.average(contact.depth, weights=np.maximum(areas, 1e-30))) if contact.count else 0.0,
        "max_depth": float(np.max(contact.depth)) if contact.count else 0.0,
        "components_count": int(len(components)),
        "components": components,
    }


def compare_contact_force(linear_contact, cubic_contact, linear_force, cubic_force) -> Dict:
    area_lin = float(np.sum(linear_contact.areas()))
    area_cub = float(np.sum(cubic_contact.areas()))
    f_lin = linear_force.total_force
    f_cub = cubic_force.total_force
    return {
        "area_linear": area_lin,
        "area_cubic": area_cub,
        "relative_area_difference_cubic_vs_linear": float(abs(area_cub - area_lin) / (abs(area_cub) + 1e-30)),
        "force_linear": f_lin.tolist(),
        "force_cubic": f_cub.tolist(),
        "relative_force_difference_cubic_vs_linear": float(np.linalg.norm(f_cub - f_lin) / (np.linalg.norm(f_cub) + 1e-30)),
    }
