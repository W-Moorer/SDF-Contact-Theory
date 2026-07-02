from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from sdf_contact.sdf import SDFGrid, mesh_to_sdf_grid
from sdf_contact.contact import compute_contact_forces, extract_contact_region, label_connected_components

from .force_evaluator import SurfaceContactEvaluator, SurfaceContactResult
from .scenarios import DynamicsCase


@dataclass
class DynamicsResult:
    case_name: str
    method: str
    backend_used: str
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    contact_force: np.ndarray
    contact_torque: np.ndarray
    contact_area: np.ndarray
    contact_count: np.ndarray
    max_depth: np.ndarray
    kinetic_energy: np.ndarray
    gravitational_energy: np.ndarray
    contact_energy: np.ndarray
    total_energy: np.ndarray
    events: Dict[str, float | None]
    snapshots: List[dict]
    settings: Dict

    def to_dict(self) -> Dict:
        return {
            "case_name": self.case_name,
            "method": self.method,
            "backend_used": self.backend_used,
            "events": self.events,
            "settings": self.settings,
            "samples": int(len(self.time)),
            "duration": float(self.time[-1]) if len(self.time) else 0.0,
            "max_contact_force_norm": float(np.max(np.linalg.norm(self.contact_force, axis=1))) if len(self.time) else 0.0,
            "max_contact_force_z": float(np.max(self.contact_force[:, 2])) if len(self.time) else 0.0,
            "max_depth": float(np.max(self.max_depth)) if len(self.time) else 0.0,
            "max_contact_area": float(np.max(self.contact_area)) if len(self.time) else 0.0,
            "energy_initial": float(self.total_energy[0]) if len(self.time) else 0.0,
            "energy_final": float(self.total_energy[-1]) if len(self.time) else 0.0,
            "energy_relative_drift": _energy_drift(self.total_energy),
            "snapshots": self.snapshots,
        }

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            time=self.time,
            position=self.position,
            velocity=self.velocity,
            contact_force=self.contact_force,
            contact_torque=self.contact_torque,
            contact_area=self.contact_area,
            contact_count=self.contact_count,
            max_depth=self.max_depth,
            kinetic_energy=self.kinetic_energy,
            gravitational_energy=self.gravitational_energy,
            contact_energy=self.contact_energy,
            total_energy=self.total_energy,
            method=self.method,
            backend_used=self.backend_used,
            case_name=self.case_name,
        )


def _energy_drift(total_energy: np.ndarray) -> float:
    if len(total_energy) == 0:
        return 0.0
    e0 = float(total_energy[0])
    denom = max(abs(e0), 1e-12)
    return float((total_energy[-1] - e0) / denom)


def _events(time: np.ndarray, contact_force: np.ndarray, max_depth: np.ndarray, contact_count: np.ndarray, force_threshold: float) -> Dict[str, float | None]:
    if len(time) == 0:
        return {"first_contact": None, "max_compression": None, "liftoff_after_contact": None}
    force_norm = np.linalg.norm(contact_force, axis=1)
    active = (contact_count > 0) | (force_norm > force_threshold) | (max_depth > 1e-10)
    idx = np.flatnonzero(active)
    first = int(idx[0]) if idx.size else None
    max_i = int(np.argmax(max_depth)) if np.max(max_depth) > 0 else None
    liftoff = None
    if first is not None:
        inactive_after = np.flatnonzero(~active[first + 1 :])
        if inactive_after.size:
            liftoff = int(first + 1 + inactive_after[0])
    return {
        "first_contact": float(time[first]) if first is not None else None,
        "max_compression": float(time[max_i]) if max_i is not None else None,
        "liftoff_after_contact": float(time[liftoff]) if liftoff is not None else None,
    }


def build_dynamic_sdf_grid(case: DynamicsCase, sdf_source: str, backend: str, resolution: int, device: str | None = "cuda:0") -> SDFGrid:
    sdf_source = sdf_source.lower()
    if sdf_source == "analytic":
        return SDFGrid.from_analytic(case.passive_sdf, case.bounds, resolution=resolution, name=f"{case.name}_dynamic_analytic_sdf")
    if sdf_source == "mesh":
        return mesh_to_sdf_grid(
            case.passive_mesh,
            case.bounds,
            resolution=resolution,
            backend=backend,
            device=device,
            analytic_sign_fn=case.passive_sdf,
            name=f"{case.name}_dynamic_mesh_sdf_{backend}",
        )
    raise ValueError("sdf_source must be 'analytic' or 'mesh'")


def _analytic_first_contact_time(case: DynamicsCase) -> float | None:
    """Analytic first-contact estimate for plane-like support cases.

    This is only a reference event time for clean free-fall validation.  It uses
    the minimum active local z and passive top z=0 convention.
    """
    g = -float(case.gravity[2])
    if g <= 0.0:
        return None
    z_min_local = float(case.active_mesh.bounds[0][2])
    clearance = float(case.initial_position[2] + z_min_local)  # plane/cube top z=0 in our dynamic cases
    if clearance <= 0.0:
        return 0.0
    return float(np.sqrt(2.0 * clearance / g))


def simulate_case(
    case: DynamicsCase,
    grid: SDFGrid,
    method: str = "linear",
    backend: str = "auto",
    kn: float = 8.0e4,
    alpha: float = 1.0,
    cn: float = 120.0,
    dt: float | None = None,
    steps: int | None = None,
    snapshot_stride: int = 30,
    exact_manifold_stride: int = 0,
    contact_max_depth: int = 3,
) -> DynamicsResult:
    """Run translation-only explicit dynamics with SDF contact forces.

    Integrator: symplectic Euler, v_{n+1}=v_n+dt*a_n, x_{n+1}=x_n+dt*v_{n+1}.
    The force evaluator is parallelized over active mesh face centroids and uses
    the same SDFGrid interpolation modes as the static contact manifold code.
    """
    dt = float(case.dt if dt is None else dt)
    steps = int(case.steps if steps is None else steps)
    method = "cubic" if method.lower().startswith("cub") else "linear"

    evaluator = SurfaceContactEvaluator(case.active_mesh, grid, method=method, backend=backend)
    pos = case.initial_position.astype(np.float64).copy()
    vel = case.initial_velocity.astype(np.float64).copy()
    mass = float(case.mass)
    g = case.gravity.astype(np.float64)
    com_local = np.asarray(case.local_com, dtype=np.float64)

    time = np.zeros(steps + 1, dtype=np.float64)
    position = np.zeros((steps + 1, 3), dtype=np.float64)
    velocity = np.zeros((steps + 1, 3), dtype=np.float64)
    contact_force = np.zeros((steps + 1, 3), dtype=np.float64)
    contact_torque = np.zeros((steps + 1, 3), dtype=np.float64)
    contact_area = np.zeros(steps + 1, dtype=np.float64)
    contact_count = np.zeros(steps + 1, dtype=np.int64)
    max_depth = np.zeros(steps + 1, dtype=np.float64)
    kinetic = np.zeros(steps + 1, dtype=np.float64)
    grav_e = np.zeros(steps + 1, dtype=np.float64)
    contact_e = np.zeros(steps + 1, dtype=np.float64)
    total_e = np.zeros(steps + 1, dtype=np.float64)
    snapshots: List[dict] = []

    def record(i: int, result: SurfaceContactResult) -> None:
        t = i * dt
        com_world = pos + com_local
        time[i] = t
        position[i] = pos
        velocity[i] = vel
        contact_force[i] = result.total_force
        contact_torque[i] = result.total_torque
        contact_area[i] = result.contact_area
        contact_count[i] = result.contact_count
        max_depth[i] = result.max_depth
        kinetic[i] = 0.5 * mass * float(np.dot(vel, vel))
        grav_e[i] = -mass * float(np.dot(g, com_world))
        contact_e[i] = result.elastic_energy
        total_e[i] = kinetic[i] + grav_e[i] + contact_e[i]
        if snapshot_stride > 0 and (i % snapshot_stride == 0 or i == steps or result.contact_count > 0 and len(snapshots) < 4):
            active = result.depths > 0.0
            # Limit payload size for HTML report.
            idx = np.flatnonzero(active)
            if idx.size > 120:
                idx = idx[np.linspace(0, idx.size - 1, 120).astype(int)]
            snapshots.append(
                {
                    "step": int(i),
                    "time": float(t),
                    "position": pos.tolist(),
                    "velocity": vel.tolist(),
                    "force": result.total_force.tolist(),
                    "force_norm": result.force_norm,
                    "contact_count": int(result.contact_count),
                    "contact_area": float(result.contact_area),
                    "max_depth": result.max_depth,
                    "contact_points": result.points[idx].tolist() if idx.size else [],
                    "contact_depths": result.depths[idx].tolist() if idx.size else [],
                }
            )

    # Initial force and state record.
    res = evaluator.evaluate(pos, vel, kn=kn, alpha=alpha, cn=cn, torque_center=pos + com_local)
    record(0, res)

    exact_snapshots: list[dict] = []
    for i in range(steps):
        force = res.total_force + mass * g
        acc = force / mass
        vel = vel + dt * acc
        pos = pos + dt * vel
        res = evaluator.evaluate(pos, vel, kn=kn, alpha=alpha, cn=cn, torque_center=pos + com_local)
        record(i + 1, res)

        if exact_manifold_stride > 0 and (i + 1) % exact_manifold_stride == 0 and res.contact_count > 0:
            moved_mesh = case.active_mesh.translated(pos)
            manifold = extract_contact_region(
                moved_mesh,
                grid,
                method=method,
                max_depth=contact_max_depth,
                min_edge_length=0.8 * float(np.max(grid.spacing)),
                candidate_epsilon=2.5 * float(np.max(grid.spacing)),
            )
            _, comps = label_connected_components(manifold, threshold=2.0 * float(np.max(grid.spacing)))
            exact_force = compute_contact_forces(manifold, kn=kn, alpha=alpha, center_of_mass=pos + com_local)
            exact_snapshots.append(
                {
                    "step": int(i + 1),
                    "time": float((i + 1) * dt),
                    "components": len(comps),
                    "area": float(np.sum(manifold.areas())),
                    "force": exact_force.total_force.tolist(),
                    "force_norm": float(np.linalg.norm(exact_force.total_force)),
                }
            )

    force_threshold = max(1.0e-8, 1.0e-4 * mass * np.linalg.norm(g))
    events = _events(time, contact_force, max_depth, contact_count, force_threshold=force_threshold)
    analytic_t = _analytic_first_contact_time(case)
    if analytic_t is not None:
        events["analytic_ballistic_first_contact"] = analytic_t
        if events["first_contact"] is not None:
            events["first_contact_error"] = float(events["first_contact"] - analytic_t)
        else:
            events["first_contact_error"] = None

    settings = {
        "dt": dt,
        "steps": steps,
        "mass": mass,
        "gravity": g.tolist(),
        "kn": float(kn),
        "alpha": float(alpha),
        "cn": float(cn),
        "sdf_grid_shape": list(grid.shape),
        "sdf_grid_spacing": grid.spacing.tolist(),
        "backend_request": backend,
        "backend_used": evaluator.backend_used,
        "method": method,
        "explicit_integrator": "symplectic_euler",
        "force_quadrature": "one face-centroid sample per active triangle",
        "exact_manifold_snapshots": exact_snapshots,
    }
    return DynamicsResult(
        case_name=case.name,
        method=method,
        backend_used=evaluator.backend_used,
        time=time,
        position=position,
        velocity=velocity,
        contact_force=contact_force,
        contact_torque=contact_torque,
        contact_area=contact_area,
        contact_count=contact_count,
        max_depth=max_depth,
        kinetic_energy=kinetic,
        gravitational_energy=grav_e,
        contact_energy=contact_e,
        total_energy=total_e,
        events=events,
        snapshots=snapshots,
        settings=settings,
    )


def write_result_json(path: str | Path, result: DynamicsResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_dynamics_result(npz_path: str | Path, json_path: str | Path) -> DynamicsResult:
    """Load a DynamicsResult from the saved NPZ time history and JSON metadata."""
    npz_path = Path(npz_path)
    json_path = Path(json_path)
    arrays = np.load(npz_path, allow_pickle=False)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    settings = dict(meta.get("settings", {}))
    snapshots = list(meta.get("snapshots", []))
    return DynamicsResult(
        case_name=str(meta.get("case_name", arrays.get("case_name", "dynamic_case"))),
        method=str(meta.get("method", arrays.get("method", "linear"))),
        backend_used=str(meta.get("backend_used", arrays.get("backend_used", "unknown"))),
        time=arrays["time"],
        position=arrays["position"],
        velocity=arrays["velocity"],
        contact_force=arrays["contact_force"],
        contact_torque=arrays["contact_torque"],
        contact_area=arrays["contact_area"],
        contact_count=arrays["contact_count"],
        max_depth=arrays["max_depth"],
        kinetic_energy=arrays["kinetic_energy"],
        gravitational_energy=arrays["gravitational_energy"],
        contact_energy=arrays["contact_energy"],
        total_energy=arrays["total_energy"],
        events=dict(meta.get("events", {})),
        snapshots=snapshots,
        settings=settings,
    )
