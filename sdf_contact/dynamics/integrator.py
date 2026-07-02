from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from sdf_contact.sdf import SDFGrid, mesh_to_sdf_grid
from sdf_contact.contact import compute_contact_forces, extract_contact_region, label_connected_components

from .force_evaluator import SurfaceContactEvaluator, SurfaceContactResult
from .quaternion import (
    quat_conjugate,
    quat_derivative,
    quat_identity,
    quat_mul,
    quat_normalize,
    quat_rotate_vector,
    quat_to_rotation_matrix,
    rotation_matrix_to_quat,
    compute_inertia_tensor_tet as _compute_inertia_tensor_tet,
)
from .scenarios import DynamicsCase


@dataclass
class DynamicsResult:
    case_name: str
    method: str
    backend_used: str
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None
    contact_force: np.ndarray | None = None
    contact_torque: np.ndarray | None = None
    contact_area: np.ndarray | None = None
    contact_count: np.ndarray | None = None
    max_depth: np.ndarray | None = None
    kinetic_energy: np.ndarray | None = None
    rotational_energy: np.ndarray | None = None
    gravitational_energy: np.ndarray | None = None
    contact_energy: np.ndarray | None = None
    total_energy: np.ndarray | None = None
    events: Dict[str, float | None] = field(default_factory=dict)
    snapshots: List[dict] = field(default_factory=list)
    settings: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = {
            "case_name": self.case_name,
            "method": self.method,
            "backend_used": self.backend_used,
            "events": self.events,
            "settings": self.settings,
            "samples": int(len(self.time)),
            "duration": float(self.time[-1]) if len(self.time) else 0.0,
            "max_contact_force_norm": float(np.max(np.linalg.norm(self.contact_force, axis=1))) if len(self.time) and self.contact_force is not None else 0.0,
            "max_contact_force_z": float(np.max(self.contact_force[:, 2])) if len(self.time) and self.contact_force is not None else 0.0,
            "max_depth": float(np.max(self.max_depth)) if len(self.time) and self.max_depth is not None else 0.0,
            "max_contact_area": float(np.max(self.contact_area)) if len(self.time) and self.contact_area is not None else 0.0,
            "energy_initial": float(self.total_energy[0]) if len(self.time) and self.total_energy is not None else 0.0,
            "energy_final": float(self.total_energy[-1]) if len(self.time) and self.total_energy is not None else 0.0,
            "energy_relative_drift": _energy_drift(self.total_energy) if self.total_energy is not None else 0.0,
            "snapshots": self.snapshots,
        }
        return d

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            time=self.time, position=self.position, velocity=self.velocity,
            contact_force=self.contact_force, contact_torque=self.contact_torque,
            contact_area=self.contact_area, contact_count=self.contact_count,
            max_depth=self.max_depth, kinetic_energy=self.kinetic_energy,
            gravitational_energy=self.gravitational_energy,
            contact_energy=self.contact_energy, total_energy=self.total_energy,
            method=self.method, backend_used=self.backend_used, case_name=self.case_name,
        )
        if self.orientation is not None:
            arrays["orientation"] = self.orientation
        if self.angular_velocity is not None:
            arrays["angular_velocity"] = self.angular_velocity
        if self.rotational_energy is not None:
            arrays["rotational_energy"] = self.rotational_energy
        np.savez_compressed(path, **arrays)


def _energy_drift(total_energy: np.ndarray | None) -> float:
    if total_energy is None or len(total_energy) == 0:
        return 0.0
    e0 = float(total_energy[0])
    denom = max(abs(e0), 1e-12)
    return float((total_energy[-1] - e0) / denom)


def _events(time, contact_force, max_depth, contact_count, force_threshold):
    if len(time) == 0:
        return {"first_contact": None, "max_compression": None, "liftoff_after_contact": None}
    force_norm = np.linalg.norm(contact_force, axis=1)
    active = (contact_count > 0) | (force_norm > force_threshold) | (max_depth > 1e-10)
    idx = np.flatnonzero(active)
    first = int(idx[0]) if idx.size else None
    max_i = int(np.argmax(max_depth)) if np.max(max_depth) > 0 else None
    liftoff = None
    if first is not None:
        inactive_after = np.flatnonzero(~active[first + 1:])
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
            case.passive_mesh, case.bounds, resolution=resolution,
            backend=backend, device=device, analytic_sign_fn=case.passive_sdf,
            name=f"{case.name}_dynamic_mesh_sdf_{backend}",
        )
    raise ValueError("sdf_source must be 'analytic' or 'mesh'")


def _analytic_first_contact_time(case: DynamicsCase) -> float | None:
    g = -float(case.gravity[2])
    if g <= 0.0:
        return None
    z_min_local = float(case.active_mesh.bounds[0][2])
    clearance = float(case.initial_position[2] + z_min_local)
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
    mu: float = 0.0,
    ct: float = 0.0,
    dt: float | None = None,
    steps: int | None = None,
    snapshot_stride: int = 30,
    exact_manifold_stride: int = 0,
    contact_max_depth: int = 3,
    enable_rotation: bool = False,
    inertia_tensor: np.ndarray | None = None,
) -> DynamicsResult:
    dt = float(case.dt if dt is None else dt)
    steps = int(case.steps if steps is None else steps)
    method = "cubic" if method.lower().startswith("cub") else "linear"

    evaluator = SurfaceContactEvaluator(case.active_mesh, grid, method=method, backend=backend, mu=mu, ct=ct)
    pos = case.initial_position.astype(np.float64).copy()
    vel = case.initial_velocity.astype(np.float64).copy()
    mass = float(case.mass)
    g = case.gravity.astype(np.float64)
    com_local = np.asarray(case.local_com, dtype=np.float64)

    I_body = inertia_tensor if inertia_tensor is not None else _compute_inertia_tensor_tet(
        case.active_mesh.vertices, case.active_mesh.faces, mass
    )
    I_body_inv = np.linalg.inv(I_body)

    R0 = quat_to_rotation_matrix(quat_identity()) if enable_rotation else np.eye(3, dtype=np.float64)
    quat = rotation_matrix_to_quat(R0).copy()
    omega = np.zeros(3, dtype=np.float64)

    time = np.zeros(steps + 1, dtype=np.float64)
    position = np.zeros((steps + 1, 3), dtype=np.float64)
    velocity = np.zeros((steps + 1, 3), dtype=np.float64)
    contact_force = np.zeros((steps + 1, 3), dtype=np.float64)
    contact_torque = np.zeros((steps + 1, 3), dtype=np.float64)
    contact_area = np.zeros(steps + 1, dtype=np.float64)
    contact_count = np.zeros(steps + 1, dtype=np.int64)
    max_depth = np.zeros(steps + 1, dtype=np.float64)
    kinetic = np.zeros(steps + 1, dtype=np.float64)
    rot_energy = np.zeros(steps + 1, dtype=np.float64)
    grav_e = np.zeros(steps + 1, dtype=np.float64)
    contact_e = np.zeros(steps + 1, dtype=np.float64)
    total_e = np.zeros(steps + 1, dtype=np.float64)
    orientations = np.zeros((steps + 1, 4), dtype=np.float64) if enable_rotation else None
    ang_vels = np.zeros((steps + 1, 3), dtype=np.float64) if enable_rotation else None
    snapshots: List[dict] = []

    def record(i, result):
        t = i * dt
        R = quat_to_rotation_matrix(quat) if enable_rotation else np.eye(3, dtype=np.float64)
        com_world = pos + R @ com_local
        time[i] = t
        position[i] = pos
        velocity[i] = vel
        contact_force[i] = result.total_force
        contact_torque[i] = result.total_torque
        contact_area[i] = result.contact_area
        contact_count[i] = result.contact_count
        max_depth[i] = result.max_depth
        kinetic[i] = 0.5 * mass * float(np.dot(vel, vel))
        rot_energy[i] = 0.5 * float(omega @ I_body @ omega) if enable_rotation else 0.0
        grav_e[i] = -mass * float(np.dot(g, com_world))
        contact_e[i] = result.elastic_energy
        total_e[i] = kinetic[i] + rot_energy[i] + grav_e[i] + contact_e[i]
        if enable_rotation:
            orientations[i] = quat
            ang_vels[i] = omega
        if snapshot_stride > 0 and (i % snapshot_stride == 0 or i == steps or result.contact_count > 0 and len(snapshots) < 4):
            active = result.depths > 0.0
            idx = np.flatnonzero(active)
            if idx.size > 120:
                idx = idx[np.linspace(0, idx.size - 1, 120).astype(int)]
            snap = {
                "step": int(i), "time": float(t),
                "position": pos.tolist(), "velocity": vel.tolist(),
                "force": result.total_force.tolist(), "force_norm": result.force_norm,
                "contact_count": int(result.contact_count),
                "contact_area": float(result.contact_area),
                "max_depth": result.max_depth,
                "contact_points": result.points[idx].tolist() if idx.size else [],
                "contact_depths": result.depths[idx].tolist() if idx.size else [],
            }
            if enable_rotation:
                snap["orientation"] = quat.tolist()
                snap["angular_velocity"] = omega.tolist()
            snapshots.append(snap)

    com_world_init = pos + com_local
    res = evaluator.evaluate(pos, vel, kn=kn, alpha=alpha, cn=cn, torque_center=com_world_init, quaternion=quat if enable_rotation else None)
    record(0, res)

    exact_snapshots = []
    for i in range(steps):
        force = res.total_force + mass * g
        acc = force / mass
        vel = vel + dt * acc
        pos = pos + dt * vel

        if enable_rotation:
            tau_world = res.total_torque
            R_body = quat_to_rotation_matrix(quat)
            I_world = R_body @ I_body @ R_body.T
            I_world_inv = np.linalg.inv(I_world)
            alpha_ang = I_world_inv @ tau_world
            omega = omega + dt * alpha_ang
            dq = quat_derivative(quat, omega)
            quat = quat_normalize(quat + dt * dq)

        R_cur = quat_to_rotation_matrix(quat) if enable_rotation else np.eye(3, dtype=np.float64)
        com_world = pos + R_cur @ com_local
        res = evaluator.evaluate(
            pos, vel, kn=kn, alpha=alpha, cn=cn,
            torque_center=com_world,
            quaternion=quat if enable_rotation else None,
        )
        record(i + 1, res)

        if exact_manifold_stride > 0 and (i + 1) % exact_manifold_stride == 0 and res.contact_count > 0:
            moved_mesh = case.active_mesh.translated(pos)
            if enable_rotation:
                moved_mesh = moved_mesh.transformed(R=R_cur)
            manifold = extract_contact_region(
                moved_mesh, grid, method=method, max_depth=contact_max_depth,
                min_edge_length=0.8 * float(np.max(grid.spacing)),
                candidate_epsilon=2.5 * float(np.max(grid.spacing)),
            )
            _, comps = label_connected_components(manifold, threshold=2.0 * float(np.max(grid.spacing)))
            exact_force = compute_contact_forces(manifold, kn=kn, alpha=alpha, center_of_mass=com_world)
            exact_snapshots.append({
                "step": int(i + 1), "time": float((i + 1) * dt),
                "components": len(comps),
                "area": float(np.sum(manifold.areas())),
                "force": exact_force.total_force.tolist(),
                "force_norm": float(np.linalg.norm(exact_force.total_force)),
            })

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
        "dt": dt, "steps": steps, "mass": mass, "gravity": g.tolist(),
        "kn": float(kn), "alpha": float(alpha), "cn": float(cn),
        "mu": float(mu), "ct": float(ct),
        "enable_rotation": enable_rotation,
        "inertia_trace": float(np.trace(I_body)),
        "sdf_grid_shape": list(grid.shape),
        "sdf_grid_spacing": grid.spacing.tolist(),
        "backend_request": backend, "backend_used": evaluator.backend_used,
        "method": method,
        "explicit_integrator": "symplectic_euler_6dof" if enable_rotation else "symplectic_euler",
        "force_quadrature": "four-point per triangle (3 vertices + centroid)",
        "exact_manifold_snapshots": exact_snapshots,
    }
    return DynamicsResult(
        case_name=case.name, method=method, backend_used=evaluator.backend_used,
        time=time, position=position, velocity=velocity,
        orientation=orientations, angular_velocity=ang_vels,
        contact_force=contact_force, contact_torque=contact_torque,
        contact_area=contact_area, contact_count=contact_count, max_depth=max_depth,
        kinetic_energy=kinetic, rotational_energy=rot_energy,
        gravitational_energy=grav_e, contact_energy=contact_e, total_energy=total_e,
        events=events, snapshots=snapshots, settings=settings,
    )


def write_result_json(path: str | Path, result: DynamicsResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_dynamics_result(npz_path: str | Path, json_path: str | Path) -> DynamicsResult:
    npz_path = Path(npz_path)
    json_path = Path(json_path)
    arrays = np.load(npz_path, allow_pickle=False)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    settings = dict(meta.get("settings", {}))
    snapshots = list(meta.get("snapshots", []))
    orientation = arrays.get("orientation", None)
    angular_velocity = arrays.get("angular_velocity", None)
    rotational_energy = arrays.get("rotational_energy", None)
    return DynamicsResult(
        case_name=str(meta.get("case_name", "dynamic_case")),
        method=str(meta.get("method", "linear")),
        backend_used=str(meta.get("backend_used", "unknown")),
        time=arrays["time"], position=arrays["position"], velocity=arrays["velocity"],
        orientation=orientation, angular_velocity=angular_velocity,
        contact_force=arrays["contact_force"], contact_torque=arrays["contact_torque"],
        contact_area=arrays["contact_area"], contact_count=arrays["contact_count"],
        max_depth=arrays["max_depth"], kinetic_energy=arrays["kinetic_energy"],
        rotational_energy=rotational_energy,
        gravitational_energy=arrays["gravitational_energy"],
        contact_energy=arrays["contact_energy"], total_energy=arrays["total_energy"],
        events=dict(meta.get("events", {})), snapshots=snapshots, settings=settings,
    )
