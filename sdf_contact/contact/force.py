from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .region import ContactTriangles


@dataclass
class ForceResult:
    point_forces: np.ndarray
    points: np.ndarray
    pressures: np.ndarray
    total_force: np.ndarray
    total_torque: np.ndarray
    area: float
    force_by_component: list[dict]

    def to_dict(self) -> Dict:
        return {
            "area": float(self.area),
            "total_force": self.total_force.tolist(),
            "total_force_norm": float(np.linalg.norm(self.total_force)),
            "total_torque": self.total_torque.tolist(),
            "total_torque_norm": float(np.linalg.norm(self.total_torque)),
            "pressure_min": float(np.min(self.pressures)) if len(self.pressures) else 0.0,
            "pressure_max": float(np.max(self.pressures)) if len(self.pressures) else 0.0,
            "pressure_mean": float(np.mean(self.pressures)) if len(self.pressures) else 0.0,
            "components": self.force_by_component,
        }


def compute_contact_forces(
    contact: ContactTriangles,
    kn: float = 1.0e5,
    alpha: float = 1.0,
    center_of_mass: np.ndarray | None = None,
) -> ForceResult:
    if center_of_mass is None:
        center_of_mass = np.zeros(3, dtype=np.float64)
    center_of_mass = np.asarray(center_of_mass, dtype=np.float64)
    n = contact.count
    if n == 0:
        return ForceResult(
            point_forces=np.zeros((0, 3), dtype=np.float64),
            points=np.zeros((0, 3), dtype=np.float64),
            pressures=np.zeros(0, dtype=np.float64),
            total_force=np.zeros(3, dtype=np.float64),
            total_torque=np.zeros(3, dtype=np.float64),
            area=0.0,
            force_by_component=[],
        )
    areas = contact.areas()
    pts = contact.centroids()
    pressures = kn * np.power(np.maximum(contact.depth, 0.0), alpha)
    point_forces = pressures[:, None] * contact.normals * areas[:, None]
    total_force = np.sum(point_forces, axis=0)
    total_torque = np.sum(np.cross(pts - center_of_mass[None, :], point_forces), axis=0)
    labels = contact.labels if contact.labels is not None else np.zeros(n, dtype=np.int64)
    force_by_component: List[dict] = []
    for lab in np.unique(labels):
        mask = labels == lab
        f = np.sum(point_forces[mask], axis=0)
        tau = np.sum(np.cross(pts[mask] - center_of_mass[None, :], point_forces[mask]), axis=0)
        force_by_component.append(
            {
                "id": int(lab),
                "triangles": int(np.sum(mask)),
                "area": float(np.sum(areas[mask])),
                "force": f.tolist(),
                "force_norm": float(np.linalg.norm(f)),
                "torque": tau.tolist(),
                "torque_norm": float(np.linalg.norm(tau)),
                "mean_pressure": float(np.average(pressures[mask], weights=np.maximum(areas[mask], 1e-30))),
                "max_pressure": float(np.max(pressures[mask])),
            }
        )
    force_by_component.sort(key=lambda x: x["area"], reverse=True)
    return ForceResult(point_forces, pts, pressures, total_force, total_torque, float(np.sum(areas)), force_by_component)
