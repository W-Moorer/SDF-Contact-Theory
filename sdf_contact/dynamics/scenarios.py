from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from sdf_contact.geometry import (
    TriangleMesh,
    make_cone,
    make_hollow_cylinder,
    make_plane_slab,
    make_subdivided_box,
    make_torus,
    make_wavy_bottom_box,
)
from sdf_contact.geometry.mesh import rotation_matrix_xyz
from sdf_contact.sdf import BoxSDF, PlaneSDF, padded_bounds_from_meshes


@dataclass
class DynamicsCase:
    """Rigid-body free-fall validation case.

    The passive object is fixed and represented by the same SDFGrid interface as
    the static contact framework.  The active body is translated by the explicit
    integrator; rotation is intentionally frozen in this validation stage so that
    the response curves isolate the SDF/contact-force chain.
    """

    name: str
    active_mesh: TriangleMesh
    passive_mesh: TriangleMesh
    passive_sdf: object
    bounds: Tuple[np.ndarray, np.ndarray]
    initial_position: np.ndarray
    initial_velocity: np.ndarray
    mass: float
    dt: float
    steps: int
    gravity: np.ndarray
    expected: Dict
    description: str = ""
    local_com: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.initial_position = np.asarray(self.initial_position, dtype=np.float64).reshape(3)
        self.initial_velocity = np.asarray(self.initial_velocity, dtype=np.float64).reshape(3)
        self.gravity = np.asarray(self.gravity, dtype=np.float64).reshape(3)
        if self.local_com is None:
            mn, mx = self.active_mesh.bounds
            self.local_com = 0.5 * (mn + mx)
        else:
            self.local_com = np.asarray(self.local_com, dtype=np.float64).reshape(3)


def _params(quick: bool):
    if quick:
        return {
            "torus": dict(nu=40, nv=28),
            "box_n": 14,
            "hollow": dict(ntheta=48, nz=8, nr=3),
            "cone": dict(ntheta=72, nz=12, nr=4),
            "wavy": dict(nx=24, ny=24, nz_side=8),
            "plane_n": 14,
            "steps": 360,
            "dt": 8.0e-4,
        }
    return {
        "torus": dict(nu=128, nv=64),
        "box_n": 32,
        "hollow": dict(ntheta=192, nz=32, nr=8),
        "cone": dict(ntheta=256, nz=32, nr=16),
        "wavy": dict(nx=80, ny=80, nz_side=24),
        "plane_n": 80,
        "steps": 1600,
        "dt": 4.0e-4,
    }


def _bounds_for_drop(active: TriangleMesh, passive: TriangleMesh, z0: float, zmin_extra: float = 0.4, zmax_extra: float = 0.35):
    a0 = active.translated(np.array([0.0, 0.0, z0], dtype=np.float64))
    mn, mx = padded_bounds_from_meshes([a0, passive], padding=0.25)
    mn[2] = min(mn[2], passive.bounds[0][2] - zmin_extra)
    mx[2] = max(mx[2], active.bounds[1][2] + z0 + zmax_extra)
    return mn, mx


def build_dynamics_cases(quick: bool = False) -> list[DynamicsCase]:
    """Build physically plausible free-fall cases from the existing analytic meshes.

    These cases deliberately avoid the nonphysical static interpenetration setups
    used only for topological stress tests.  They reuse the same model families:
    cube, torus, hollow cylinder, cone and wavy-bottom cube.
    """
    p = _params(quick)
    g = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    cases: list[DynamicsCase] = []

    # 1. Baseline: subdivided cube free-falls onto an analytical plane.
    box = make_subdivided_box(extents=(0.82, 0.82, 0.40), center=(0.0, 0.0, 0.0), n=p["box_n"], name="active_freefall_cube")
    plane = make_plane_slab(size=(2.8, 2.8), thickness=0.12, top_z=0.0, n=p["plane_n"], name="passive_plane_slab")
    z0 = 0.30
    cases.append(
        DynamicsCase(
            name="dyn_cube_plane_freefall_baseline",
            active_mesh=box,
            passive_mesh=plane,
            passive_sdf=PlaneSDF(z0=0.0, normal_sign=1.0),
            bounds=_bounds_for_drop(box, plane, z0),
            initial_position=np.array([0.0, 0.0, z0]),
            initial_velocity=np.zeros(3),
            mass=1.0,
            dt=p["dt"],
            steps=p["steps"],
            gravity=g,
            expected={"type": "自由落体基准", "first_contact_analytic": "sqrt(2*(z0-half_height)/g)"},
            description="细分立方体从静止自由落体到解析平面，用解析撞击时间校验显式积分和接触事件检测。",
        )
    )

    # 2. Ring manifold: torus free-falls onto a finite cube support.
    torus = make_torus(R=0.62, r=0.16, **p["torus"], name="active_freefall_torus")
    support_ring = make_subdivided_box(extents=(1.9, 1.9, 0.30), center=(0.0, 0.0, -0.15), n=p["box_n"], name="passive_cube_support_for_torus")
    z0 = 0.26
    cases.append(
        DynamicsCase(
            name="dyn_torus_cube_freefall_ring",
            active_mesh=torus,
            passive_mesh=support_ring,
            passive_sdf=BoxSDF(extents=(1.9, 1.9, 0.30), center=(0.0, 0.0, -0.15), name="cube_support_sdf"),
            bounds=_bounds_for_drop(torus, support_ring, z0),
            initial_position=np.array([0.0, 0.0, z0]),
            initial_velocity=np.zeros(3),
            mass=0.65,
            dt=p["dt"],
            steps=p["steps"],
            gravity=g,
            expected={"type": "环状自由落体接触", "contact_manifold": "torus lower tube creates ring-like patch on cube top"},
            description="圆环从静止下落到固定立方体支撑面，动力学接触应形成环状压力带，水平合力近似抵消。",
        )
    )

    # 3. Multi-region manifold: wavy-bottom cube free-falls onto a plane.
    wavy = make_wavy_bottom_box(
        size=(1.55, 1.55),
        height=0.46,
        bottom_mean_z=0.0,
        amplitude=0.065,
        modes=(3, 3),
        **p["wavy"],
        name="active_freefall_wavy_bottom_cube",
    )
    plane2 = make_plane_slab(size=(2.4, 2.4), thickness=0.12, top_z=0.0, n=p["plane_n"], name="passive_plane_slab_for_wavy")
    z0 = 0.20
    cases.append(
        DynamicsCase(
            name="dyn_wavy_cube_plane_freefall_multi_region",
            active_mesh=wavy,
            passive_mesh=plane2,
            passive_sdf=PlaneSDF(z0=0.0, normal_sign=1.0),
            bounds=_bounds_for_drop(wavy, plane2, z0),
            initial_position=np.array([0.0, 0.0, z0]),
            initial_velocity=np.zeros(3),
            mass=1.2,
            dt=p["dt"],
            steps=p["steps"],
            gravity=g,
            expected={"type": "多区域自由落体接触", "contact_manifold": "multiple wave valleys contact first"},
            description="波浪底面立方体下落到平面，多个波谷应先后进入接触，验证多区域接触流形在时域中的出现与消失。",
        )
    )

    # 4. Hollow cylinder free-falls onto a cube support.  This is a physical
    # dynamic counterpart to the hollow-cylinder geometry; the original side-wall
    # penetration case remains only a static topological stress test.
    hollow = make_hollow_cylinder(outer_radius=0.58, inner_radius=0.32, height=0.46, **p["hollow"], name="active_freefall_hollow_cylinder")
    support_hollow = make_subdivided_box(extents=(1.45, 1.45, 0.30), center=(0.0, 0.0, -0.15), n=p["box_n"], name="passive_cube_support_for_hollow")
    z0 = 0.33
    cases.append(
        DynamicsCase(
            name="dyn_hollow_cylinder_cube_freefall_annular",
            active_mesh=hollow,
            passive_mesh=support_hollow,
            passive_sdf=BoxSDF(extents=(1.45, 1.45, 0.30), center=(0.0, 0.0, -0.15), name="cube_support_sdf"),
            bounds=_bounds_for_drop(hollow, support_hollow, z0),
            initial_position=np.array([0.0, 0.0, z0]),
            initial_velocity=np.zeros(3),
            mass=0.85,
            dt=p["dt"],
            steps=p["steps"],
            gravity=g,
            expected={"type": "空心圆柱自由落体", "contact_manifold": "annular lower rim/base support contact"},
            description="空心圆柱自由落体到立方体支撑面，验证环状底面/边缘接触在动力学时程中的稳定计算。",
        )
    )

    # 5. Cone apex-down free-falls onto a cube support.  Rotation is frozen;
    # this is a clean force-spike and rebound benchmark rather than a stability test.
    cone0 = make_cone(radius=0.42, height=0.82, **p["cone"], name="active_cone_original")
    cone = cone0.transformed(R=rotation_matrix_xyz(rx=np.pi), name="active_freefall_cone_apex_down")
    support_cone = make_subdivided_box(extents=(1.25, 1.25, 0.30), center=(0.0, 0.0, -0.15), n=p["box_n"], name="passive_cube_support_for_cone")
    z0 = 0.51
    cases.append(
        DynamicsCase(
            name="dyn_cone_cube_freefall_apex",
            active_mesh=cone,
            passive_mesh=support_cone,
            passive_sdf=BoxSDF(extents=(1.25, 1.25, 0.30), center=(0.0, 0.0, -0.15), name="cube_support_sdf"),
            bounds=_bounds_for_drop(cone, support_cone, z0),
            initial_position=np.array([0.0, 0.0, z0]),
            initial_velocity=np.zeros(3),
            mass=0.55,
            dt=p["dt"],
            steps=p["steps"],
            gravity=g,
            expected={"type": "锥体自由落体尖端接触", "contact_manifold": "small apex/side patch after impact"},
            description="尖端向下的圆锥下落到立方体支撑面，验证尖锐几何接触中的力峰和接触事件标注。",
        )
    )
    return cases
