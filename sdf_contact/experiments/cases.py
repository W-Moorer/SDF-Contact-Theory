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
from sdf_contact.sdf import BoxSDF, HollowCylinderSDF, PlaneSDF, TorusSDF, padded_bounds_from_meshes


@dataclass
class ContactCase:
    name: str
    active_mesh: TriangleMesh
    passive_mesh: TriangleMesh
    passive_sdf: object
    bounds: Tuple[np.ndarray, np.ndarray]
    expected: Dict
    description: str = ""


def build_cases(quick: bool = False) -> list[ContactCase]:
    if quick:
        params = {
            "torus": dict(nu=40, nv=28),
            "box_n": 14,
            "hollow": dict(ntheta=48, nz=8, nr=3),
            "cone": dict(ntheta=72, nz=12, nr=4),
            "wavy": dict(nx=24, ny=24, nz_side=8),
            "plane_n": 14,
        }
    else:
        params = {
            "torus": dict(nu=128, nv=64),
            "box_n": 32,
            "hollow": dict(ntheta=192, nz=32, nr=8),
            "cone": dict(ntheta=256, nz=32, nr=16),
            "wavy": dict(nx=80, ny=80, nz_side=24),
            "plane_n": 80,
        }

    cases: list[ContactCase] = []

    # 1. Ring contact: cube top face intersects lower half of torus tube.
    torus = make_torus(R=1.0, r=0.25, **params["torus"], name="passive_torus")
    cube_ring = make_subdivided_box(extents=(2.8, 2.8, 0.36), center=(0.0, 0.0, -0.36), n=params["box_n"], name="active_cube_for_ring")
    sdf_torus = TorusSDF(R=1.0, r=0.25)
    cases.append(
        ContactCase(
            name="torus_cube_ring",
            active_mesh=cube_ring,
            passive_mesh=torus,
            passive_sdf=sdf_torus,
            bounds=padded_bounds_from_meshes([torus, cube_ring], padding=0.35),
            expected={"type": "环状接触", "components_expected": 1},
            description="立方体上表面切入圆环下半管，接触区应为一个主环形带。",
        )
    )

    # 2. Four-point/edge contact: cube corners penetrate inner wall of hollow cylinder.
    hollow = make_hollow_cylinder(outer_radius=1.0, inner_radius=0.72, height=1.3, **params["hollow"], name="passive_hollow_cylinder")
    cube_inner = make_subdivided_box(extents=(1.08, 1.08, 1.05), center=(0.0, 0.0, 0.0), n=params["box_n"], name="active_cube_inside_hollow_cylinder")
    sdf_hollow = HollowCylinderSDF(outer_radius=1.0, inner_radius=0.72, height=1.3)
    cases.append(
        ContactCase(
            name="hollow_cylinder_cube_four_point",
            active_mesh=cube_inner,
            passive_mesh=hollow,
            passive_sdf=sdf_hollow,
            bounds=padded_bounds_from_meshes([hollow, cube_inner], padding=0.3),
            expected={"type": "多点接触", "components_expected": 4},
            description="立方体四条竖直角边轻微切入空心圆柱内壁，预期约四个接触分量。",
        )
    )

    # 3. Cone-cube multi-patch contact: cone side cuts through a thin cube slab.
    # The cone radius within the slab lies between cube half side and half diagonal,
    # producing four separated arcs/patches clipped by the square footprint.
    cone = make_cone(radius=0.90, height=1.60, center=(0.0, 0.0, 0.0), **params["cone"], name="active_cone")
    cube_slab = make_subdivided_box(extents=(0.70, 0.70, 0.20), center=(0.0, 0.0, 0.0), n=params["box_n"], name="passive_cube_slab")
    sdf_cube_slab = BoxSDF(extents=(0.70, 0.70, 0.20), center=(0.0, 0.0, 0.0), name="cube_slab_sdf")
    cases.append(
        ContactCase(
            name="cone_cube_four_patch",
            active_mesh=cone,
            passive_mesh=cube_slab,
            passive_sdf=sdf_cube_slab,
            bounds=padded_bounds_from_meshes([cone, cube_slab], padding=0.25),
            expected={"type": "锥体-立方体多点/多区域接触", "components_expected": 4},
            description="圆锥侧面穿过薄立方体，圆截线被方形截面裁成四个接触斑块。",
        )
    )

    # 4. Wavy bottom cube against an analytical plane z=0. Closed plane surrogate is still exported as a thin slab.
    wavy = make_wavy_bottom_box(size=(2.0, 2.0), height=0.55, bottom_mean_z=0.02, amplitude=0.08, modes=(3, 3), **params["wavy"], name="active_wavy_bottom_cube")
    plane = make_plane_slab(size=(2.8, 2.8), thickness=0.12, top_z=0.0, n=params["plane_n"], name="passive_plane_slab")
    sdf_plane = PlaneSDF(z0=0.0, normal_sign=1.0)
    cases.append(
        ContactCase(
            name="wavy_cube_plane_multi_region",
            active_mesh=wavy,
            passive_mesh=plane,
            passive_sdf=sdf_plane,
            bounds=padded_bounds_from_meshes([wavy, plane], padding=0.25),
            expected={"type": "多区域接触", "components_expected": "约等于波谷数量"},
            description="波浪底面的多个波谷下穿平面，形成周期性多斑块接触。",
        )
    )

    # 5. Same wavy cube against a finite cube support, clipping several contact patches.
    wavy2 = make_wavy_bottom_box(size=(2.0, 2.0), height=0.55, bottom_mean_z=0.02, amplitude=0.08, modes=(3, 3), **params["wavy"], name="active_wavy_bottom_cube")
    support = make_subdivided_box(extents=(1.25, 1.25, 0.35), center=(0.0, 0.0, -0.175), n=params["box_n"], name="passive_finite_cube_support")
    sdf_support = BoxSDF(extents=(1.25, 1.25, 0.35), center=(0.0, 0.0, -0.175), name="finite_cube_support_sdf")
    cases.append(
        ContactCase(
            name="wavy_cube_cube_clipped_multi_region",
            active_mesh=wavy2,
            passive_mesh=support,
            passive_sdf=sdf_support,
            bounds=padded_bounds_from_meshes([wavy2, support], padding=0.25),
            expected={"type": "边界裁剪后的多区域接触", "components_expected": "小于或等于无限平面接触斑块数"},
            description="有限立方体支撑面裁剪波浪底面接触区域，验证边界裁剪。",
        )
    )

    return cases
