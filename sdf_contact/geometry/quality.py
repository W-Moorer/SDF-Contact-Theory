from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import numpy as np

from .mesh import TriangleMesh


def mesh_quality_report(mesh: TriangleMesh) -> Dict[str, Any]:
    areas = mesh.face_areas()
    edges = []
    for f in mesh.faces:
        a, b, c = map(int, f)
        edges.extend([tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))])
    counts = Counter(edges)
    boundary_edges = sum(1 for v in counts.values() if v == 1)
    nonmanifold_edges = sum(1 for v in counts.values() if v != 2)
    mn, mx = mesh.bounds
    report: Dict[str, Any] = {
        "name": mesh.name,
        "vertices": int(mesh.vertex_count),
        "faces": int(mesh.face_count),
        "min_area": float(np.min(areas)) if len(areas) else 0.0,
        "max_area": float(np.max(areas)) if len(areas) else 0.0,
        "mean_area": float(np.mean(areas)) if len(areas) else 0.0,
        "zero_or_tiny_area_faces": int(np.sum(areas < 1e-14)),
        "boundary_edges": int(boundary_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
        "watertight_by_edge_count": bool(boundary_edges == 0 and nonmanifold_edges == 0),
        "aabb_min": mn.tolist(),
        "aabb_max": mx.tolist(),
    }
    try:
        tm = mesh.to_trimesh()
        report.update(
            {
                "trimesh_is_watertight": bool(tm.is_watertight),
                "trimesh_euler_number": int(tm.euler_number),
                "trimesh_volume": float(tm.volume),
            }
        )
    except Exception as exc:
        report["trimesh_error"] = repr(exc)
    return report
