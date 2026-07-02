from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from sdf_contact.contact import compute_contact_forces, extract_contact_region, label_connected_components
from sdf_contact.geometry.quality import mesh_quality_report
from sdf_contact.sdf import SDFGrid, mesh_to_sdf_grid
from sdf_contact.validation import (
    basic_sdf_metrics,
    compare_contact_force,
    compare_grid_to_analytic,
    contact_summary,
    eikonal_metrics,
)
from sdf_contact.visualization import save_contact_html, save_force_html, save_sdf_slices, write_case_report, write_index

from .cases import ContactCase, build_cases


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _median_mesh_edge_length(mesh) -> float:
    f = mesh.faces
    v = mesh.vertices
    e01 = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
    e12 = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1)
    e20 = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)
    return float(np.median(np.concatenate([e01, e12, e20])))


def build_sdf_for_case(case: ContactCase, sdf_source: str, backend: str, resolution: int, device: str | None) -> SDFGrid:
    sdf_source = sdf_source.lower()
    if sdf_source == "analytic":
        return SDFGrid.from_analytic(case.passive_sdf, case.bounds, resolution=resolution, name=f"{case.name}_analytic_sdf")
    if sdf_source == "mesh":
        return mesh_to_sdf_grid(
            case.passive_mesh,
            case.bounds,
            resolution=resolution,
            backend=backend,
            device=device,
            analytic_sign_fn=case.passive_sdf,
            name=f"{case.name}_mesh_sdf_{backend}",
        )
    raise ValueError("sdf_source must be 'analytic' or 'mesh'")


def run_case(
    case: ContactCase,
    out_dir: str | Path,
    sdf_source: str = "analytic",
    backend: str = "auto",
    resolution: int = 64,
    device: str | None = "cuda:0",
    kn: float = 1.0e5,
    alpha: float = 1.0,
    contact_max_depth: int = 4,
    sample_count: int = 8000,
    make_visuals: bool = True,
) -> Dict:
    case_dir = Path(out_dir) / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    case.active_mesh.export_obj(case_dir / "active_mesh.obj")
    case.passive_mesh.export_obj(case_dir / "passive_mesh.obj")

    active_q = mesh_quality_report(case.active_mesh)
    passive_q = mesh_quality_report(case.passive_mesh)
    _write_json(case_dir / "mesh_quality.json", {"active": active_q, "passive": passive_q, "expected": case.expected})

    grid = build_sdf_for_case(case, sdf_source=sdf_source, backend=backend, resolution=resolution, device=device)
    grid.save_npz(case_dir / "passive_sdf_grid.npz")

    sdf_metrics = {
        "basic": basic_sdf_metrics(grid),
        "analytic_comparison": compare_grid_to_analytic(grid, case.passive_sdf, sample_count=sample_count),
        "eikonal": eikonal_metrics(grid, sample_count=sample_count, band=4.0 * float(np.max(grid.spacing))),
    }
    _write_json(case_dir / "sdf_metrics.json", sdf_metrics)

    if make_visuals:
        save_sdf_slices(grid, case_dir / "sdf_slices.png", title=f"{case.name} passive SDF")

    contacts = {}
    forces = {}
    summaries = {}
    component_threshold = 1.25 * _median_mesh_edge_length(case.active_mesh)
    for method in ["linear", "cubic"]:
        contact = extract_contact_region(
            case.active_mesh,
            grid,
            method=method,
            max_depth=contact_max_depth,
            min_edge_length=0.8 * float(np.max(grid.spacing)),
            candidate_epsilon=2.5 * float(np.max(grid.spacing)),
        )
        labels, components = label_connected_components(contact, threshold=component_threshold)
        force = compute_contact_forces(contact, kn=kn, alpha=alpha, center_of_mass=np.mean(case.active_mesh.vertices, axis=0))
        contacts[method] = contact
        forces[method] = force
        summaries[method] = contact_summary(contact, components)
        _write_json(case_dir / f"contact_{method}.json", summaries[method])
        _write_json(case_dir / f"force_{method}.json", force.to_dict())
        if make_visuals:
            save_contact_html(case.active_mesh, case.passive_mesh, contact, case_dir / f"contact_{method}.html", title=f"{case.name} contact {method}")
            save_force_html(case.active_mesh, case.passive_mesh, contact, force, case_dir / f"force_{method}.html", title=f"{case.name} force {method}")

    comparison = compare_contact_force(contacts["linear"], contacts["cubic"], forces["linear"], forces["cubic"])
    _write_json(case_dir / "linear_vs_cubic.json", comparison)

    metrics = {
        "case": case.name,
        "description": case.description,
        "expected": case.expected,
        "sdf_source": sdf_source,
        "backend": backend,
        "resolution": resolution,
        "mesh_quality": {"active": active_q, "passive": passive_q},
        "sdf_metrics": sdf_metrics,
        "contact": summaries,
        "force": {"linear": forces["linear"].to_dict(), "cubic": forces["cubic"].to_dict()},
        "linear_vs_cubic": comparison,
    }
    _write_json(case_dir / "case_metrics.json", metrics)

    if make_visuals:
        write_case_report(
            case_dir,
            case.name,
            metrics,
            {
                "SDF slices": "sdf_slices.png",
                "Contact linear": "contact_linear.html",
                "Contact cubic": "contact_cubic.html",
                "Force linear": "force_linear.html",
                "Force cubic": "force_cubic.html",
            },
        )

    return {
        "case": case.name,
        "active_faces": int(case.active_mesh.face_count),
        "passive_faces": int(case.passive_mesh.face_count),
        "linear_components": int(summaries["linear"]["components_count"]),
        "cubic_components": int(summaries["cubic"]["components_count"]),
        "linear_area": float(summaries["linear"]["area"]),
        "cubic_area": float(summaries["cubic"]["area"]),
        "linear_force_norm": float(np.linalg.norm(forces["linear"].total_force)),
        "cubic_force_norm": float(np.linalg.norm(forces["cubic"].total_force)),
        "expected": case.expected,
        "report": str(case_dir / "report.html"),
    }


def run_all(
    out_dir: str | Path,
    quick: bool = False,
    sdf_source: str = "analytic",
    backend: str = "auto",
    resolution: int = 64,
    device: str | None = "cuda:0",
    kn: float = 1.0e5,
    alpha: float = 1.0,
    contact_max_depth: int | None = None,
    sample_count: int | None = None,
    make_visuals: bool = True,
) -> list[Dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(quick=quick)
    if contact_max_depth is None:
        contact_max_depth = 3 if quick else 5
    if sample_count is None:
        sample_count = 3000 if quick else 15000
    summary = []
    for case in cases:
        row = run_case(
            case,
            out_dir=out_dir,
            sdf_source=sdf_source,
            backend=backend,
            resolution=resolution,
            device=device,
            kn=kn,
            alpha=alpha,
            contact_max_depth=contact_max_depth,
            sample_count=sample_count,
            make_visuals=make_visuals,
        )
        summary.append(row)
        _write_json(out_dir / "summary.partial.json", summary)
    _write_json(out_dir / "summary.json", summary)
    if make_visuals:
        write_index(out_dir, summary)
    return summary
