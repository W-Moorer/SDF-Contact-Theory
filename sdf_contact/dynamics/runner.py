from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from sdf_contact.geometry.quality import mesh_quality_report

from .integrator import build_dynamic_sdf_grid, simulate_case, write_result_json
from .scenarios import DynamicsCase, build_dynamics_cases
from .visualization import save_dynamic_animation_html, save_energy_plot, save_response_curves, write_dynamics_case_report, write_dynamics_index


def _write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_dynamics_case(
    case: DynamicsCase,
    out_dir: str | Path,
    sdf_source: str = "analytic",
    sdf_backend: str = "auto",
    resolution: int = 48,
    device: str | None = "cuda:0",
    methods: Iterable[str] = ("linear", "cubic"),
    dynamics_backend: str = "auto",
    kn: float = 8.0e4,
    alpha: float = 1.0,
    cn: float = 120.0,
    dt: float | None = None,
    steps: int | None = None,
    snapshot_stride: int = 30,
    exact_manifold_stride: int = 0,
    make_visuals: bool = True,
) -> List[Dict]:
    case_root = Path(out_dir) / case.name
    case_root.mkdir(parents=True, exist_ok=True)
    case.active_mesh.export_obj(case_root / "active_reference_mesh.obj")
    case.passive_mesh.export_obj(case_root / "passive_mesh.obj")
    _write_json(
        case_root / "case_setup.json",
        {
            "name": case.name,
            "description": case.description,
            "expected": case.expected,
            "initial_position": case.initial_position.tolist(),
            "initial_velocity": case.initial_velocity.tolist(),
            "mass": case.mass,
            "dt": case.dt if dt is None else dt,
            "steps": case.steps if steps is None else steps,
            "gravity": case.gravity.tolist(),
            "mesh_quality": {"active": mesh_quality_report(case.active_mesh), "passive": mesh_quality_report(case.passive_mesh)},
        },
    )

    grid = build_dynamic_sdf_grid(case, sdf_source=sdf_source, backend=sdf_backend, resolution=resolution, device=device)
    grid.save_npz(case_root / "passive_sdf_grid.npz")

    rows: List[Dict] = []
    for method in methods:
        method = "cubic" if method.lower().startswith("cub") else "linear"
        method_dir = case_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        result = simulate_case(
            case,
            grid=grid,
            method=method,
            backend=dynamics_backend,
            kn=kn,
            alpha=alpha,
            cn=cn,
            dt=dt,
            steps=steps,
            snapshot_stride=snapshot_stride,
            exact_manifold_stride=exact_manifold_stride,
        )
        runtime = time.perf_counter() - t0
        result.settings["runtime_seconds"] = float(runtime)
        result.settings["steps_per_second"] = float((len(result.time) - 1) / runtime) if runtime > 0 else 0.0
        result.save_npz(method_dir / "time_history.npz")
        write_result_json(method_dir / "dynamics_result.json", result)
        if make_visuals:
            save_response_curves(result, method_dir / "response_curves.png")
            save_energy_plot(result, method_dir / "energy.png")
            save_dynamic_animation_html(case.active_mesh, case.passive_mesh, result, method_dir / "state_animation.html")
            write_dynamics_case_report(
                method_dir,
                result,
                {
                    "time history npz": "time_history.npz",
                    "JSON summary": "dynamics_result.json",
                    "state animation": "state_animation.html",
                    "response curves": "response_curves.png",
                    "energy plot": "energy.png",
                },
            )
        d = result.to_dict()
        rows.append(
            {
                "case": case.name,
                "method": method,
                "backend_used": result.backend_used,
                "runtime_seconds": runtime,
                "steps_per_second": result.settings["steps_per_second"],
                "first_contact": d["events"].get("first_contact"),
                "analytic_ballistic_first_contact": d["events"].get("analytic_ballistic_first_contact"),
                "first_contact_error": d["events"].get("first_contact_error"),
                "max_force_z": float(np.max(result.contact_force[:, 2])),
                "max_force_norm": d["max_contact_force_norm"],
                "max_depth": d["max_depth"],
                "max_contact_area": d["max_contact_area"],
                "energy_relative_drift": d["energy_relative_drift"],
                "report": str(method_dir / "report.html"),
            }
        )
    _write_json(case_root / "summary.json", rows)
    return rows


def run_dynamics_all(
    out_dir: str | Path,
    quick: bool = False,
    sdf_source: str = "analytic",
    sdf_backend: str = "auto",
    resolution: int = 48,
    device: str | None = "cuda:0",
    methods: Iterable[str] = ("linear", "cubic"),
    dynamics_backend: str = "auto",
    kn: float = 8.0e4,
    alpha: float = 1.0,
    cn: float = 120.0,
    dt: float | None = None,
    steps: int | None = None,
    snapshot_stride: int | None = None,
    exact_manifold_stride: int = 0,
    make_visuals: bool = True,
    case_limit: int | None = None,
) -> List[Dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if snapshot_stride is None:
        snapshot_stride = 24 if quick else 60
    cases = build_dynamics_cases(quick=quick)
    if case_limit is not None:
        cases = cases[: int(case_limit)]
    summary: List[Dict] = []
    for case in cases:
        rows = run_dynamics_case(
            case,
            out_dir=out_dir,
            sdf_source=sdf_source,
            sdf_backend=sdf_backend,
            resolution=resolution,
            device=device,
            methods=methods,
            dynamics_backend=dynamics_backend,
            kn=kn,
            alpha=alpha,
            cn=cn,
            dt=dt,
            steps=steps,
            snapshot_stride=snapshot_stride,
            exact_manifold_stride=exact_manifold_stride,
            make_visuals=make_visuals,
        )
        summary.extend(rows)
        _write_json(out_dir / "summary.partial.json", summary)
    _write_json(out_dir / "summary.json", summary)
    if make_visuals:
        write_dynamics_index(out_dir, summary)
    return summary
