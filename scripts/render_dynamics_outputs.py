from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdf_contact.dynamics.integrator import load_dynamics_result
from sdf_contact.dynamics.scenarios import build_dynamics_cases
from sdf_contact.dynamics.visualization import (
    save_dynamic_animation_html,
    save_energy_plot,
    save_response_curves,
    write_dynamics_case_report,
    write_dynamics_index,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Render plots/reports from saved SDF-Mesh dynamics time histories.")
    p.add_argument("--out", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--methods", default="linear,cubic")
    p.add_argument("--skip-animation", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    cases = {c.name: c for c in build_dynamics_cases(quick=args.quick)}
    summary = []
    for case_name, case in cases.items():
        for method in methods:
            d = out / case_name / method
            npz = d / "time_history.npz"
            js = d / "dynamics_result.json"
            if not npz.exists() or not js.exists():
                continue
            result = load_dynamics_result(npz, js)
            save_response_curves(result, d / "response_curves.png")
            save_energy_plot(result, d / "energy.png")
            assets = {
                "time history npz": "time_history.npz",
                "JSON summary": "dynamics_result.json",
                "response curves": "response_curves.png",
                "energy plot": "energy.png",
            }
            if not args.skip_animation:
                save_dynamic_animation_html(case.active_mesh, case.passive_mesh, result, d / "state_animation.html")
                assets["state animation"] = "state_animation.html"
            write_dynamics_case_report(d, result, assets)
            rd = result.to_dict()
            summary.append(
                {
                    "case": case_name,
                    "method": method,
                    "backend_used": result.backend_used,
                    "first_contact": rd["events"].get("first_contact"),
                    "analytic_ballistic_first_contact": rd["events"].get("analytic_ballistic_first_contact"),
                    "first_contact_error": rd["events"].get("first_contact_error"),
                    "max_force_z": rd["max_contact_force_z"],
                    "max_force_norm": rd["max_contact_force_norm"],
                    "max_depth": rd["max_depth"],
                    "max_contact_area": rd["max_contact_area"],
                    "energy_relative_drift": rd["energy_relative_drift"],
                    "report": str(d / "report.html"),
                }
            )
    (out / "summary.rendered.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_dynamics_index(out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
