from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdf_contact.dynamics.runner import run_dynamics_all


def main() -> None:
    p = argparse.ArgumentParser(description="Run SDF-Mesh explicit free-fall dynamics validation.")
    p.add_argument("--out", default="outputs_dynamics", help="Output directory")
    p.add_argument("--quick", action="store_true", help="Use lower resolution meshes and fewer steps; all meshes still keep >=2000 faces.")
    p.add_argument("--sdf-source", choices=["analytic", "mesh"], default="analytic")
    p.add_argument("--sdf-backend", default="auto", help="SDF generation backend when --sdf-source mesh: auto/cpu/cupy/warp")
    p.add_argument("--resolution", type=int, default=48)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dynamics-backend", choices=["auto", "cuda", "numba", "numba_cpu", "numpy"], default="auto")
    p.add_argument("--method", choices=["linear", "cubic", "both"], default="both")
    p.add_argument("--kn", type=float, default=8.0e4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--cn", type=float, default=120.0)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--snapshot-stride", type=int, default=None)
    p.add_argument("--exact-manifold-stride", type=int, default=0, help="Optional expensive exact clipped-manifold snapshots.")
    p.add_argument("--case-limit", type=int, default=None)
    p.add_argument("--no-visuals", action="store_true")
    args = p.parse_args()
    methods = ["linear", "cubic"] if args.method == "both" else [args.method]
    summary = run_dynamics_all(
        out_dir=args.out,
        quick=args.quick,
        sdf_source=args.sdf_source,
        sdf_backend=args.sdf_backend,
        resolution=args.resolution,
        device=args.device,
        methods=methods,
        dynamics_backend=args.dynamics_backend,
        kn=args.kn,
        alpha=args.alpha,
        cn=args.cn,
        dt=args.dt,
        steps=args.steps,
        snapshot_stride=args.snapshot_stride,
        exact_manifold_stride=args.exact_manifold_stride,
        make_visuals=not args.no_visuals,
        case_limit=args.case_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
