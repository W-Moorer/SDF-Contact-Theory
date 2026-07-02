#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sdf_contact.experiments import run_all


def main() -> None:
    p = argparse.ArgumentParser(description="Run SDF-Mesh contact validation cases.")
    p.add_argument("--out", type=str, default="outputs", help="Output directory")
    p.add_argument("--quick", action="store_true", help="Use lower SDF resolution and smaller-but-still-closed meshes")
    p.add_argument("--sdf-source", choices=["analytic", "mesh"], default="analytic", help="Use analytic truth SDF or Mesh→SDF grid")
    p.add_argument("--backend", choices=["auto", "warp", "cupy", "cpu", "numba"], default="auto", help="Mesh→SDF backend")
    p.add_argument("--resolution", type=int, default=None, help="SDF grid resolution per axis")
    p.add_argument("--device", type=str, default="cuda:0", help="GPU device for Warp/CuPy where applicable")
    p.add_argument("--kn", type=float, default=1e5, help="Penalty normal stiffness")
    p.add_argument("--alpha", type=float, default=1.0, help="Penalty exponent: 1.0 linear, 1.5 Hertz-like")
    p.add_argument("--contact-max-depth", type=int, default=None, help="Recursive triangle subdivision depth")
    p.add_argument("--sample-count", type=int, default=None, help="Random samples for SDF metrics")
    p.add_argument("--no-visuals", action="store_true", help="Skip PNG/HTML visual outputs")
    args = p.parse_args()

    resolution = args.resolution
    if resolution is None:
        resolution = 36 if args.quick else 96

    summary = run_all(
        out_dir=Path(args.out),
        quick=args.quick,
        sdf_source=args.sdf_source,
        backend=args.backend,
        resolution=resolution,
        device=args.device,
        kn=args.kn,
        alpha=args.alpha,
        contact_max_depth=args.contact_max_depth,
        sample_count=args.sample_count,
        make_visuals=not args.no_visuals,
    )
    print("\nCompleted cases:")
    for row in summary:
        print(
            f"- {row['case']}: faces(active/passive)={row['active_faces']}/{row['passive_faces']}, "
            f"components linear/cubic={row['linear_components']}/{row['cubic_components']}, "
            f"area linear/cubic={row['linear_area']:.6g}/{row['cubic_area']:.6g}, "
            f"|F| linear/cubic={row['linear_force_norm']:.6g}/{row['cubic_force_norm']:.6g}"
        )
    print(f"\nOpen report: {Path(args.out) / 'index.html'}")


if __name__ == "__main__":
    main()
