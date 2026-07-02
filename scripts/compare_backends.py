from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sdf_contact.dynamics.scenarios import build_dynamics_cases
from sdf_contact.dynamics.integrator import build_dynamic_sdf_grid, simulate_case
from sdf_contact.dynamics.force_evaluator import NUMBA_AVAILABLE, TORCH_AVAILABLE


def compare_backends(
    case_idx: int = 0,
    resolution: int = 32,
    steps: int = 200,
    method: str = "linear",
    kn: float = 8.0e4,
    alpha: float = 1.0,
    cn: float = 120.0,
    mu: float = 0.0,
    ct: float = 0.0,
    out_dir: str = "outputs_backend_compare",
) -> dict:
    cases = build_dynamics_cases(quick=True)
    case = cases[case_idx]
    print(f"Case: {case.name}")
    print(f"  Active mesh faces: {case.active_mesh.face_count}")
    print(f"  Passive mesh faces: {case.passive_mesh.face_count}")

    grid = build_dynamic_sdf_grid(case, sdf_source="analytic", backend="auto", resolution=resolution)

    backends = ["numpy"]
    if TORCH_AVAILABLE:
        backends.append("torch")
    if NUMBA_AVAILABLE:
        backends.append("numba_cpu")

    results = {}
    for backend in backends:
        print(f"\nRunning backend: {backend}")
        t0 = time.perf_counter()
        try:
            result = simulate_case(
                case, grid=grid, method=method, backend=backend,
                kn=kn, alpha=alpha, cn=cn, mu=mu, ct=ct,
                dt=None, steps=steps, snapshot_stride=steps,
            )
            runtime = time.perf_counter() - t0
            results[backend] = {
                "runtime_seconds": runtime,
                "steps_per_second": (len(result.time) - 1) / runtime if runtime > 0 else 0,
                "position_final": result.position[-1].tolist(),
                "velocity_final": result.velocity[-1].tolist(),
                "max_force_norm": float(np.max(np.linalg.norm(result.contact_force, axis=1))),
                "max_depth": float(np.max(result.max_depth)),
                "energy_drift": result.settings.get("energy_relative_drift", 0),
                "first_contact": result.events.get("first_contact"),
                "contact_count_peak": int(np.max(result.contact_count)),
            }
            print(f"  Runtime: {runtime:.3f}s ({results[backend]['steps_per_second']:.0f} steps/s)")
            print(f"  Final pos: {result.position[-1]}")
            print(f"  Max force: {results[backend]['max_force_norm']:.4f}")
            print(f"  Energy drift: {results[backend]['energy_drift']:.6f}")
        except Exception as e:
            results[backend] = {"error": str(e)}
            print(f"  ERROR: {e}")

    if "numpy" in results and "cuda" in results and "error" not in results["numpy"] and "error" not in results["cuda"]:
        numpy_res = results["numpy"]
        cuda_res = results["cuda"]
        pos_diff = np.linalg.norm(np.array(numpy_res["position_final"]) - np.array(cuda_res["position_final"]))
        vel_diff = np.linalg.norm(np.array(numpy_res["velocity_final"]) - np.array(cuda_res["velocity_final"]))
        force_diff = abs(numpy_res["max_force_norm"] - cuda_res["max_force_norm"])
        results["consistency"] = {
            "position_l2_error": float(pos_diff),
            "velocity_l2_error": float(vel_diff),
            "max_force_abs_error": float(force_diff),
            "consistent": pos_diff < 1e-6 and vel_diff < 1e-6,
        }
        print(f"\nConsistency check (numpy vs cuda):")
        print(f"  Position L2 error: {pos_diff:.2e}")
        print(f"  Velocity L2 error: {vel_diff:.2e}")
        print(f"  Max force abs error: {force_diff:.2e}")
        print(f"  Consistent: {results['consistency']['consistent']}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path / 'comparison.json'}")
    return results


def main():
    p = argparse.ArgumentParser(description="Compare numpy/cuda/numba dynamics backends.")
    p.add_argument("--case", type=int, default=0, help="Case index (0-4)")
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--method", choices=["linear", "cubic"], default="linear")
    p.add_argument("--kn", type=float, default=8.0e4)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--ct", type=float, default=0.0)
    p.add_argument("--out", default="outputs_backend_compare")
    args = p.parse_args()
    compare_backends(
        case_idx=args.case, resolution=args.resolution, steps=args.steps,
        method=args.method, kn=args.kn, mu=args.mu, ct=args.ct, out_dir=args.out,
    )


if __name__ == "__main__":
    main()
