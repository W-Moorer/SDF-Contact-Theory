import numpy as np

from sdf_contact.dynamics.scenarios import build_dynamics_cases
from sdf_contact.dynamics.integrator import build_dynamic_sdf_grid, simulate_case
from sdf_contact.dynamics.force_evaluator import SurfaceContactEvaluator


def test_dynamic_cases_mesh_counts():
    for case in build_dynamics_cases(quick=True):
        assert case.active_mesh.face_count >= 2000
        assert case.passive_mesh.face_count >= 2000


def test_cube_plane_freefall_first_contact_smoke():
    case = build_dynamics_cases(quick=True)[0]
    grid = build_dynamic_sdf_grid(case, sdf_source="analytic", backend="auto", resolution=24)
    result = simulate_case(case, grid, method="linear", backend="numpy", steps=80, dt=case.dt, kn=8e4, cn=120, snapshot_stride=0)
    assert result.backend_used in {"numba_cpu", "numpy", "cuda"}
    assert np.all(np.isfinite(result.position))
    assert np.all(np.isfinite(result.contact_force))
    assert result.events["analytic_ballistic_first_contact"] is not None


def test_surface_evaluator_linear_cubic_shapes():
    case = build_dynamics_cases(quick=True)[0]
    grid = build_dynamic_sdf_grid(case, sdf_source="analytic", backend="auto", resolution=24)
    for method in ["linear", "cubic"]:
        ev = SurfaceContactEvaluator(case.active_mesh, grid, method=method, backend="numpy")
        res = ev.evaluate(case.initial_position, case.initial_velocity, kn=1e5, cn=0.0)
        assert res.points.shape[0] == 4 * case.active_mesh.face_count
        assert res.forces.shape == res.points.shape
        assert np.all(np.isfinite(res.forces))
