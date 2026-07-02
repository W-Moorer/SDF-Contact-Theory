import numpy as np

from sdf_contact.sdf import SDFGrid, PlaneSDF


def test_plane_sdf_grid_sampling_linear_and_cubic():
    sdf = PlaneSDF(z0=0.0)
    grid = SDFGrid.from_analytic(sdf, bounds=(np.array([-1, -1, -1]), np.array([1, 1, 1])), resolution=24)
    pts = np.array([[0.0, 0.0, 0.2], [0.1, -0.2, -0.3], [0.5, 0.5, 0.0]])
    for method in ["linear", "cubic"]:
        phi, grad = grid.sample(pts, method=method, return_grad=True)
        assert np.allclose(phi, pts[:, 2], atol=1e-5)
        assert np.allclose(grad, np.array([[0, 0, 1]] * 3), atol=2e-4)
