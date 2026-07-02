from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from sdf_contact.geometry.mesh import TriangleMesh
from sdf_contact.contact.region import ContactTriangles
from sdf_contact.contact.force import ForceResult
from sdf_contact.sdf.grid import SDFGrid


def save_sdf_slices(grid: SDFGrid, path: str | Path, title: str = "SDF slices") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = grid.values
    nx, ny, nz = grid.shape
    slices = [
        (vals[nx // 2, :, :].T, "x-mid", 1, 2),
        (vals[:, ny // 2, :].T, "y-mid", 0, 2),
        (vals[:, :, nz // 2].T, "z-mid", 0, 1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    vmax = np.percentile(np.abs(vals), 95)
    for ax, (sl, name, ax0, ax1) in zip(axes, slices):
        im = ax.imshow(sl, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        try:
            ax.contour(sl, levels=[0.0], colors="k", linewidths=1.0)
        except Exception:
            pass
        ax.set_title(name)
        ax.set_xlabel(f"axis {ax0}")
        ax.set_ylabel(f"axis {ax1}")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _mesh3d_trace(mesh: TriangleMesh, opacity: float = 0.22, name: str | None = None):
    v = mesh.vertices
    f = mesh.faces
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        name=name or mesh.name,
        opacity=opacity,
        color="lightgray",
        showscale=False,
    )


def save_contact_html(
    active_mesh: TriangleMesh,
    passive_mesh: TriangleMesh | None,
    contact: ContactTriangles,
    path: str | Path,
    title: str = "contact region",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    traces.append(_mesh3d_trace(active_mesh, opacity=0.14, name=f"active: {active_mesh.name}"))
    if passive_mesh is not None:
        traces.append(_mesh3d_trace(passive_mesh, opacity=0.10, name=f"passive: {passive_mesh.name}"))
    if contact.count > 0:
        tris = contact.triangles.reshape(-1, 3)
        k = np.arange(contact.count * 3).reshape(-1, 3)
        intensity = np.repeat(contact.depth, 3)
        traces.append(
            go.Mesh3d(
                x=tris[:, 0], y=tris[:, 1], z=tris[:, 2],
                i=k[:, 0], j=k[:, 1], k=k[:, 2],
                intensity=intensity,
                colorscale="Turbo",
                colorbar=dict(title="penetration"),
                name="contact depth",
                opacity=1.0,
                showscale=True,
            )
        )
        cent = contact.centroids()
        traces.append(
            go.Scatter3d(
                x=cent[:, 0], y=cent[:, 1], z=cent[:, 2],
                mode="markers",
                marker=dict(size=2, color=contact.depth, colorscale="Turbo", showscale=False),
                name="contact quadrature points",
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


def save_force_html(
    active_mesh: TriangleMesh,
    passive_mesh: TriangleMesh | None,
    contact: ContactTriangles,
    force: ForceResult,
    path: str | Path,
    title: str = "contact force distribution",
    arrow_scale: float | None = None,
    max_arrows: int = 2000,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    traces.append(_mesh3d_trace(active_mesh, opacity=0.12, name=f"active: {active_mesh.name}"))
    if passive_mesh is not None:
        traces.append(_mesh3d_trace(passive_mesh, opacity=0.09, name=f"passive: {passive_mesh.name}"))
    pts = force.points
    vec = force.point_forces
    if len(pts):
        idx = np.arange(len(pts))
        if len(idx) > max_arrows:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(idx, size=max_arrows, replace=False))
        pts_s = pts[idx]
        vec_s = vec[idx]
        mag = np.linalg.norm(vec_s, axis=1)
        if arrow_scale is None:
            diag = np.linalg.norm(active_mesh.bounds[1] - active_mesh.bounds[0])
            denom = np.percentile(mag[mag > 0], 95) if np.any(mag > 0) else 1.0
            arrow_scale = 0.08 * diag / denom
        end = pts_s + vec_s * arrow_scale
        # line segments
        xs, ys, zs = [], [], []
        for p, e in zip(pts_s, end):
            xs.extend([p[0], e[0], None])
            ys.extend([p[1], e[1], None])
            zs.extend([p[2], e[2], None])
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="force vectors"))
        traces.append(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=2, color=force.pressures, colorscale="Turbo", colorbar=dict(title="pressure")),
                name="pressure samples",
            )
        )
        # resultant force at contact centroid
        c = np.average(pts, axis=0, weights=np.maximum(force.pressures, 1e-30))
        rf = force.total_force
        rf_norm = np.linalg.norm(rf)
        if rf_norm > 0:
            re = c + rf / rf_norm * 0.25 * np.linalg.norm(active_mesh.bounds[1] - active_mesh.bounds[0])
            traces.append(go.Scatter3d(x=[c[0], re[0]], y=[c[1], re[1]], z=[c[2], re[2]], mode="lines+markers", name="resultant force"))
    fig = go.Figure(data=traces)
    fig.update_layout(title=title, scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(str(path), include_plotlyjs="cdn")
