from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from sdf_contact.geometry.mesh import TriangleMesh

from .integrator import DynamicsResult


def _event_lines(ax, events: Dict[str, float | None], ymax=None):
    for name, t in events.items():
        if t is None or "error" in name:
            continue
        if name == "analytic_ballistic_first_contact":
            linestyle = ":"
        else:
            linestyle = "--"
        ax.axvline(float(t), linestyle=linestyle, linewidth=1.0)
        y0, y1 = ax.get_ylim()
        y = y1 if ymax is None else ymax
        ax.text(float(t), y, name.replace("_", "\n"), rotation=90, va="top", ha="right", fontsize=7)


def save_response_curves(result: DynamicsResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = result.time
    f_norm = np.linalg.norm(result.contact_force, axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    axes[0].plot(t, result.position[:, 2], label="z displacement")
    axes[0].set_ylabel("z [m]")
    axes[0].legend(loc="best")
    _event_lines(axes[0], result.events)

    axes[1].plot(t, result.velocity[:, 2], label="z velocity")
    axes[1].set_ylabel("vz [m/s]")
    axes[1].legend(loc="best")
    _event_lines(axes[1], result.events)

    axes[2].plot(t, result.contact_force[:, 2], label="contact Fz")
    axes[2].plot(t, f_norm, label="|contact F|")
    axes[2].set_ylabel("force [N]")
    axes[2].legend(loc="best")
    _event_lines(axes[2], result.events)

    axes[3].plot(t, result.max_depth, label="max penetration depth")
    axes[3].plot(t, result.contact_area, label="estimated contact area")
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("depth [m] / area [m²]")
    axes[3].legend(loc="best")
    _event_lines(axes[3], result.events)

    fig.suptitle(f"{result.case_name} / {result.method} / {result.backend_used}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_energy_plot(result: DynamicsResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = result.time
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, result.kinetic_energy, label="kinetic")
    ax.plot(t, result.gravitational_energy, label="gravitational")
    ax.plot(t, result.contact_energy, label="contact elastic")
    ax.plot(t, result.total_energy, label="total")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("energy [J]")
    ax.legend(loc="best")
    ax.set_title(f"Energy check: drift={result.to_dict()['energy_relative_drift']:.3e}")
    _event_lines(ax, result.events)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _mesh_trace(mesh: TriangleMesh, name: str, opacity: float = 0.35):
    v = mesh.vertices
    f = mesh.faces
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        name=name,
        opacity=opacity,
        flatshading=True,
        showscale=False,
    )


def save_dynamic_animation_html(active_mesh: TriangleMesh, passive_mesh: TriangleMesh, result: DynamicsResult, path: str | Path, max_frames: int = 10) -> None:
    """Create a lightweight Plotly slider animation of active mesh motion.

    The contact points come from the accelerated face-centroid evaluator snapshots,
    which are sufficient for state-sequence diagnosis.  Static exact manifold HTMLs
    remain the authoritative region visualization.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snaps = result.snapshots
    if not snaps:
        snaps = [{"time": 0.0, "position": result.position[0].tolist(), "contact_points": [], "contact_depths": []}]
    if len(snaps) > max_frames:
        idx = np.linspace(0, len(snaps) - 1, max_frames).astype(int)
        snaps = [snaps[i] for i in idx]

    passive_trace = _mesh_trace(passive_mesh, "passive fixed", opacity=0.28)

    def active_trace_for(snap):
        pos = np.asarray(snap["position"], dtype=np.float64)
        m = active_mesh.translated(pos)
        tr = _mesh_trace(m, "active falling", opacity=0.55)
        return tr

    def contact_trace_for(snap):
        pts = np.asarray(snap.get("contact_points", []), dtype=np.float64)
        depths = np.asarray(snap.get("contact_depths", []), dtype=np.float64)
        if pts.size == 0:
            pts = np.zeros((0, 3), dtype=np.float64)
            depths = np.zeros(0)
        return go.Scatter3d(
            x=pts[:, 0] if len(pts) else [],
            y=pts[:, 1] if len(pts) else [],
            z=pts[:, 2] if len(pts) else [],
            mode="markers",
            marker=dict(size=3, color=depths if len(depths) else [], colorscale="Viridis", showscale=True, colorbar=dict(title="depth")),
            name="contact samples",
        )

    frames = []
    for k, snap in enumerate(snaps):
        frames.append(go.Frame(data=[active_trace_for(snap), contact_trace_for(snap)], name=str(k)))

    fig = go.Figure(data=[passive_trace, active_trace_for(snaps[0]), contact_trace_for(snaps[0])], frames=frames)
    fig.update_layout(
        title=f"Dynamic contact sequence: {result.case_name} / {result.method}",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=45),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[dict(method="animate", args=[[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=f"{snaps[k]['time']:.3f}s") for k in range(len(snaps))],
                active=0,
            )
        ],
    )
    fig.write_html(path, include_plotlyjs="cdn")


def write_dynamics_case_report(case_dir: str | Path, result: DynamicsResult, assets: Dict[str, str]) -> None:
    case_dir = Path(case_dir)
    data = result.to_dict()
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in data["events"].items()
    )
    asset_links = "".join(f'<li><a href="{html.escape(rel)}">{html.escape(name)}</a></li>' for name, rel in assets.items())
    html_text = f"""
<!doctype html>
<meta charset="utf-8">
<title>{html.escape(result.case_name)} dynamics report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
code {{ background: #f5f5f5; padding: 2px 4px; }}
</style>
<h1>{html.escape(result.case_name)}</h1>
<p>Method: <code>{html.escape(result.method)}</code>; backend: <code>{html.escape(result.backend_used)}</code>.</p>
<h2>Contact events</h2>
<table>{rows}</table>
<h2>Summary</h2>
<pre>{html.escape(str(data))}</pre>
<h2>Assets</h2>
<ul>{asset_links}</ul>
<h2>Response curves</h2>
<img src="response_curves.png">
<h2>Energy check</h2>
<img src="energy.png">
</html>
"""
    (case_dir / "report.html").write_text(html_text, encoding="utf-8")


def write_dynamics_index(out_dir: str | Path, summary: Iterable[dict]) -> None:
    out_dir = Path(out_dir)
    rows = []
    for row in summary:
        rows.append(
            "<tr>"
            f"<td><a href='{html.escape(row['case'])}/{html.escape(row['method'])}/report.html'>{html.escape(row['case'])}</a></td>"
            f"<td>{html.escape(row['method'])}</td>"
            f"<td>{html.escape(row['backend_used'])}</td>"
            f"<td>{row['first_contact']}</td>"
            f"<td>{row['max_force_z']:.6g}</td>"
            f"<td>{row['max_depth']:.6g}</td>"
            f"<td>{row['energy_relative_drift']:.3e}</td>"
            "</tr>"
        )
    html_text = f"""
<!doctype html>
<meta charset="utf-8">
<title>SDF-Mesh explicit dynamics validation</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; }}
</style>
<h1>SDF-Mesh explicit dynamics validation</h1>
<p>自由落体动力学阶段用于验证上一阶段的 SDF 插值、接触区域与 penalty 接触力逻辑能否形成合理的位移、速度、接触力和能量响应。</p>
<table>
<tr><th>case</th><th>method</th><th>backend</th><th>first contact [s]</th><th>max Fz [N]</th><th>max depth [m]</th><th>energy drift</th></tr>
{''.join(rows)}
</table>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")
