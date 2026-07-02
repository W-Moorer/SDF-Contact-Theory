from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def write_case_report(case_dir: str | Path, case_name: str, metrics: Dict, images: Dict[str, str]) -> None:
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    html = ["<html><head><meta charset='utf-8'><title>{}</title></head><body>".format(case_name)]
    html.append(f"<h1>{case_name}</h1>")
    html.append("<h2>Metrics</h2><pre>{}</pre>".format(json.dumps(metrics, ensure_ascii=False, indent=2)))
    html.append("<h2>Outputs</h2><ul>")
    for label, rel in images.items():
        html.append(f"<li><a href='{rel}'>{label}</a></li>")
    html.append("</ul></body></html>")
    (case_dir / "report.html").write_text("\n".join(html), encoding="utf-8")


def write_index(out_dir: str | Path, summary: List[Dict]) -> None:
    out_dir = Path(out_dir)
    html = ["<html><head><meta charset='utf-8'><title>SDF-Mesh Contact Validation</title></head><body>"]
    html.append("<h1>SDF-Mesh Contact Validation Summary</h1>")
    html.append("<table border='1' cellspacing='0' cellpadding='4'>")
    html.append("<tr><th>Case</th><th>Active faces</th><th>Passive faces</th><th>Linear components</th><th>Cubic components</th><th>Linear area</th><th>Cubic area</th><th>|F_lin|</th><th>|F_cub|</th><th>Report</th></tr>")
    for row in summary:
        html.append(
            "<tr>"
            f"<td>{row['case']}</td>"
            f"<td>{row.get('active_faces', '')}</td>"
            f"<td>{row.get('passive_faces', '')}</td>"
            f"<td>{row.get('linear_components', '')}</td>"
            f"<td>{row.get('cubic_components', '')}</td>"
            f"<td>{row.get('linear_area', 0):.6g}</td>"
            f"<td>{row.get('cubic_area', 0):.6g}</td>"
            f"<td>{row.get('linear_force_norm', 0):.6g}</td>"
            f"<td>{row.get('cubic_force_norm', 0):.6g}</td>"
            f"<td><a href='{row['case']}/report.html'>open</a></td>"
            "</tr>"
        )
    html.append("</table>")
    html.append("<h2>Raw summary</h2><pre>{}</pre>".format(json.dumps(summary, ensure_ascii=False, indent=2)))
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
