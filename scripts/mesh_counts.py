#!/usr/bin/env python3
from sdf_contact.experiments import build_cases

for quick in [True, False]:
    print("quick=" + str(quick))
    for c in build_cases(quick=quick):
        print(f"{c.name:42s} active={c.active_mesh.face_count:8d} passive={c.passive_mesh.face_count:8d}")
