from __future__ import annotations

import numpy as np

from .mesh import TriangleMesh, merge_duplicate_vertices


def _add_grid_faces(faces: list[list[int]], idx: np.ndarray, flip: bool = False) -> None:
    """Add two triangles per grid cell for a rectangular index grid."""
    nx, ny = idx.shape
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = int(idx[i, j])
            b = int(idx[i + 1, j])
            c = int(idx[i + 1, j + 1])
            d = int(idx[i, j + 1])
            if not flip:
                faces.append([a, b, c])
                faces.append([a, c, d])
            else:
                faces.append([a, c, b])
                faces.append([a, d, c])


def make_torus(R: float = 1.0, r: float = 0.25, nu: int = 128, nv: int = 64, name: str = "torus") -> TriangleMesh:
    """Closed torus mesh with 2*nu*nv faces."""
    u = np.linspace(0.0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0.0, 2 * np.pi, nv, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    vertices = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    def id_(i: int, j: int) -> int:
        return (i % nu) * nv + (j % nv)

    faces: list[list[int]] = []
    for i in range(nu):
        for j in range(nv):
            a = id_(i, j)
            b = id_(i + 1, j)
            c = id_(i + 1, j + 1)
            d = id_(i, j + 1)
            faces.append([a, b, c])
            faces.append([a, c, d])
    return TriangleMesh(vertices, np.asarray(faces, dtype=np.int64), name).ensure_outward_orientation()


def make_subdivided_box(
    extents=(1.0, 1.0, 1.0),
    center=(0.0, 0.0, 0.0),
    n: int | tuple[int, int, int] = 32,
    name: str = "subdivided_box",
) -> TriangleMesh:
    """Closed six-face subdivided box.

    With n=32, it has 6*2*n*n = 12288 faces.
    """
    if isinstance(n, int):
        nx = ny = nz = n
    else:
        nx, ny, nz = n
    ex, ey, ez = np.asarray(extents, dtype=np.float64)
    cx, cy, cz = np.asarray(center, dtype=np.float64)
    hx, hy, hz = ex / 2.0, ey / 2.0, ez / 2.0

    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    def add_patch(xgrid: np.ndarray, ygrid: np.ndarray, zgrid: np.ndarray, flip: bool = False):
        base = len(vertices)
        pts = np.stack([xgrid, ygrid, zgrid], axis=-1).reshape(-1, 3)
        vertices.extend(pts.tolist())
        idx = np.arange(base, base + pts.shape[0], dtype=np.int64).reshape(xgrid.shape)
        _add_grid_faces(faces, idx, flip=flip)

    # z faces
    xs = np.linspace(cx - hx, cx + hx, nx + 1)
    ys = np.linspace(cy - hy, cy + hy, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    add_patch(X, Y, np.full_like(X, cz + hz), flip=False)  # top
    add_patch(X, Y, np.full_like(X, cz - hz), flip=True)   # bottom

    # y faces
    xs = np.linspace(cx - hx, cx + hx, nx + 1)
    zs = np.linspace(cz - hz, cz + hz, nz + 1)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    add_patch(X, np.full_like(X, cy + hy), Z, flip=True)
    add_patch(X, np.full_like(X, cy - hy), Z, flip=False)

    # x faces
    ys = np.linspace(cy - hy, cy + hy, ny + 1)
    zs = np.linspace(cz - hz, cz + hz, nz + 1)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    add_patch(np.full_like(Y, cx + hx), Y, Z, flip=False)
    add_patch(np.full_like(Y, cx - hx), Y, Z, flip=True)

    merged = merge_duplicate_vertices(np.asarray(vertices), np.asarray(faces, dtype=np.int64), tol=1e-12)
    merged.name = name
    return merged.ensure_outward_orientation()


def make_hollow_cylinder(
    outer_radius: float = 1.0,
    inner_radius: float = 0.7,
    height: float = 1.2,
    ntheta: int = 192,
    nz: int = 32,
    nr: int = 8,
    center=(0.0, 0.0, 0.0),
    name: str = "hollow_cylinder",
) -> TriangleMesh:
    """Closed annular cylinder mesh: outer wall + inner wall + top/bottom ring caps."""
    cx, cy, cz = np.asarray(center, dtype=np.float64)
    zmin, zmax = cz - height / 2.0, cz + height / 2.0
    theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
    zvals = np.linspace(zmin, zmax, nz + 1)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    def wall(radius: float, flip: bool):
        base = len(vertices)
        for iz, z in enumerate(zvals):
            for th in theta:
                vertices.append([cx + radius * np.cos(th), cy + radius * np.sin(th), z])
        def idx(iz: int, it: int) -> int:
            return base + iz * ntheta + (it % ntheta)
        for iz in range(nz):
            for it in range(ntheta):
                a, b, c, d = idx(iz, it), idx(iz + 1, it), idx(iz + 1, it + 1), idx(iz, it + 1)
                if not flip:
                    faces.append([a, d, c])
                    faces.append([a, c, b])
                else:
                    faces.append([a, b, c])
                    faces.append([a, c, d])

    wall(outer_radius, flip=False)
    wall(inner_radius, flip=True)

    radii = np.linspace(inner_radius, outer_radius, nr + 1)
    def ring_cap(z: float, top: bool):
        base = len(vertices)
        for ir, rr in enumerate(radii):
            for th in theta:
                vertices.append([cx + rr * np.cos(th), cy + rr * np.sin(th), z])
        def idx(ir: int, it: int) -> int:
            return base + ir * ntheta + (it % ntheta)
        for ir in range(nr):
            for it in range(ntheta):
                a, b, c, d = idx(ir, it), idx(ir + 1, it), idx(ir + 1, it + 1), idx(ir, it + 1)
                if top:
                    faces.append([a, b, c])
                    faces.append([a, c, d])
                else:
                    faces.append([a, c, b])
                    faces.append([a, d, c])
    ring_cap(zmax, top=True)
    ring_cap(zmin, top=False)

    merged = merge_duplicate_vertices(np.asarray(vertices), np.asarray(faces, dtype=np.int64), tol=1e-12)
    merged.name = name
    return merged.ensure_outward_orientation()


def make_cone(
    radius: float = 0.8,
    height: float = 1.2,
    ntheta: int = 256,
    nz: int = 32,
    nr: int = 16,
    center=(0.0, 0.0, 0.0),
    name: str = "cone",
) -> TriangleMesh:
    """Closed circular cone with a disk base and apex.

    z runs from center_z-height/2 at base to center_z+height/2 at apex.
    """
    cx, cy, cz = np.asarray(center, dtype=np.float64)
    z_base = cz - height / 2.0
    z_apex = cz + height / 2.0
    theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    # Side rings, excluding apex duplicate ring.
    for iz in range(nz):
        s = iz / nz
        rr = radius * (1.0 - s)
        z = z_base + s * height
        for th in theta:
            vertices.append([cx + rr * np.cos(th), cy + rr * np.sin(th), z])
    apex_idx = len(vertices)
    vertices.append([cx, cy, z_apex])

    def idx(iz: int, it: int) -> int:
        return iz * ntheta + (it % ntheta)

    for iz in range(nz - 1):
        for it in range(ntheta):
            a, b, c, d = idx(iz, it), idx(iz, it + 1), idx(iz + 1, it + 1), idx(iz + 1, it)
            faces.append([a, b, c])
            faces.append([a, c, d])
    iz = nz - 1
    for it in range(ntheta):
        faces.append([idx(iz, it), idx(iz, it + 1), apex_idx])

    # Base cap with radial subdivisions.
    base_center = len(vertices)
    vertices.append([cx, cy, z_base])
    radii = np.linspace(0.0, radius, nr + 1)[1:]
    base_start = len(vertices)
    for rr in radii:
        for th in theta:
            vertices.append([cx + rr * np.cos(th), cy + rr * np.sin(th), z_base])

    def bidx(ir: int, it: int) -> int:
        # ir=0 is center; ir>=1 ring index
        if ir == 0:
            return base_center
        return base_start + (ir - 1) * ntheta + (it % ntheta)

    for it in range(ntheta):
        faces.append([bidx(0, 0), bidx(1, it + 1), bidx(1, it)])
    for ir in range(1, nr):
        for it in range(ntheta):
            a, b, c, d = bidx(ir, it), bidx(ir + 1, it), bidx(ir + 1, it + 1), bidx(ir, it + 1)
            faces.append([a, c, b])
            faces.append([a, d, c])

    merged = merge_duplicate_vertices(np.asarray(vertices), np.asarray(faces, dtype=np.int64), tol=1e-12)
    merged.name = name
    return merged.ensure_outward_orientation()


def make_wavy_bottom_box(
    size=(2.0, 2.0),
    height: float = 0.6,
    bottom_mean_z: float = 0.0,
    amplitude: float = 0.08,
    modes=(3, 3),
    nx: int = 80,
    ny: int = 80,
    nz_side: int = 24,
    center_xy=(0.0, 0.0),
    name: str = "wavy_bottom_box",
) -> TriangleMesh:
    """Closed box with sinusoidal lower surface and flat top."""
    lx, ly = np.asarray(size, dtype=np.float64)
    cx, cy = np.asarray(center_xy, dtype=np.float64)
    m, n = modes
    x = np.linspace(cx - lx / 2.0, cx + lx / 2.0, nx + 1)
    y = np.linspace(cy - ly / 2.0, cy + ly / 2.0, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # sin(pi-scale) puts valleys/peaks inside the footprint and zero displacement at boundaries.
    xi = (X - (cx - lx / 2.0)) / lx
    eta = (Y - (cy - ly / 2.0)) / ly
    Zb = bottom_mean_z + amplitude * np.sin(2 * np.pi * m * xi) * np.sin(2 * np.pi * n * eta)
    Zt = np.full_like(Zb, bottom_mean_z + height)

    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    def add_patch(xgrid: np.ndarray, ygrid: np.ndarray, zgrid: np.ndarray, flip: bool = False):
        base = len(vertices)
        pts = np.stack([xgrid, ygrid, zgrid], axis=-1).reshape(-1, 3)
        vertices.extend(pts.tolist())
        idx = np.arange(base, base + pts.shape[0], dtype=np.int64).reshape(xgrid.shape)
        _add_grid_faces(faces, idx, flip=flip)

    add_patch(X, Y, Zt, flip=False)
    add_patch(X, Y, Zb, flip=True)

    # Four side patches. Each side interpolates from wavy bottom edge to top edge.
    s = np.linspace(0.0, 1.0, nz_side + 1)

    # x-min / x-max sides, parameter y,z
    for side_i, flip in [(0, True), (-1, False)]:
        Yg, Sg = np.meshgrid(y, s, indexing="ij")
        Xg = np.full_like(Yg, x[side_i])
        Zedge = Zb[side_i, :]
        Zg = (1.0 - Sg) * Zedge[:, None] + Sg * (bottom_mean_z + height)
        add_patch(Xg, Yg, Zg, flip=flip)

    # y-min / y-max sides, parameter x,z
    for side_j, flip in [(0, False), (-1, True)]:
        Xg, Sg = np.meshgrid(x, s, indexing="ij")
        Yg = np.full_like(Xg, y[side_j])
        Zedge = Zb[:, side_j]
        Zg = (1.0 - Sg) * Zedge[:, None] + Sg * (bottom_mean_z + height)
        add_patch(Xg, Yg, Zg, flip=flip)

    merged = merge_duplicate_vertices(np.asarray(vertices), np.asarray(faces, dtype=np.int64), tol=1e-12)
    merged.name = name
    return merged.ensure_outward_orientation()


def make_plane_slab(
    size=(3.0, 3.0),
    thickness: float = 0.1,
    top_z: float = 0.0,
    n: int = 80,
    name: str = "plane_slab",
) -> TriangleMesh:
    return make_subdivided_box(extents=(size[0], size[1], thickness), center=(0.0, 0.0, top_z - thickness / 2.0), n=n, name=name)
