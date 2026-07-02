from __future__ import annotations

import numpy as np


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return np.array([
        qw * rw - qx * rx - qy * ry - qz * rz,
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw,
    ], dtype=np.float64)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-30:
        return quat_identity()
    return q / n


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    rotated = quat_mul(quat_mul(q, qv), quat_conjugate(q))
    return rotated[1:4]


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_normalize(q)
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    wx, wy, wz = omega
    return 0.5 * np.array([
        -qx*wx - qy*wy - qz*wz,
         qw*wx + qy*wz - qz*wy,
         qw*wy - qx*wz + qz*wx,
         qw*wz + qx*wy - qy*wx,
    ], dtype=np.float64)


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z], dtype=np.float64))


def compute_inertia_tensor(vertices: np.ndarray, faces: np.ndarray, mass: float) -> np.ndarray:
    n = len(faces)
    I = np.zeros((3, 3), dtype=np.float64)
    for k in range(n):
        v0 = vertices[faces[k, 0]]
        v1 = vertices[faces[k, 1]]
        v2 = vertices[faces[k, 2]]
        tri = np.stack([v0, v1, v2], axis=0)
        center = tri.mean(axis=0)
        e0 = v1 - v0
        e1 = v2 - v0
        cross = np.cross(e0, e1)
        det = abs(float(np.dot(v0, cross)))
        tet_vol = det / 6.0
        for i in range(3):
            for j in range(3):
                val = 0.0
                for s in range(3):
                    for t in range(3):
                        val += (1.0 + float(i == j)) * tri[s, i] * tri[t, j]
                        if i == j:
                            val += tri[s, i] * tri[t, i]
                        val -= float(i == j) * tri[s, i] * tri[t, i]
                        val -= tri[s, i] * tri[j, t] if i == j else 0.0
                I[i, j] += tet_vol * val
    I *= mass / np.abs(np.sum(np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    ), axis=0).mean() + 1e-30)
    total_vol = sum(
        abs(np.dot(vertices[faces[k, 0]], np.cross(
            vertices[faces[k, 1]] - vertices[faces[k, 0]],
            vertices[faces[k, 2]] - vertices[faces[k, 0]],
        ))) / 6.0
        for k in range(n)
    )
    if total_vol > 0:
        I = I * (mass / (2.0 * total_vol))
    return I


def compute_inertia_tensor_tet(vertices: np.ndarray, faces: np.ndarray, mass: float) -> np.ndarray:
    face_verts = vertices[faces]
    e01 = face_verts[:, 1] - face_verts[:, 0]
    e02 = face_verts[:, 2] - face_verts[:, 0]
    cross = np.cross(e01, e02)
    dets = np.abs(np.sum(face_verts[:, 0] * cross, axis=1))
    tet_vols = dets / 6.0
    total_vol = float(np.sum(tet_vols))
    if total_vol < 1e-30:
        return np.eye(3, dtype=np.float64) * (mass / 6.0)

    I = np.zeros((3, 3), dtype=np.float64)
    for k in range(len(faces)):
        v0 = face_verts[k, 0]
        v1 = face_verts[k, 1]
        v2 = face_verts[k, 2]
        w = tet_vols[k] / total_vol
        verts = np.stack([v0, v1, v2, np.zeros(3)], axis=0)
        for i in range(3):
            for j in range(3):
                contrib = 0.0
                for s in range(4):
                    for t in range(4):
                        contrib += verts[s, i] * verts[t, j]
                I[i, j] += w * contrib
    I_diag = np.array([
        I[1, 1] + I[2, 2] - I[0, 0],
        I[0, 0] + I[2, 2] - I[1, 1],
        I[0, 0] + I[1, 1] - I[2, 2],
    ], dtype=np.float64)
    I_off = np.zeros((3, 3), dtype=np.float64)
    I_off[0, 1] = -I[0, 1]
    I_off[0, 2] = -I[0, 2]
    I_off[1, 0] = -I[1, 0]
    I_off[1, 2] = -I[1, 2]
    I_off[2, 0] = -I[2, 0]
    I_off[2, 1] = -I[2, 1]
    result = mass * (np.diag(I_diag) + I_off)
    return 0.5 * (result + result.T)
