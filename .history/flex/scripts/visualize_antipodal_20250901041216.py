#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import trimesh
import matplotlib.pyplot as plt

def draw_triad(ax, origin, R, length, alpha=0.9):
    """在 origin 画手爪坐标系，长度自适应 length（基于场景尺度）"""
    o = np.asarray(origin).ravel()
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    ax.quiver(o[0], o[1], o[2], *(length * x), color='r', alpha=alpha)
    ax.quiver(o[0], o[1], o[2], *(length * y), color='g', alpha=alpha)
    ax.quiver(o[0], o[1], o[2], *(length * z), color='b', alpha=alpha)

def load_pairs_with_meta(npy_path):
    arr = np.load(npy_path, allow_pickle=True)
    meta = None
    if len(arr) > 0 and isinstance(arr[-1], dict) and arr[-1].get("__meta__", False):
        meta = arr[-1]
        arr = arr[:-1]
    return list(arr), meta

def compute_scene_bounds(mesh, pairs, num_to_show):
    """把 mesh 顶点 + 将要展示的抓取点一并纳入，求统一包围盒"""
    V = mesh.vertices
    mins = V.min(axis=0).astype(float)
    maxs = V.max(axis=0).astype(float)

    K = min(num_to_show, len(pairs))
    for i in range(K):
        rec = pairs[i]
        if isinstance(rec, dict):
            p1, p2 = np.asarray(rec['p1'], float), np.asarray(rec['p2'], float)
        else:
            p1, p2, _, _ = rec
            p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
        mins = np.minimum(mins, np.minimum(p1, p2))
        maxs = np.maximum(maxs, np.maximum(p1, p2))

    center = 0.5 * (mins + maxs)
    span = (maxs - mins)
    radius = 0.5 * float(np.max(span))
    # 防止极端薄片半径过小
    radius = max(radius, 1e-6)
    return center, radius

def set_equal_aspect_3d(ax, center, radius):
    """尽可能设置 X/Y/Z 等比；优先 set_box_aspect 回退到手动限幅"""
    try:
        ax.set_box_aspect((1, 1, 1))  # matplotlib >=3.3
    except Exception:
        pass  # 老版本无此方法，继续用手动限幅

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

def main():
    # === 路径配置 ===
    npy_path  = "/home/rose/legograsp/flex/results/antipodal_pairs/antipodal_pairs_ray_with_pose.npy"
    mesh_path = "../lego.obj"

    pairs, meta = load_pairs_with_meta(npy_path)
    print(f"Loaded {len(pairs)} items from {npy_path}")
    if not pairs:
        print("未找到任何数据"); return

    # === 读取 mesh 并复现生成时的缩放/旋转（若有 meta）===
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    if meta is not None:
        scale = float(meta.get("scale", 1.0))
        if scale != 1.0: mesh.apply_scale(scale)
        T_mesh = meta.get("T_mesh_4x4", None)
        if T_mesh is not None:
            mesh.apply_transform(np.asarray(T_mesh, dtype=float))
        print("应用了 meta 中的 mesh 旋转/缩放复现。")
    else:
        mesh.apply_scale(0.01)  # 兼容旧数据

    # === 先算场景尺度，用于等比 & 坐标轴长度 ===
    num_to_show = min(20, len(pairs))
    center, radius = compute_scene_bounds(mesh, pairs, num_to_show)
    triad_len = 0.08 * radius     # 手爪坐标系箭头长度（可调）
    normal_len = 0.06 * radius    # 法向箭头长度（可调）

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 可选：用正交投影（不强制）——减少“看起来不等比”的透视错觉
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass

    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                    triangles=mesh.faces,
                    color=(0.7, 0.7, 0.9, 0.35),
                    edgecolor='gray', linewidth=0.2)

    # 统一设置等比范围（一定要在画完物体后设置）
    set_equal_aspect_3d(ax, center, radius)

    # 画点/法向/连线/坐标系
    for i in range(num_to_show):
        rec = pairs[i]
        if isinstance(rec, dict):
            p1, p2, n1, n2 = map(np.asarray, (rec["p1"], rec["p2"], rec["n1"], rec["n2"]))
            center_i = rec.get("center", 0.5 * (p1 + p2))
            R_i = rec.get("R", None)
            if R_i is None:
                # 兜底构造一个 R
                u = p2 - p1; u /= (np.linalg.norm(u) + 1e-12)
                nz = n1 + n2
                if np.linalg.norm(nz) < 1e-8:
                    tmp = np.array([0., 0., 1.]) if abs(u[2]) < 0.9 else np.array([0., 1., 0.])
                    z = np.cross(tmp, u); z /= (np.linalg.norm(z) + 1e-12)
                else:
                    z = -(nz / (np.linalg.norm(nz) + 1e-12))
                    if abs(np.dot(z, u)) > 0.99:
                        tmp = np.array([0., 0., 1.]) if abs(u[2]) < 0.9 else np.array([0., 1., 0.])
                        z = np.cross(tmp, u); z /= (np.linalg.norm(z) + 1e-12)
                y = np.cross(z, u); y /= (np.linalg.norm(y) + 1e-12)
                z = np.cross(u, y); z /= (np.linalg.norm(z) + 1e-12)
                R_i = np.column_stack([u, y, z])
        else:
            p1, p2, n1, n2 = rec
            p1 = np.asarray(p1); p2 = np.asarray(p2)
            n1 = np.asarray(n1); n2 = np.asarray(n2)
            center_i = 0.5 * (p1 + p2)
            # 兜底构造 R
            u = p2 - p1; u /= (np.linalg.norm(u) + 1e-12)
            nz = n1 + n2
            if np.linalg.norm(nz) < 1e-8:
                tmp = np.array([0., 0., 1.]) if abs(u[2]) < 0.9 else np.array([0., 1., 0.])
                z = np.cross(tmp, u); z /= (np.linalg.norm(z) + 1e-12)
            else:
                z = -(nz / (np.linalg.norm(nz) + 1e-12))
                if abs(np.dot(z, u)) > 0.99:
                    tmp = np.array([0., 0., 1.]) if abs(u[2]) < 0.9 else np.array([0., 1., 0.])
                    z = np.cross(tmp, u); z /= (np.linalg.norm(z) + 1e-12)
            y = np.cross(z, u); y /= (np.linalg.norm(y) + 1e-12)
            z = np.cross(u, y); z /= (np.linalg.norm(z) + 1e-12)
            R_i = np.column_stack([u, y, z])

        # 画
        ax.scatter(*p1, color='red',   s=30 if i else 60, label='p1' if i == 0 else "")
        ax.scatter(*p2, color='green', s=30 if i else 60, label='p2' if i == 0 else "")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='k', linestyle='--', linewidth=0.8)
        ax.quiver(p1[0], p1[1], p1[2], *(normal_len * n1), color='red',   alpha=0.7)
        ax.quiver(p2[0], p2[1], p2[2], *(normal_len * n2), color='green', alpha=0.7)
        draw_triad(ax, center_i, R_i, length=triad_len, alpha=0.95)

    ax.set_title("Antipodal Pairs with Pose (Equal XYZ Scale)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
