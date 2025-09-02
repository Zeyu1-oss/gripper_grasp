#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import trimesh
import matplotlib.pyplot as plt

def draw_triad(ax, origin, R, length=0.02, alpha=0.9):
    o = np.asarray(origin).ravel()
    x, y, z = R[:,0], R[:,1], R[:,2]
    ax.quiver(o[0], o[1], o[2], *(length*x), color='r', alpha=alpha)
    ax.quiver(o[0], o[1], o[2], *(length*y), color='g', alpha=alpha)
    ax.quiver(o[0], o[1], o[2], *(length*z), color='b', alpha=alpha)

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    span = limits[:,1]-limits[:,0]
    centers = limits.mean(axis=1)
    radius = 0.5*max(span)
    ax.set_xlim3d([centers[0]-radius, centers[0]+radius])
    ax.set_ylim3d([centers[1]-radius, centers[1]+radius])
    ax.set_zlim3d([centers[2]-radius, centers[2]+radius])

def load_pairs_with_meta(npy_path):
    arr = np.load(npy_path, allow_pickle=True)
    # 检查最后一条是否 meta
    meta = None
    if len(arr)>0 and isinstance(arr[-1], dict) and arr[-1].get("__meta__", False):
        meta = arr[-1]
        arr = arr[:-1]
    return list(arr), meta

def main():
    # === 路径配置 ===
    npy_path  = "/home/rose/legograsp/flex/results/antipodal_pairs/antipodal_pairs_ray_with_pose.npy"
    mesh_path = "../lego.obj"

    pairs, meta = load_pairs_with_meta(npy_path)
    print(f"Loaded {len(pairs)} items from {npy_path}")
    if not pairs:
        print("未找到任何数据"); return

    # === 读取 mesh 并应用与生成时相同的 scale/旋转（若 meta 提供）===
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
        # 没 meta（兼容旧文件）：按你习惯的固定缩放
        mesh.apply_scale(0.01)

    # === 可视化 ===
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                    triangles=mesh.faces, color=(0.7,0.7,0.9,0.35),
                    edgecolor='gray', linewidth=0.2)

    num_to_show = min(20, len(pairs))  # 想多看一些就调大
    for i in range(num_to_show):
        rec = pairs[i]
        # 兼容：若只有 (p1,p2,n1,n2)，则临时算一个 pose
        if isinstance(rec, dict):
            p1, p2, n1, n2 = rec["p1"], rec["p2"], rec["n1"], rec["n2"]
            center, R = rec.get("center", None), rec.get("R", None)
            if center is None or R is None:
                # 兜底：即时计算
                from numpy.linalg import norm
                x = (p2 - p1) / (norm(p2 - p1) + 1e-12)
                nz = n1 + n2
                if norm(nz) < 1e-8:
                    tmp = np.array([0.,0.,1.]) if abs(x[2])<0.9 else np.array([0.,1.,0.])
                    z = np.cross(tmp, x); z /= (norm(z)+1e-12)
                else:
                    z = -(nz / (norm(nz)+1e-12))
                    if abs(np.dot(z, x)) > 0.99:
                        tmp = np.array([0.,0.,1.]) if abs(x[2])<0.9 else np.array([0.,1.,0.])
                        z = np.cross(tmp, x); z /= (norm(z)+1e-12)
                y = np.cross(z, x); y /= (norm(y)+1e-12)
                z = np.cross(x, y); z /= (norm(z)+1e-12)
                R = np.column_stack([x, y, z])
                center = 0.5*(p1+p2)
        else:
            p1, p2, n1, n2 = rec
            # 同样兜底计算
            from numpy.linalg import norm
            x = (p2 - p1) / (norm(p2 - p1) + 1e-12)
            nz = n1 + n2
            if norm(nz) < 1e-8:
                tmp = np.array([0.,0.,1.]) if abs(x[2])<0.9 else np.array([0.,1.,0.])
                z = np.cross(tmp, x); z /= (norm(z)+1e-12)
            else:
                z = -(nz / (norm(nz)+1e-12))
                if abs(np.dot(z, x)) > 0.99:
                    tmp = np.array([0.,0.,1.]) if abs(x[2])<0.9 else np.array([0.,1.,0.])
                    z = np.cross(tmp, x); z /= (norm(z)+1e-12)
            y = np.cross(z, x); y /= (norm(y)+1e-12)
            z = np.cross(x, y); z /= (norm(z)+1e-12)
            R = np.column_stack([x, y, z])
            center = 0.5*(p1+p2)

        # 画点/法向/连线/坐标系
        ax.scatter(*p1, color='red',   s=20 if i else 60, label='p1' if i==0 else "")
        ax.scatter(*p2, color='green', s=20 if i else 60, label='p2' if i==0 else "")
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], color='k', linestyle='--', linewidth=0.8)
        ax.quiver(p1[0],p1[1],p1[2], n1[0],n1[1],n1[2], length=0.02, color='red',   alpha=0.6)
        ax.quiver(p2[0],p2[1],p2[2], n2[0],n2[1],n2[2], length=0.02, color='green', alpha=0.6)
        draw_triad(ax, center, R, length=0.03, alpha=0.9)

    set_axes_equal(ax)
    ax.set_title("Antipodal Pairs with Pose")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

