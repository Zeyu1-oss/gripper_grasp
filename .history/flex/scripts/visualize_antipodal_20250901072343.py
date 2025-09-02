#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# ================== 工具函数 ==================
def plot_grasp_distribution(mesh, pairs, save=None):
    """绘制抓取点分布图（p1 和 p2）"""
    all_p1, all_p2 = [], []
    for rec in pairs:
        if isinstance(rec, dict):
            p1 = np.asarray(rec['p1'], float)
            p2 = np.asarray(rec['p2'], float)
        else:
            p1, p2, _, _ = rec
            p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        all_p1.append(p1); all_p2.append(p2)

    all_p1 = np.array(all_p1)
    all_p2 = np.array(all_p2)

    fig = plt.figure(figsize=(10, 5))

    # ---- 左侧：投影分布 (XY 平面) ----
    ax1 = fig.add_subplot(121)
    ax1.scatter(all_p1[:,0], all_p1[:,1], c='red', s=5, alpha=0.6, label='p1')
    ax1.scatter(all_p2[:,0], all_p2[:,1], c='green', s=5, alpha=0.6, label='p2')
    ax1.set_title("Grasp Points Distribution (XY projection)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.legend()

    # ---- 右侧：3D 散点 ----
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(all_p1[:,0], all_p1[:,1], all_p1[:,2], c='red', s=5, alpha=0.6, label='p1')
    ax2.scatter(all_p2[:,0], all_p2[:,1], all_p2[:,2], c='green', s=5, alpha=0.6, label='p2')
    ax2.set_title("3D Grasp Points Distribution")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.legend()

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
        print(f"已保存分布图到: {save}")
    plt.show()


def draw_triad(ax, origin, R, length, alpha=0.95):
    """在 origin 用旋转矩阵 R 画三轴"""
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


def unit(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n < eps else v / n


def rec_to_pose(rec):
    """统一取出 (p1,p2,n1,n2,center,R)"""
    if isinstance(rec, dict):
        p1, p2 = np.asarray(rec["p1"], float), np.asarray(rec["p2"], float)
        n1, n2 = np.asarray(rec["n1"], float), np.asarray(rec["n2"], float)
        center = rec.get("center", None)
        R = rec.get("R", None)
    else:
        p1, p2, n1, n2 = rec
        p1, p2, n1, n2 = np.asarray(p1, float), np.asarray(p2, float), np.asarray(n1, float), np.asarray(n2, float)
        center, R = None, None

    if center is None or R is None:
        x = unit(p2 - p1)
        nz = n1 + n2
        if np.linalg.norm(nz) < 1e-8:
            tmp = np.array([0.,0.,1.]) if abs(x[2]) < 0.9 else np.array([0.,1.,0.])
            z = unit(np.cross(tmp, x))
        else:
            z = -unit(nz)
            if abs(np.dot(z, x)) > 0.99:
                tmp = np.array([0.,0.,1.]) if abs(x[2]) < 0.9 else np.array([0.,1.,0.])
                z = unit(np.cross(tmp, x))
        y = unit(np.cross(z, x))
        z = unit(np.cross(x, y))
        R = np.column_stack([x, y, z])
        center = 0.5 * (p1 + p2)

    return p1, p2, n1, n2, center, R


def compute_scene_bounds(mesh, picks, pairs):
    """统一包围盒"""
    V = mesh.vertices
    mins = V.min(axis=0).astype(float)
    maxs = V.max(axis=0).astype(float)
    for idx in picks:
        if 0 <= idx < len(pairs):
            rec = pairs[idx]
            if isinstance(rec, dict):
                p1, p2 = np.asarray(rec['p1'], float), np.asarray(rec['p2'], float)
            else:
                p1, p2, _, _ = rec
                p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
            mins = np.minimum(mins, np.minimum(p1, p2))
            maxs = np.maximum(maxs, np.maximum(p1, p2))
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    return center, max(radius, 1e-6)


def set_equal_aspect_3d(ax, center, radius):
    try: ax.set_box_aspect((1,1,1))
    except Exception: pass
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)


# ================== 主逻辑 ==================
def plot_scene(mesh, pairs, picks, ortho=True, show_mesh=True, save=None):
    center, radius = compute_scene_bounds(mesh, picks, pairs)
    triad_len, normal_len = 0.08*radius, 0.06*radius

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if ortho:
        try: ax.set_proj_type('ortho')
        except Exception: pass

    if show_mesh:
        ax.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                        triangles=mesh.faces, color=(0.7,0.7,0.9,0.35),
                        edgecolor='gray', linewidth=0.2)

    set_equal_aspect_3d(ax, center, radius)

    for k, idx in enumerate(picks):
        rec = pairs[idx]
        p1, p2, n1, n2, c, R = rec_to_pose(rec)
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], 'k--')
        ax.scatter(*p1, c='r', s=40); ax.scatter(*p2, c='g', s=40)
        ax.quiver(*p1, *(normal_len*n1), color='r')
        ax.quiver(*p2, *(normal_len*n2), color='g')
        draw_triad(ax, c, R, triad_len)
        print(f"=== #{idx} ===\np1={p1}, n1={n1}\np2={p2}, n2={n2}\ncenter={c}\n")

    ax.set_title("Antipodal Grasp Pairs (with Normals & Pose)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    if save: plt.savefig(save, dpi=200)
    plt.show()


def run(args):
    pairs, meta = load_pairs_with_meta(args.npy_path)
    if not pairs: return
    mesh = trimesh.load(args.mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene): mesh = mesh.dump().sum()
    if meta: mesh.apply_scale(float(meta.get("scale",1.0)))

    # ✅ 支持 topn
    if args.indices:
        picks = [int(x) for x in args.indices.split(',')]
    elif args.idx is not None:
        picks = [args.idx]
    elif args.topn is not None:
        picks = list(range(min(args.topn, len(pairs))))
    else:
        picks = [0]

    plot_scene(mesh, pairs, picks, ortho=not args.persp, show_mesh=not args.hide_mesh, save=args.save)


# ================== CLI ==================
def build_argparser():
    ap = argparse.ArgumentParser("可视化抓取对 (支持 topn)")
    ap.add_argument("--npy_path", type=str, default="/home/rose/legograsp/flex/results/antipodal_pairs/lego_pairs.npy")
    ap.add_argument("--mesh_path", type=str, default="../lego.obj")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--indices", type=str, default=None)
    ap.add_argument("--topn", type=int, default=100, help="展示前 N 个抓取对")
    ap.add_argument("--persp", action="store_true")
    ap.add_argument("--hide_mesh", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())

# import numpy as np
# import trimesh
# import matplotlib.pyplot as plt

# # === 路径配置 ===
# npy_path = "/home/rose/legograsp/flex/results/antipodal_pairs/antipodal_pairs_ray.npy"
# mesh_path = "../lego.obj"   

# # === 读取 antipodal 点对 ===
# pairs = np.load(npy_path, allow_pickle=True)
# print(f"Loaded {len(pairs)} antipodal pairs from: {npy_path}")
# print("每个元素格式: (p1, p2, n1, n2)")

# if len(pairs) == 0:
#     print(" 未找到任何 antipodal 点对！")
#     exit()

# # === 读取 mesh ===
# mesh = trimesh.load(mesh_path, force='mesh')
# scale = 0.01
# mesh.apply_scale(scale)
# if isinstance(mesh, trimesh.Scene):
#     mesh = mesh.dump().sum()

# # === 可视化 ===
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
#                 triangles=mesh.faces, color=(0.7, 0.7, 0.9, 0.4),
#                 edgecolor='gray', linewidth=0.2)

# # 绘制前几个点对
# num_to_show = min(5, len(pairs))
# for i in range(num_to_show):
#     p1, p2, n1, n2 = pairs[i]
#     ax.scatter(*p1, color='red', s=60, label='p1' if i == 0 else "")
#     ax.scatter(*p2, color='green', s=60, label='p2' if i == 0 else "")
#     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linestyle='--')
#     # 画法向
#     ax.quiver(p1[0], p1[1], p1[2], n1[0], n1[1], n1[2], length=0.02, color='red', alpha=0.6)
#     ax.quiver(p2[0], p2[1], p2[2], n2[0], n2[1], n2[2], length=0.02, color='green', alpha=0.6)

# # 设置等比例缩放
# def set_axes_equal(ax):
#     """让 3D 图的 xyz 比例相等"""
#     limits = np.array([
#         ax.get_xlim3d(),
#         ax.get_ylim3d(),
#         ax.get_zlim3d()
#     ])
#     span = limits[:,1] - limits[:,0]
#     centers = np.mean(limits, axis=1)
#     radius = 0.5 * max(span)
#     ax.set_xlim3d([centers[0]-radius, centers[0]+radius])
#     ax.set_ylim3d([centers[1]-radius, centers[1]+radius])
#     ax.set_zlim3d([centers[2]-radius, centers[2]+radius])

# set_axes_equal(ax)

# # 标签与图例
# ax.set_title("Antipodal Contact Pairs on LEGO Mesh")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# axrequired=True, .set_zlabel("Z")
# ax.legend()
# plt.tight_layout()
# plt.show()
