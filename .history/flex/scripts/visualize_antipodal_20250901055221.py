#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# ================== 工具函数 ==================

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
    """
    统一取出 (p1,p2,n1,n2,center,R)。
    若 rec 字典里已有 center/R 就直接用；否则按 (p1,p2,n1,n2) 现场构造。
    """
    if isinstance(rec, dict):
        p1, p2 = np.asarray(rec["p1"], float), np.asarray(rec["p2"], float)
        n1, n2 = np.asarray(rec["n1"], float), np.asarray(rec["n2"], float)
        center = rec.get("center", None)
        R = rec.get("R", None)
    else:
        p1, p2, n1, n2 = rec
        p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
        n1 = np.asarray(n1, float); n2 = np.asarray(n2, float)
        center, R = None, None

    if center is None or R is None:
        x = unit(p2 - p1)                 # 夹爪闭合方向
        nz = n1 + n2
        if np.linalg.norm(nz) < 1e-8:
            tmp = np.array([0., 0., 1.]) if abs(x[2]) < 0.9 else np.array([0., 1., 0.])
            z = unit(np.cross(tmp, x))
        else:
            z = -unit(nz)                 # 手爪“朝向”取反法向平均
            if abs(np.dot(z, x)) > 0.99:  # 退化处理
                tmp = np.array([0., 0., 1.]) if abs(x[2]) < 0.9 else np.array([0., 1., 0.])
                z = unit(np.cross(tmp, x))
        y = unit(np.cross(z, x))
        z = unit(np.cross(x, y))
        R = np.column_stack([x, y, z])
        center = 0.5 * (p1 + p2)

    return p1, p2, n1, n2, center, R

def compute_scene_bounds(mesh, picks, pairs):
    """
    用 mesh 顶点 + 被选中的点对来算统一包围盒
    picks: 选择的索引列表
    """
    V = mesh.vertices
    mins = V.min(axis=0).astype(float)
    maxs = V.max(axis=0).astype(float)

    for idx in picks:
        if 0 <= idx < len(pairs):
            rec = pairs[idx]
            if isinstance(rec, dict):
                p1 = np.asarray(rec['p1'], float)
                p2 = np.asarray(rec['p2'], float)
            else:
                p1, p2, _, _ = rec
                p1 = np.asarray(p1, float)
                p2 = np.asarray(p2, float)
            mins = np.minimum(mins, np.minimum(p1, p2))
            maxs = np.maximum(maxs, np.maximum(p1, p2))

    center = 0.5 * (mins + maxs)
    span = (maxs - mins)
    radius = 0.5 * float(np.max(span))
    radius = max(radius, 1e-6)
    return center, radius

def set_equal_aspect_3d(ax, center, radius):
    """尽量设置 xyz 等比"""
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

# ================== 主逻辑（可选编号/交互） ==================

def plot_scene(mesh, pairs, picks, ortho=True, show_mesh=True, save=None):
    # 包围盒/尺度
    center, radius = compute_scene_bounds(mesh, picks, pairs)
    triad_len  = 0.08 * radius
    normal_len = 0.06 * radius

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if ortho:
        try: ax.set_proj_type('ortho')
        except Exception: pass

    if show_mesh:
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                        triangles=mesh.faces,
                        color=(0.7, 0.7, 0.9, 0.35),
                        edgecolor='gray', linewidth=0.2)

    set_equal_aspect_3d(ax, center, radius)

    # 画选择的样本
    for k, idx in enumerate(picks):
        if idx < 0 or idx >= len(pairs):
            print(f"⚠️ 索引 {idx} 越界，跳过")
            continue
        rec = pairs[idx]
        p1, p2, n1, n2, c, R = rec_to_pose(rec)

        # 线、点
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='k', linestyle='--', linewidth=1.5 if k == 0 else 0.8)
        ax.scatter(*p1, color='red',   s=70 if k == 0 else 40, label='p1' if k == 0 else "")
        ax.scatter(*p2, color='green', s=70 if k == 0 else 40, label='p2' if k == 0 else "")

        # 法向（加上！）
        ax.quiver(p1[0], p1[1], p1[2], *(normal_len * n1), color='red',   alpha=0.8, linewidth=2 if k==0 else 1)
        ax.quiver(p2[0], p2[1], p2[2], *(normal_len * n2), color='green', alpha=0.8, linewidth=2 if k==0 else 1)

        # 在图上标注法向数值（简短）
        def short(v): return f"[{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}]"
        ax.text(*(p1 + 0.02 * radius * np.array([1,0,0])), f"#{idx} n1={short(n1)}", color='red',  fontsize=9)
        ax.text(*(p2 + 0.02 * radius * np.array([1,0,0])), f"#{idx} n2={short(n2)}", color='green',fontsize=9)

        # 画抓取坐标系
        draw_triad(ax, c, R, length=triad_len, alpha=0.95)

        # 终端详细打印
        width = float(np.linalg.norm(p2 - p1))
        print(f"\n=== 抓取 #{idx} ===")
        print(f"p1 = {p1}")
        print(f"p2 = {p2}")
        print(f"n1 = {n1}")
        print(f"n2 = {n2}")
        print(f"center = {c}, width = {width:.6f}")

    ax.set_title("Antipodal Pair(s) with Normals & Pose (XYZ Equal Scale)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    if len(picks) > 0: ax.legend(loc='upper right')
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
        print(f"已保存图片到: {save}")
    plt.show()

def run(args):
    # 读取数据
    pairs, meta = load_pairs_with_meta(args.npy_path)
    if not pairs:
        print("未找到任何数据"); return

    # 读 mesh + 应用 meta 变换
    mesh = trimesh.load(args.mesh_path, force='mesh')
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
        mesh.apply_scale(args.default_scale)

    # 选择索引
    picks = []
    if args.indices:
        picks = [int(x) for x in args.indices.split(',') if x.strip() != ""]
    elif args.idx is not None:
        picks = [int(args.idx)]
    else:
        picks = [0]  # 默认看第0个

    # 越界裁剪提示
    picks = [i for i in picks if 0 <= i < len(pairs)]
    if len(picks) == 0:
        print("选择的索引都不合法，默认显示第 0 个。")
        picks = [0]

    # 交互模式：A/D 或 ←/→ 切换
    if args.interactive:
        cur = picks[0]
        def on_key(event):
            nonlocal cur
            if event.key in ['right', 'd']:
                cur = min(cur + 1, len(pairs) - 1)
            elif event.key in ['left', 'a']:
                cur = max(cur - 1, 0)
            else:
                return
            plt.close('all')
            plot_scene(mesh, pairs, [cur], ortho=not args.persp, show_mesh=not args.hide_mesh, save=None)

        # 先画一次
        fig = plt.figure()
        plt.close(fig)
        cid = plt.gcf().canvas.mpl_connect('key_press_event', on_key)
        plot_scene(mesh, pairs, [cur], ortho=not args.persp, show_mesh=not args.hide_mesh, save=args.save)
    else:
        plot_scene(mesh, pairs, picks, ortho=not args.persp, show_mesh=not args.hide_mesh, save=args.save)

# ================== CLI ==================

def build_argparser():
    ap = argparse.ArgumentParser("可选择抓取编号的可视化（带法向/坐标系,XYZ 等比）")
    ap.add_argument("--npy_path",  type=str,
                    default="/home/rose/legograsp/flex/results/antipodal_pairs/lego_pairs.npy")
    ap.add_argument("--mesh_path", type=str, default="../lego.obj")
    ap.add_argument("--default_scale", type=float, default=0.01,
                    help="当 npy 里没有 meta 时对 mesh 施加的默认缩放")
    # 选择方式（二选一）
    ap.add_argument("--idx", type=int, default=None, help="单个编号，例如 --idx 100")
    ap.add_argument("--indices", type=str, default=None, help="多个编号，逗号分隔，例如 --indices 10,100,256")
    # 交互/显示
    ap.add_argument("--interactive", action="store_true", help="开启交互模式:A/D 或 ←/→ 切换编号")
    ap.add_argument("--persp", action="store_true", help="使用透视投影（默认正交投影）")
    ap.add_argument("--hide_mesh", action="store_true", help="不画 mesh,只画抓取/法向")
    ap.add_argument("--save", type=str, default=None, help="保存图片路径（可选）")
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
