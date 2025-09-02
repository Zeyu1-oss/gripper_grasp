#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# ================== 工具函数 ==================
def unit(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def draw_triad(ax, origin, R, length=0.02, alpha=0.95):
    """在 origin 用旋转矩阵 R 画三轴 (x=红, y=绿, z=蓝)"""
    o = np.asarray(origin).ravel()
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    ax.quiver(o[0], o[1], o[2], *(length * x), color='r', alpha=alpha)
    ax.quiver(o[0], o[1], o[2], *(length * y), color='g', alpha=alpha)
    ax.quiver(o[0], o[1], o[2], *(length * z), color='b', alpha=alpha)

def set_equal_aspect_3d(ax, center, radius):
    """尽量设置 xyz 等比"""
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

def compute_scene_bounds(mesh, poses, use_pairs=True):
    """
    用 mesh 顶点 + 所有抓取中心/点对 计算统一包围盒
    poses: list of dicts, 每个含 'center','rotation','p1','p2'
    """
    V = mesh.vertices.astype(float)
    mins = V.min(axis=0)
    maxs = V.max(axis=0)

    for rec in poses:
        if 'center' in rec:
            c = np.asarray(rec['center'], float)
            mins = np.minimum(mins, c); maxs = np.maximum(maxs, c)
        if use_pairs and ('p1' in rec and 'p2' in rec):
            p1 = np.asarray(rec['p1'], float); p2 = np.asarray(rec['p2'], float)
            mins = np.minimum(mins, np.minimum(p1, p2))
            maxs = np.maximum(maxs, np.maximum(p1, p2))

    center = 0.5 * (mins + maxs)
    span = (maxs - mins)
    radius = 0.5 * float(np.max(span))
    radius = max(radius, 1e-6)
    return center, radius

def load_poses(npy_path):
    poses = np.load(npy_path, allow_pickle=True)
    # 兼容 numpy 保存为 object 的 list
    return list(poses)
def draw_two_parallel_fingers(ax, center, R, width, finger_len=0.04, lw=3.0, color='k', along_negative_z=True):
    """
    在 palm 原点画两根平行线（虚拟手指）：
      - x 轴：夹爪开合方向（用来左右分布手指：±width/2）
      - z 轴：抓取/接近方向（手指线沿 ±z 方向延伸）
    参数：
      center        : 3D np.array，palm 中心（世界系）
      R             : 3x3 旋转矩阵，列向量分别是 [x, y, z]
      width         : 两指间距（米）
      finger_len    : 线段长度（米）
      lw            : 线宽
      color         : 线颜色
      along_negative_z : True 表示沿 -z 方向延伸（朝向工件）
    """
    c = np.asarray(center, float)
    R = np.asarray(R, float)
    x, z = R[:, 0], R[:, 2]

    # 两个手指的基点（在 x 轴两侧）
    left_base  = c - 0.5 * width * x
    right_base = c + 0.5 * width * x

    # 手指线方向（通常沿 -z 指向工件）
    dir_vec = -z if along_negative_z else z
    left_tip  = left_base  + finger_len * dir_vec
    right_tip = right_base + finger_len * dir_vec

    # 画线
    ax.plot([left_base[0],  left_tip[0]],  [left_base[1],  left_tip[1]],  [left_base[2],  left_tip[2]],
            color=color, linewidth=lw)
    ax.plot([right_base[0], right_tip[0]], [right_base[1], right_tip[1]], [right_base[2], right_tip[2]],
            color=color, linewidth=lw)

# ================== 绘图 ==================
def plot_scene(mesh, poses, picks, ortho=True, show_mesh=True, save=None,
               tip_line=True, label=True, triad_scale=0.04):
    """
    picks: 要显示的条目的索引列表（基于 poses 的顺序）
    """
    center, radius = compute_scene_bounds(mesh, [poses[i] for i in picks])
    triad_len = triad_scale * radius
    normal_len = 0.06 * radius

    fig = plt.figure(figsize=(12, 6))
    # ---- 左：3D ----
    ax3d = fig.add_subplot(121, projection='3d')
    if ortho:
        try: ax3d.set_proj_type('ortho')
        except Exception: pass

    if show_mesh:
        ax3d.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                          triangles=mesh.faces,
                          color=(0.7, 0.7, 0.9, 0.35),
                          edgecolor='gray', linewidth=0.2)

    set_equal_aspect_3d(ax3d, center, radius)

    xs, ys = [], []

    for k, idx in enumerate(picks):
        rec = poses[idx]
        c = np.asarray(rec['center'], float)
        R = np.asarray(rec['rotation'], float)
        draw_triad(ax3d, c, R, length=triad_len, alpha=0.95)

        if 'p1' in rec and 'p2' in rec and tip_line:
            p1 = np.asarray(rec['p1'], float); p2 = np.asarray(rec['p2'], float)
            ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                      color='k', linestyle='--', linewidth=1.0 if k == 0 else 0.6)
            ax3d.scatter(*p1, color='red', s=25, label='p1' if k == 0 else "")
            ax3d.scatter(*p2, color='green', s=25, label='p2' if k == 0 else "")
            width = float(np.linalg.norm(p2 - p1))
        else:
            width = np.nan

        if label:
            ax3d.text(*(c + 0.015*radius*np.array([1,0,0])),
                      f"#{idx} w={width:.3f}", color='k', fontsize=8)
        xs.append(c[0]); ys.append(c[1])

    ax3d.set_title("Grasp Poses (3D) — triads at palm centers")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    if len(picks) > 0: ax3d.legend(loc='upper right', fontsize=8)

    # ---- 右：顶视 XY 投影 ----
    ax2d = fig.add_subplot(122)
    ax2d.scatter(xs, ys, s=8, alpha=0.7)
    ax2d.set_aspect('equal', 'box')
    ax2d.set_title("Palm centers (XY projection)")
    ax2d.set_xlabel("X"); ax2d.set_ylabel("Y")
    ax2d.grid(True, ls='--', alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=220)
        print(f"已保存图片到: {save}")
    plt.show()

# ================== 主逻辑（支持随机/步长选择） ==================
def main():
    ap = argparse.ArgumentParser("可视化 grasp 6D 位姿（palm triad + p1/p2）")
    ap.add_argument("--poses_path", type=str, default="../results/6d/grasp_poses.npy",
                    help="由 GraspPoseGenerator 生成的 6D 位姿文件")
    ap.add_argument("--mesh_path", type=str, default="../lego.obj")
    ap.add_argument("--scale", type=float, default=0.01, help="可选缩放 mesh")
    ap.add_argument("--max_show", type=int, default=20, help="最多显示多少条")
    ap.add_argument("--stride", type=int, default=1, help="步长抽样（例如每隔多少条取一个）")
    ap.add_argument("--random", type=int, default=0, help="随机抽样数量（>0 生效，优先于 stride/max_show）")
    ap.add_argument("--ortho", action="store_true", help="正交投影（默认透视）")
    ap.add_argument("--hide_mesh", action="store_true", help="不画 mesh, 只画抓取")
    ap.add_argument("--save", type=str, default=None, help="保存图片路径（可选）")
    ap.add_argument("--no_tipline", action="store_true", help="不画 p1-p2 连线")
    ap.add_argument("--triad_scale", type=float, default=0.04, help="坐标轴长度比例（相对场景半径）")
    args = ap.parse_args()

    # 读取 mesh
    mesh = trimesh.load(args.mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if args.scale and args.scale != 1.0:
        mesh.apply_scale(args.scale)

    # 读取 poses
    poses = load_poses(args.poses_path)
    if len(poses) == 0:
        print("⚠️ poses 为空")
        return

    # 选择要显示的索引
    all_idx = np.arange(len(poses))
    if args.random and args.random > 0:
        rng = np.random.default_rng(123)
        picks = rng.choice(all_idx, size=min(args.random, len(poses)), replace=False)
    else:
        picks = all_idx[::max(1, args.stride)]
        picks = picks[:min(args.max_show, len(picks))]

    print(f"将显示 {len(picks)} 条 (总 {len(poses)})")

    plot_scene(mesh, poses, list(picks),
               ortho=args.ortho,
               show_mesh=not args.hide_mesh,
               save=args.save,
               tip_line=not args.no_tipline,
               triad_scale=args.triad_scale)

if __name__ == "__main__":
    main()
