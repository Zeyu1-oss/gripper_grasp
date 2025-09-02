#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# ================== 工具 ==================

def unit(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def set_equal_aspect_3d(ax, center, radius):
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)

def draw_triad(ax, origin, R, length=0.02, alpha=0.95):
    o = np.asarray(origin).ravel()
    x, y, z = R[:,0], R[:,1], R[:,2]
    ax.quiver(o[0], o[1], o[2], *(length*x), color='r', alpha=alpha)
    ax.quiver(o[0], o[1], o[2], *(length*y), color='g', alpha=alpha)
    ax.quiver(o[0], o[1], o[2], *(length*z), color='b', alpha=alpha)

def draw_two_parallel_fingers(ax, center, R, width, finger_len=0.04, lw=3.0,
                              color='k', along_negative_z=True):
    """
    在 palm 原点处画两根平行线（虚拟手指）：
      - x 轴：夹爪开合方向（左右分布：±width/2）
      - z 轴：抓取/接近方向（线段沿 ±z 方向延伸）
    """
    c = np.asarray(center, float)
    R = np.asarray(R, float)
    x, z = R[:,0], R[:,2]
    left_base  = c - 0.5*width*x
    right_base = c + 0.5*width*x
    dir_vec = -z if along_negative_z else z
    left_tip  = left_base  + finger_len*dir_vec
    right_tip = right_base + finger_len*dir_vec

    ax.plot([left_base[0],  left_tip[0]],
            [left_base[1],  left_tip[1]],
            [left_base[2],  left_tip[2]], color=color, linewidth=lw)
    ax.plot([right_base[0], right_tip[0]],
            [right_base[1], right_tip[1]],
            [right_base[2], right_tip[2]], color=color, linewidth=lw)

def load_pairs_with_meta(npy_path):
    arr = np.load(npy_path, allow_pickle=True)
    meta = None
    # 末尾是 meta 的情况
    if len(arr)>0 and isinstance(arr[-1], dict) and arr[-1].get("__meta__", False):
        meta = arr[-1]
        arr = arr[:-1]
    return list(arr), meta

def robust_pair_to_pose(record, palm_offset=0.0, world_up=np.array([0.,0.,1.])):
    """
    统一从 record 里得到 (center, R, width, p1, p2, n1, n2)
    - 若已含 'center'/'rotation' 就直接使用；
    - 否则用 p1/p2 构造：x = p2-p1；y = world_up×x；z = x×y；可沿 z 方向加 palm_offset。
    """
    p1 = p2 = n1 = n2 = None
    if isinstance(record, dict):
        if "center" in record and ("rotation" in record or "R" in record):
            R = np.asarray(record.get("rotation", record.get("R")), float)
            center = np.asarray(record["center"], float)
            # 如果也有点对，带上 width
            if "p1" in record and "p2" in record:
                p1 = np.asarray(record["p1"], float)
                p2 = np.asarray(record["p2"], float)
            width = float(np.linalg.norm(p2-p1)) if (p1 is not None and p2 is not None) else None
            return center, R, width, p1, p2, record.get("n1", None), record.get("n2", None)
        # p1/p2/... 情况
        p1 = np.asarray(record["p1"], float) if "p1" in record else None
        p2 = np.asarray(record["p2"], float) if "p2" in record else None
        n1 = np.asarray(record["n1"], float) if "n1" in record else None
        n2 = np.asarray(record["n2"], float) if "n2" in record else None
    else:
        # tuple/list: (p1,p2,n1,n2)
        p1, p2, n1, n2 = record
        p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
        n1 = np.asarray(n1, float); n2 = np.asarray(n2, float)

    if p1 is None or p2 is None:
        raise ValueError("record 中缺少 p1/p2，无法构造姿态")

    x = unit(p2 - p1)
    up = world_up.astype(float)
    # 如果 x 与 up 近平行，换一个 up
    if abs(np.dot(x, up)) > 0.98:
        up = np.array([0.,1.,0.], float)
        if abs(np.dot(x, up)) > 0.98:
            up = np.array([1.,0.,0.], float)
    y = unit(np.cross(up, x))
    z = unit(np.cross(x, y))
    R = np.column_stack([x, y, z])
    center = 0.5*(p1+p2) + palm_offset*z
    width = float(np.linalg.norm(p2 - p1))
    return center, R, width, p1, p2, n1, n2

def compute_scene_bounds(mesh, centers, extra_pts=None):
    V = mesh.vertices
    mins = V.min(axis=0).astype(float)
    maxs = V.max(axis=0).astype(float)
    if centers is not None and len(centers)>0:
        C = np.asarray(centers, float)
        mins = np.minimum(mins, C.min(axis=0))
        maxs = np.maximum(maxs, C.max(axis=0))
    if extra_pts is not None and len(extra_pts)>0:
        P = np.asarray(extra_pts, float)
        mins = np.minimum(mins, P.min(axis=0))
        maxs = np.maximum(maxs, P.max(axis=0))
    center = 0.5*(mins+maxs)
    span = (maxs - mins)
    radius = 0.5*float(np.max(span))
    radius = max(radius, 1e-6)
    return center, radius

# ================== 主绘制 ==================

def plot_scene(mesh, records, picks,
               finger_len=0.04, palm_offset=0.0, width_override=None,
               ortho=True, show_mesh=True, save=None,
               draw_normals=True, draw_pose=True, draw_fingers=True,
               along_negative_z=True):
    centers, widths, extra_pts = [], [], []
    for idx in picks:
        rec = records[idx]
        try:
            center, R, width, p1, p2, n1, n2 = robust_pair_to_pose(rec, palm_offset=palm_offset)
        except Exception as e:
            print(f"索引 {idx} 解析失败: {e}")
            continue
        centers.append(center)
        if width is not None: widths.append(width)
        if p1 is not None and p2 is not None:
            extra_pts.append(p1); extra_pts.append(p2)

    scene_c, scene_r = compute_scene_bounds(mesh, centers, extra_pts=extra_pts)
    triad_len = 0.08*scene_r
    normal_len = 0.06*scene_r

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    if ortho:
        try: ax.set_proj_type('ortho')
        except Exception: pass

    if show_mesh:
        ax.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                        triangles=mesh.faces,
                        color=(0.7,0.7,0.9,0.35),
                        edgecolor='gray', linewidth=0.2)

    # 逐个画
    for k, idx in enumerate(picks):
        rec = records[idx]
        try:
            center, R, width, p1, p2, n1, n2 = robust_pair_to_pose(rec, palm_offset=palm_offset)
        except Exception as e:
            print(f"索引 {idx} 解析失败: {e}")
            continue

        # 抓取线与点
        if p1 is not None and p2 is not None:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color='k', linestyle='--', linewidth=1.2 if k==0 else 0.8)
            ax.scatter(*p1, c='r', s=50 if k==0 else 30)
            ax.scatter(*p2, c='g', s=50 if k==0 else 30)
            if draw_normals and (n1 is not None) and (n2 is not None):
                ax.quiver(p1[0], p1[1], p1[2], *(normal_len*unit(n1)), color='r', alpha=0.9)
                ax.quiver(p2[0], p2[1], p2[2], *(normal_len*unit(n2)), color='g', alpha=0.9)

        # 坐标系
        if draw_pose:
            draw_triad(ax, center, R, length=triad_len, alpha=0.95)

        # 虚拟手指线
        if draw_fingers:
            w_use = (width_override if (width_override is not None) else
                     (width if width is not None else 0.03))
            draw_two_parallel_fingers(ax, center, R, width=w_use,
                                      finger_len=finger_len, lw=3.0,
                                      color='k', along_negative_z=along_negative_z)

        # 终端打印
        print(f"\n=== 抓取 #{idx} ===")
        print(f"center = {center}")
        print(f"R =\n{R}")
        if width is not None: print(f"width = {width:.6f}")
        if p1 is not None: print(f"p1 = {p1}")
        if p2 is not None: print(f"p2 = {p2}")

    set_equal_aspect_3d(ax, scene_c, scene_r)
    ax.set_title("Grasps with virtual fingers at palm center")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
        print(f"已保存图片到: {save}")
    plt.show()

# ================== CLI ==================

def main():
    ap = argparse.ArgumentParser("可视化抓取 + 虚拟手指")
    ap.add_argument("--npy_path",  type=str,
                    default="../results/antipodal_pairs/lego_pairs.npy",
                    help="保存的 pairs 或 6D 位姿 npy")
    ap.add_argument("--mesh_path", type=str, default="../lego.obj")
    ap.add_argument("--default_scale", type=float, default=0.01,
                    help="当 npy 没有 meta 时对 mesh 施加的缩放")
    # index 选择
    ap.add_argument("--idx", type=int, default=None, help="单个编号")
    ap.add_argument("--indices", type=str, default=None, help="多个编号, 逗号分隔")
    # 虚拟手指 & 位姿
    ap.add_argument("--finger_len", type=float, default=0.04, help="手指线长度(米)")
    ap.add_argument("--palm_offset", type=float, default=0.0, help="沿 z 轴的偏移(米)")
    ap.add_argument("--width_override", type=float, default=None, help="覆盖手指间距(米)")
    ap.add_argument("--along_neg_z", action="store_true", help="手指线沿 -Z 方向（默认 False 为 +Z）")
    # 显示
    ap.add_argument("--persp", action="store_true", help="使用透视投影（默认正交）")
    ap.add_argument("--hide_mesh", action="store_true", help="不画 mesh")
    ap.add_argument("--save", type=str, default=None, help="保存图片路径")
    args = ap.parse_args()

    # 1) 读 npy
    records, meta = load_pairs_with_meta(args.npy_path)
    if len(records) == 0:
        print("❌ NPY 里没有记录"); return

    # 2) 读 mesh
    mesh = trimesh.load(args.mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    # 如果有 meta（例如 T/scale），尽量复现
    if meta is not None:
        scale = float(meta.get("scale", 1.0))
        if scale != 1.0: mesh.apply_scale(scale)
        T = meta.get("T_mesh_4x4", None)
        if T is not None:
            mesh.apply_transform(np.asarray(T, dtype=float))
        print("已应用 npy meta 的 mesh 缩放/变换复现。")
    else:
        if args.default_scale and args.default_scale != 1.0:
            mesh.apply_scale(args.default_scale)

    # 3) 选择要画的索引
    if args.indices:
        picks = [int(x) for x in args.indices.split(',') if x.strip()!=""]
    elif args.idx is not None:
        picks = [int(args.idx)]
    else:
        # 默认画前几个
        picks = list(range(min(8, len(records))))

    # 越界裁剪
    picks = [i for i in picks if 0 <= i < len(records)]
    if len(picks) == 0:
        print("索引不合法，默认显示第 0 个。")
        picks = [0]

    # 4) 画
    plot_scene(mesh, records, picks,
               finger_len=args.finger_len,
               palm_offset=args.palm_offset,
               width_override=args.width_override,
               ortho=not args.persp,
               show_mesh=not args.hide_mesh,
               save=args.save,
               along_negative_z=args.along_neg_z)

if __name__ == "__main__":
    main()
