#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- utils ----------
def unit(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def rot_to_quat_wxyz(R):
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], float)
    q /= (np.linalg.norm(q) + 1e-12)
    return q

def rot_to_euler_zyx(R):
    R = np.asarray(R, float)
    sy = -R[2,0]
    cy = np.sqrt(max(0.0, 1.0 - sy*sy))
    if cy > 1e-8:
        yaw   = np.arctan2(R[1,0], R[0,0])
        pitch = np.arcsin(sy)
        roll  = np.arctan2(R[2,1], R[2,2])
    else:
        yaw   = np.arctan2(-R[0,1], R[1,1])
        pitch = np.arcsin(sy)
        roll  = 0.0
    return np.array([yaw, pitch, roll], float)

def draw_triad(ax, origin, R, length=0.02, alpha=0.95):
    o = np.asarray(origin).ravel()
    x, y, z = R[:,0], R[:,1], R[:,2]
    ax.quiver(o[0], o[1], o[2], *(length*x), color='r', alpha=alpha)
    ax.quiver(o[0], o[1], o[2], *(length*y), color='g', alpha=alpha)
    ax.quiver(o[0], o[1], o[2], *(length*z), color='b', alpha=alpha)

def draw_two_parallel_fingers(ax, center, R, width, finger_len=0.04,
                              color='k', lw=3.0, along_negative_z=True):
    c = np.asarray(center, float)
    x, z = R[:,0], R[:,2]
    left_base  = c - 0.5*width*x
    right_base = c + 0.5*width*x
    dir_vec = -z if along_negative_z else z
    left_tip  = left_base  + finger_len * dir_vec
    right_tip = right_base + finger_len * dir_vec
    ax.plot([left_base[0], left_tip[0]],
            [left_base[1], left_tip[1]],
            [left_base[2], left_tip[2]], color=color, linewidth=lw)
    ax.plot([right_base[0], right_tip[0]],
            [right_base[1], right_tip[1]],
            [right_base[2], right_tip[2]], color=color, linewidth=lw)

def draw_palm_box(ax, center, R, width, height, thickness,
                  color=(0.2,0.2,0.25), edge=(0,0,0), alpha=0.5):
    """
    在 (center,R) 画一个薄盒子作为 palm：
      - 它的局部 x 轴沿 R[:,0]（开合方向），局部 y 轴沿 R[:,1]，法向是 R[:,2]
      - width 沿 x，height 沿 y，thickness 沿 z
    """
    cx, cy, cz = width/2.0, height/2.0, thickness/2.0
    # 8 顶点（局部坐标）
    corners_local = np.array([
        [+cx,+cy,+cz], [+cx,-cy,+cz], [-cx,-cy,+cz], [-cx,+cy,+cz],  # 上面 (+z)
        [+cx,+cy,-cz], [+cx,-cy,-cz], [-cx,-cy,-cz], [-cx,+cy,-cz],  # 下面 (-z)
    ], float)
    # 变到世界系
    R = np.asarray(R, float)
    C = np.asarray(center, float)
    corners = (corners_local @ R.T) + C

    # 六个面
    faces = [
        [corners[i] for i in [0,1,2,3]],  # +z
        [corners[i] for i in [4,5,6,7]],  # -z
        [corners[i] for i in [0,1,5,4]],  # +x
        [corners[i] for i in [2,3,7,6]],  # -x
        [corners[i] for i in [1,2,6,5]],  # -y
        [corners[i] for i in [3,0,4,7]],  # +y
    ]
    poly = Poly3DCollection(faces, facecolors=[color], edgecolors=[edge], linewidths=0.5, alpha=alpha)
    ax.add_collection3d(poly)

def set_equal_aspect_3d(ax, center, radius):
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)

def load_pairs_with_meta(npy_path):
    arr = np.load(npy_path, allow_pickle=True)
    meta = None
    if len(arr)>0 and isinstance(arr[-1], dict) and arr[-1].get("__meta__", False):
        meta = arr[-1]; arr = arr[:-1]
    return list(arr), meta

def robust_pair_to_pose(rec, palm_offset=0.0, world_up=np.array([0.,0.,1.])):
    """
    输入可以是：
      - dict: {p1,p2,n1,n2,...} 或 {center, rotation/R, width?}
      - tuple/list: (p1,p2,n1,n2)
    输出：(center, R, width, p1, p2)
    """
    p1 = p2 = None
    if isinstance(rec, dict):
        if "center" in rec and ("rotation" in rec or "R" in rec):
            R = np.asarray(rec.get("rotation", rec.get("R")), float)
            center = np.asarray(rec["center"], float)
            if "p1" in rec and "p2" in rec:
                p1 = np.asarray(rec["p1"], float)
                p2 = np.asarray(rec["p2"], float)
                width = float(np.linalg.norm(p2 - p1))
            else:
                width = rec.get("width", None)
            return center, R, width, p1, p2
        # 只有 p1/p2 的情况
        p1 = np.asarray(rec["p1"], float)
        p2 = np.asarray(rec["p2"], float)
    else:
        p1, p2 = rec[0], rec[1]
        p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)

    # 用 p1/p2 推出姿态：x=开合，z=接近
    x = unit(p2 - p1)
    up = world_up.astype(float)
    if abs(np.dot(x, up)) > 0.98:
        up = np.array([0.,1.,0.], float)
        if abs(np.dot(x, up)) > 0.98:
            up = np.array([1.,0.,0.], float)
    y = unit(np.cross(up, x))
    z = unit(np.cross(x, y))
    R = np.column_stack([x, y, z])
    center = 0.5*(p1+p2) + palm_offset * z
    width = float(np.linalg.norm(p2 - p1))
    return center, R, width, p1, p2

# ---------- main vis ----------
def plot_grasps(npy_path, mesh_path, indices=None,
                default_scale=1.0, palm_offset=0.0, finger_len=0.04,
                width_override=None, along_negative_z=True,
                show_palm=True, palm_w_margin=0.15, palm_h=None, palm_thickness=None,
                palm_alpha=0.45,
                ortho=True, show_mesh=True, save=None):
    records, meta = load_pairs_with_meta(npy_path)
    if len(records) == 0:
        print("❌ NPY 里没有抓取记录"); return

    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    # 应用保存时的 meta（若存在）
    if meta is not None:
        sc = float(meta.get("scale", 1.0))
        if sc != 1.0: mesh.apply_scale(sc)
        T = meta.get("T_mesh_4x4", None)
        if T is not None: mesh.apply_transform(np.asarray(T, float))
    else:
        if default_scale and default_scale != 1.0:
            mesh.apply_scale(default_scale)

    # 选取索引
    if indices is None:
        picks = list(range(min(8, len(records))))
    else:
        picks = [i for i in indices if 0 <= i < len(records)]
        if len(picks) == 0:
            print("索引不合法，默认显示第 0 个"); picks=[0]

    # 场景边界
    centers = []
    for i in picks:
        c, R, w, p1, p2 = robust_pair_to_pose(records[i], palm_offset=palm_offset)
        centers.append(c)
    V = mesh.vertices
    mins = np.minimum(V.min(axis=0), np.min(centers, axis=0))
    maxs = np.maximum(V.max(axis=0), np.max(centers, axis=0))
    scene_c = 0.5*(mins+maxs)
    scene_r = 0.5*np.max(maxs - mins)

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    if ortho:
        try: ax.set_proj_type('ortho')
        except Exception: pass

    if show_mesh:
        ax.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                        triangles=mesh.faces, color=(0.7,0.7,0.9,0.35),
                        edgecolor='gray', linewidth=0.2)

    # 逐条画
    for i in picks:
        rec = records[i]
        center, R, width, p1, p2 = robust_pair_to_pose(rec, palm_offset=palm_offset)

        # 打印 6D
        q = rot_to_quat_wxyz(R)
        ypr = rot_to_euler_zyx(R)
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = center
        print(f"\n=== Grasp #{i} ===")
        print(f"center (xyz): {center}")
        print(f"rotation (3x3):\n{R}")
        print(f"quat (wxyz): {q}")
        print(f"euler ZYX (rad): yaw={ypr[0]:.6f}, pitch={ypr[1]:.6f}, roll={ypr[2]:.6f}")
        print(f"T (4x4):\n{T}")
        if p1 is not None and p2 is not None:
            print(f"p1={p1}, p2={p2}, width={width:.6f}")

        # p1/p2/抓取线
        if p1 is not None and p2 is not None:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color='k', linestyle='--', linewidth=1.0)
            ax.scatter(*p1, c='r', s=40)
            ax.scatter(*p2, c='g', s=40)

        # 坐标系
        draw_triad(ax, center, R, length=0.12*scene_r, alpha=0.95)

        # 两根手指线
        w_use = width_override if (width_override is not None) else (width if width is not None else 0.03)
        draw_two_parallel_fingers(ax, center, R, width=w_use,
                                  finger_len=finger_len, lw=3.5,
                                  color='k', along_negative_z=along_negative_z)

        # palm 盒子（真正让“palm 看得见”）
        if show_palm:
            # 默认 palm 高度/厚度：跟场景尺度/指距走
            ph = palm_h if palm_h is not None else (0.18 * scene_r)
            pt = palm_thickness if palm_thickness is not None else (0.06 * scene_r)
            pw = w_use * (1.0 + palm_w_margin)  # 略比指距更宽一点
            draw_palm_box(ax, center, R, width=pw, height=ph, thickness=pt,
                          color=(0.25,0.25,0.3), edge=(0,0,0), alpha=palm_alpha)

    set_equal_aspect_3d(ax, scene_c, max(scene_r, 1e-6))
    ax.set_title("6D poses + two finger lines + palm box")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=220); print(f"已保存图片: {save}")
    plt.show()

# ---------- CLI ----------
def build_argparser():
    ap = argparse.ArgumentParser("显示 6D 位姿 + 两根手指线 + palm 盒子")
    ap.add_argument("--npy_path",  type=str, default="../results/antipodal_pairs/lego_pairs.npy")
    ap.add_argument("--mesh_path", type=str, default="../lego.obj")
    ap.add_argument("--default_scale", type=float, default=0.01,
                    help="当 npy 无 meta 时对 mesh 的缩放")
    ap.add_argument("--indices", type=str, default=None,
                    help="要查看的编号，逗号分隔，例如 1,2,3")
    ap.add_argument("--palm_offset", type=float, default=0.0,
                    help="沿抓取 z 轴偏移（米）")
    ap.add_argument("--finger_len", type=float, default=0.04,
                    help="手指线长度（米）")
    ap.add_argument("--width_override", type=float, default=None,
                    help="覆盖手指间距（米）；不设则用 |p2-p1|")
    ap.add_argument("--along_neg_z", action="store_true",
                    help="手指线沿 -Z 方向（默认 +Z）")
    ap.add_argument("--hide_mesh", action="store_true",
                    help="不渲染 mesh")
    ap.add_argument("--persp", action="store_true",
                    help="使用透视投影（默认正交）")
    # palm 外观参数
    ap.add_argument("--no_palm", action="store_true",
                    help="不画 palm 盒子")
    ap.add_argument("--palm_w_margin", type=float, default=0.15,
                    help="palm 宽度相对指距的额外比例 (e.g. 0.15 => +15%)")
    ap.add_argument("--palm_h", type=float, default=None,
                    help="palm 高度（米），默认随场景大小")
    ap.add_argument("--palm_t", type=float, default=None,
                    help="palm 厚度（米），默认随场景大小")
    ap.add_argument("--palm_alpha", type=float, default=0.45,
                    help="palm 透明度")
    ap.add_argument("--save", type=str, default=None,
                    help="保存图片路径")
    return ap

def main():
    args = build_argparser().parse_args()
    indices = None
    if args.indices:
        indices = [int(x) for x in args.indices.split(',') if x.strip()!=""]
    plot_grasps(
        npy_path=args.npy_path,
        mesh_path=args.mesh_path,
        indices=indices,
        default_scale=args.default_scale,
        palm_offset=args.palm_offset,
        finger_len=args.finger_len,
        width_override=args.width_override,
        along_negative_z=args.along_neg_z,
        show_palm=(not args.no_palm),
        palm_w_margin=args.palm_w_margin,
        palm_h=args.palm_h,
        palm_thickness=args.palm_t,
        palm_alpha=args.palm_alpha,
        ortho=(not args.persp),
        show_mesh=(not args.hide_mesh),
        save=args.save
    )

if __name__ == "__main__":
    main()
