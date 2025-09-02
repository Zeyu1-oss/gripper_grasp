#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math, argparse, numpy as np
import trimesh
from scipy.spatial import cKDTree

# ---------------- 参数 ----------------
DEFAULT_MESH = "lego.obj"   # 你的网格
SAMPLE_NUM   = 8000         # 表面随机采样数量（越大越稳）
MU           = 0.5          # 摩擦系数（决定摩擦圆锥角）
W_MIN        = 0.009        # 允许的手指间距下限(米)   ~9mm
W_MAX        = 0.040        # 允许的手指间距上限(米)   ~40mm
CLEARANCE    = 0.0015       # 手指/夹爪厚度余量(米)   ~1.5mm 走廊检验
N_TOP        = 100          # 导出前N个抓取
RAY_BIAS     = 1e-4         # 射线起点/方向数值偏置
ANGLE_TOL    = np.deg2rad(7)  # 法向/轴向对齐的容差

# ---------------- 工具 ----------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def rot_from_xy(x_hat, y_hat):
    x = unit(x_hat)
    y = unit(y_hat - np.dot(y_hat, x)*x)
    z = np.cross(x, y)
    R = np.column_stack([x, y, z])
    return R

def mat_to_quat(R):
    # w, x, y, z
    q = trimesh.transformations.quaternion_from_matrix(
        np.vstack([np.hstack([R, np.zeros((3,1))]),
                   np.array([[0,0,0,1]])])
    )
    # 转为 (x,y,z,w)
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]

def corridor_clear(mesh, p1, p2, clearance, steps=10):
    """简易‘走廊’无碰撞检查：连线中点采样，距离网格>clearance"""
    mids = np.linspace(0.05, 0.95, steps)
    seg = p2 - p1
    pts = (p1[None,:] + mids[:,None]*seg[None,:])
    try:
        # watertight 时 signed_distance 更可靠
        if mesh.is_watertight:
            d = trimesh.proximity.signed_distance(mesh, pts)
            return np.all(np.abs(d) > clearance)
        else:
            closest, _, _ = trimesh.proximity.closest_point(mesh, pts)
            dist = np.linalg.norm(pts - closest, axis=1)
            return np.all(dist > clearance)
    except Exception:
        # 退化：用最近点距离
        closest, _, _ = trimesh.proximity.closest_point(mesh, pts)
        dist = np.linalg.norm(pts - closest, axis=1)
        return np.all(dist > clearance)

# ---------------- 抓取搜索（网格 → 抓取集） ----------------
def synthesize_grasps(mesh,
                      sample_num=SAMPLE_NUM,
                      mu=MU,
                      w_min=W_MIN, w_max=W_MAX,
                      clearance=CLEARANCE,
                      angle_tol=ANGLE_TOL):
    """
    返回 grasps 列表：每个 grasp = {
        "center": [x,y,z],
        "quat_xyzw": [x,y,z,w],   # 世界系
        "width": w,               # 手指间距
        "q_score": q              # 质量分数（简易）
    }
    """
    # 表面采样
    P, face_idx = trimesh.sample.sample_surface(mesh, sample_num)
    FN = mesh.face_normals[face_idx]
    # 法向单位化
    N = np.apply_along_axis(unit, 1, FN)

    # 构造 BVH/加速结构（trimesh 内部会用 rtree/pyembree 如果有）
    r = mesh.ray

    grasps = []

    # 预计算摩擦圆锥半角
    cone_half = math.atan(mu)

    for i in range(len(P)):
        p  = P[i]
        n  = N[i]

        # 从 p 沿 -n 打射线，找“对向”接触（夹爪另一指）
        origin = p - n * RAY_BIAS
        direc  = -n
        try:
            hits, index_ray, index_tri = r.intersects_id(
                ray_origins=origin[None,:], ray_directions=direc[None,:],
                return_id=True
            )
        except Exception:
            # 有些环境用 intersects_any
            hits = r.intersects_any(
                ray_origins=origin[None,:], ray_directions=direc[None,:]
            )
            index_tri = None

        if (hasattr(hits, "__len__") and len(hits)==0) or (hits is False):
            continue

        # 取最近一次命中
        locs, _, tri_ids = r.intersects_location(
            ray_origins=origin[None,:], ray_directions=direc[None,:]
        )
        if len(locs)==0:
            continue
        pj = locs[0]           # 对向点
        tri = tri_ids[0]
        nj  = unit(mesh.face_normals[tri])

        w = np.linalg.norm(pj - p)
        if not (w_min <= w <= w_max):
            continue

        g_axis = unit(pj - p)         # 夹爪闭合方向（手指连线方向）
        # 反摩擦圆锥 / 对向性条件：
        # 接触法向要落在以 ±g_axis 为轴的摩擦圆锥内
        # 即 angle(n_i,  g_axis) <= cone_half 且 angle(n_j, -g_axis) <= cone_half
        a1 = math.acos(np.clip(np.dot(n,  g_axis), -1.0, 1.0))
        a2 = math.acos(np.clip(np.dot(nj, -g_axis), -1.0, 1.0))
        if (a1 > cone_half + angle_tol) or (a2 > cone_half + angle_tol):
            continue

        # 夹爪“接近/法向”方向，用两侧法向的和（指向外，进给方向反向）
        approach = unit(-(n + nj))
        if np.linalg.norm(n + nj) < 1e-6:
            # 法向几乎完全对消：用 g_axis 的法向空间里取一个稳定方向
            # 构造任意与 g_axis 不共线的向量
            tmp = np.array([1,0,0]) if abs(g_axis[0]) < 0.9 else np.array([0,1,0])
            approach = unit(np.cross(g_axis, tmp))

        # 构造抓取坐标系（Panda/通用：x=横向(副法线)，y=闭合方向，z=接近方向）
        y_hat = g_axis
        z_hat = approach
        x_hat = unit(np.cross(y_hat, z_hat))
        z_hat = unit(np.cross(x_hat, y_hat))  # 正交修正

        # 简易“走廊”无碰撞检查（手指间）
        if not corridor_clear(mesh, p, pj, clearance):
            continue

        center = 0.5*(p + pj)
        R = np.column_stack([x_hat, y_hat, z_hat])
        quat = mat_to_quat(R)

        # 简易质量：越接近锥内、法向对称越高
        fc_margin = max(0.0, cone_half - max(a1, a2))
        normal_align = 1.0 - abs(np.dot(n, nj))  # 反向越大越好（接近 -1）
        q = 0.5*fc_margin/cone_half + 0.5*normal_align

        grasps.append({
            "center": center.tolist(),
            "quat_xyzw": quat,
            "width": float(w),
            "q_score": float(q)
        })

    # 质量排序 & 去重（相近中心合并）
    grasps.sort(key=lambda g: g["q_score"], reverse=True)

    # 粗略去重：基于中心位置网格化
    kept = []
    grid = set()
    voxel = 0.003  # 3mm
    for g in grasps:
        c = np.array(g["center"])
        key = tuple(np.floor(c/voxel).astype(int))
        if key in grid:
            continue
        grid.add(key)
        kept.append(g)

    return kept

# ---------------- 主程序 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=str, default=DEFAULT_MESH)
    ap.add_argument("--out",  type=str, default="grasps.json")
    ap.add_argument("--n",    type=int, default=SAMPLE_NUM)
    ap.add_argument("--mu",   type=float, default=MU)
    ap.add_argument("--wmin", type=float, default=W_MIN)
    ap.add_argument("--wmax", type=float, default=W_MAX)
    ap.add_argument("--top",  type=int, default=N_TOP)
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    grasps = synthesize_grasps(mesh,
                               sample_num=args.n,
                               mu=args.mu,
                               w_min=args.wmin, w_max=args.wmax)

    grasps = grasps[:args.top]
    with open(args.out, "w") as f:
        json.dump({"mesh": args.mesh, "grasps": grasps}, f, indent=2)

    print(f"[OK] total={len(grasps)} saved → {args.out}")
    if len(grasps):
        g0 = grasps[0]
        print("[TOP1] center=", np.round(g0["center"],4),
              " quat(xyzw)=", np.round(g0["quat_xyzw"],4),
              " width(m)=", round(g0["width"],4),
              " score=", round(g0["q_score"],3))

if __name__ == "__main__":
    main()
