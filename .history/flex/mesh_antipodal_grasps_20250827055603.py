#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math, argparse, numpy as np
import trimesh
from trimesh import repair

# ================= 参数 =================
DEFAULT_MESH = "lego.obj"
SAMPLE_NUM   = 12000
MU           = 0.6
W_MIN        = -1.0     # 传负数 => 自动按尺寸估计
W_MAX        = -1.0
CLEARANCE    = 0.0015
N_TOP        = 200
RAY_BIAS     = 1e-4
ANGLE_TOL    = np.deg2rad(8.0)

def unit(v):
    n = np.linalg.norm(v)
    return v/n if n>1e-12 else v

def rot_from_xy(x_hat, y_hat):
    x = unit(x_hat)
    y = unit(y_hat - np.dot(y_hat, x)*x)
    z = np.cross(x, y)
    R = np.column_stack([x, y, z])
    return R

def mat_to_quat(R):
    H = np.eye(4); H[:3,:3] = R
    q = trimesh.transformations.quaternion_from_matrix(H)  # wxyz
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]  # xyzw

def corridor_clear(mesh, p1, p2, clearance, steps=10):
    ts  = np.linspace(0.05, 0.95, steps)
    seg = p2 - p1
    pts = p1[None,:] + ts[:,None]*seg[None,:]
    try:
        if mesh.is_watertight:
            d = trimesh.proximity.signed_distance(mesh, pts)
            return np.all(np.abs(d) > clearance)
    except Exception:
        pass
    closest, _, _ = trimesh.proximity.closest_point(mesh, pts)
    dist = np.linalg.norm(pts - closest, axis=1)
    return np.all(dist > clearance)

def ray_hits(mesh, origin, direction):
    # 返回沿 direction 的最近命中点与其三角形 id；没有命中返回 (None, None)
    try:
        locs, ray_id, tri_id = mesh.ray.intersects_location(
            ray_origins=origin[None,:], ray_directions=direction[None,:]
        )
        if len(locs)==0:
            return None, None
        return locs[0], tri_id[0]
    except Exception:
        ok = mesh.ray.intersects_any(
            ray_origins=origin[None,:], ray_directions=direction[None,:]
        )
        if not ok: return None, None
        # 再取位置
        locs, _, tri_id = mesh.ray.intersects_location(
            ray_origins=origin[None,:], ray_directions=direction[None,:]
        )
        if len(locs)==0: return None, None
        return locs[0], tri_id[0]

def opposite_contact(mesh, p, n, wmin, wmax):
    """
    从点 p、法向 n 出发，分别沿 (-n) 和 (+n) 双向打射线，返回
    最符合宽度约束的对向接触 (pj, nj)。若无，则 None。
    """
    cands = []
    for s in (-1.0, +1.0):
        d  = s * n
        o  = p + d * RAY_BIAS   # 往射线方向略偏移，避免命中自身面
        hit, tri = ray_hits(mesh, o, d)
        if hit is None: 
            continue
        w = np.linalg.norm(hit - p)
        if wmin <= w <= wmax:
            nj = unit(mesh.face_normals[tri])
            cands.append((w, hit, nj))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])  # 优先最近那一个
    _, pj, nj = cands[0]
    return pj, nj

def synthesize_grasps(mesh, sample_num, mu, wmin, wmax, clearance, angle_tol):
    P, F = trimesh.sample.sample_surface(mesh, sample_num)
    FN   = mesh.face_normals[F]
    N    = np.apply_along_axis(unit, 1, FN)

    cone_half = math.atan(mu)
    grasps = []

    for i in range(len(P)):
        p, n = P[i], N[i]
        opp = opposite_contact(mesh, p, n, wmin, wmax)
        if opp is None: 
            continue
        pj, nj = opp
        g_axis = unit(pj - p)
        # 反摩擦圆锥对向性
        a1 = math.acos(np.clip(np.dot(n,  g_axis), -1.0, 1.0))
        a2 = math.acos(np.clip(np.dot(nj, -g_axis), -1.0, 1.0))
        if (a1 > cone_half + angle_tol) or (a2 > cone_half + angle_tol):
            continue

        # 接近方向：用 -(n + nj)，退化时与 g_axis 正交
        if np.linalg.norm(n + nj) > 1e-6:
            z_hat = unit(-(n + nj))
        else:
            tmp = np.array([1,0,0]) if abs(g_axis[0])<0.9 else np.array([0,1,0])
            z_hat = unit(np.cross(g_axis, tmp))

        y_hat = g_axis
        x_hat = unit(np.cross(y_hat, z_hat))
        z_hat = unit(np.cross(x_hat, y_hat))  # 正交化

        if not corridor_clear(mesh, p, pj, clearance):
            continue

        center = 0.5*(p + pj)
        R = np.column_stack([x_hat, y_hat, z_hat])
        q = mat_to_quat(R)
        w = float(np.linalg.norm(pj - p))

        # 简易质量：锥裕度 + 法向对称
        fc_margin = max(0.0, cone_half - max(a1, a2))
        normal_sym = (1.0 - abs(np.dot(n, nj)))  # 越接近 -1 越好
        score = 0.6*(fc_margin/(cone_half+1e-9)) + 0.4*normal_sym

        grasps.append({
            "center": center.tolist(),
            "quat_xyzw": q,
            "width": w,
            "q_score": float(score)
        })

    # 排序 + 粗去重
    grasps.sort(key=lambda g: g["q_score"], reverse=True)
    kept, grid = [], set()
    voxel = 0.003
    for g in grasps:
        key = tuple(np.floor(np.array(g["center"])/voxel).astype(int))
        if key in grid: 
            continue
        grid.add(key); kept.append(g)
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=str, default=DEFAULT_MESH)
    ap.add_argument("--out",  type=str, default="grasps.json")
    ap.add_argument("--n",    type=int, default=SAMPLE_NUM)
    ap.add_argument("--mu",   type=float, default=MU)
    ap.add_argument("--wmin", type=float, default=W_MIN)
    ap.add_argument("--wmax", type=float, default=W_MAX)
    ap.add_argument("--clear",type=float, default=CLEARANCE)
    ap.add_argument("--scale",type=float, default=0.01, help="把 mesh 统一到米，LEGO 常用 0.01")
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    # —— 关键修正：缩放到米，修法向，清理网格 ——
    if args.scale != 1.0:
        mesh.apply_scale(args.scale)
    try:
        repair.fix_normals(mesh)
    except Exception:
        pass
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)

    ext = mesh.extents  # AABB 尺寸
    print(f"[INFO] AABB extents (m): {np.round(ext,4)}  | watertight={mesh.is_watertight}")
    min_side = float(np.min(ext))  # LEGO ~ 0.01m
    # 自动宽度（若你没传）
    wmin = args.wmin if args.wmin>0 else 0.20*min_side
    wmax = args.wmax if args.wmax>0 else 0.95*min_side
    if wmin >= wmax:
        wmin, wmax = 0.2*min_side, 0.95*min_side
    print(f"[INFO] width range auto={wmin:.4f} ~ {wmax:.4f} (m)")

    grasps = synthesize_grasps(mesh, sample_num=args.n, mu=args.mu,
                               wmin=wmin, wmax=wmax,
                               clearance=args.clear, angle_tol=ANGLE_TOL)

    grasps = grasps[:N_TOP]
    with open(args.out, "w") as f:
        json.dump({"mesh": args.mesh, "grasps": grasps}, f, indent=2)
    print(f"[OK] total={len(grasps)} → {args.out}")
    if len(grasps):
        g0 = grasps[0]
        print("[TOP1] center=", np.round(g0["center"],4),
              " quat(xyzw)=", np.round(g0["quat_xyzw"],4),
              " width(m)=", round(g0["width"],4),
              " score=", round(g0["q_score"],3))

if __name__ == "__main__":
    main()
