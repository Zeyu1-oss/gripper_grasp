#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, math, argparse, numpy as np, trimesh
from trimesh import repair
from scipy.spatial import cKDTree

# ---------------- 参数 ----------------
MU              = 0.6
SAMPLES         = 12000           # 表面采样数（比原来小）
SEED_STRIDE     = 3               # 种子点抽样步长
MAX_PARTNERS    = 6               # 每个种子最多尝试的对点
ANGLE_TOL       = np.deg2rad(8.0)
CLEARANCE       = 0.0015          # 指尖走廊净空
FINGER_LEN      = 0.03
TOPK            = 200
VOXEL_MM        = 3.0
UP_NZ_THRESH    = 0.2             # 只保留朝上的点（可从上方接近）
OPPOSE_DOT_MIN  = 0.8             # 快速法向相反性筛选

def unit(v):
    n = np.linalg.norm(v)
    return v/n if n > 1e-12 else v

def rot_from_xy(x_hat, y_hat):
    x = unit(x_hat)
    y = unit(y_hat - np.dot(y_hat, x) * x)
    z = unit(np.cross(x, y))
    return np.column_stack([x, y, z])

def mat_to_quat_xyzw(R):
    H = np.eye(4); H[:3,:3] = R
    q = trimesh.transformations.quaternion_from_matrix(H)  # wxyz
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]  # xyzw

def build_proximity(mesh):
    try:
        return trimesh.proximity.ProximityQuery(mesh)
    except Exception:
        return None

def corridor_clear_batch(pq, mesh, p1, p2, clearance, steps=10):
    """批量距离查询的走廊净空检查"""
    ts  = np.linspace(0.05, 0.95, steps)
    seg = (p2 - p1)
    pts = p1[None, :] + ts[:, None] * seg[None, :]

    if pq is not None:
        try:
            if mesh.is_watertight:
                d = pq.signed_distance(pts)
                return np.all(np.abs(d) > clearance)
            else:
                d = pq.distance(pts)
                return np.all(d > clearance)
        except Exception:
            pass

    # 兜底：closest_point（慢一些）
    closest, _, _ = trimesh.proximity.closest_point(mesh, pts)
    return np.all(np.linalg.norm(pts - closest, axis=1) > clearance)

def approach_not_hit_table(center, z_hat, table_z, finger_len, safety=1e-4):
    zs = (center - np.outer(np.linspace(0, finger_len, 10), z_hat))[:, 2]
    return np.all(zs > table_z + safety)

def score_grasp(n, nj, axis, cone_half, center, table_z, ext_z):
    a1 = math.acos(np.clip(np.dot(n,  axis), -1, 1))
    a2 = math.acos(np.clip(np.dot(nj,-axis), -1, 1))
    fc_margin = max(0.0, cone_half - max(a1, a2)) / (cone_half + 1e-9)
    normal_sym = 1.0 - abs(np.dot(n, nj))
    height_bonus = 0.15 * np.clip((center[2]-table_z)/max(ext_z,1e-3), 0, 1)
    return 0.6*fc_margin + 0.4*normal_sym + height_bonus

def synthesize(mesh, table_z=0.0, mu=MU, n_samples=SAMPLES,
               wmin=None, wmax=None, clearance=CLEARANCE,
               finger_len=FINGER_LEN, angle_tol=ANGLE_TOL):
    # 采样
    try:
        P, face_idx = trimesh.sample.sample_surface_even(mesh, n_samples)
    except Exception:
        P, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    FN  = mesh.face_normals[face_idx]
    N   = np.apply_along_axis(unit, 1, FN)

    # 只保留“朝上”的点作种子
    up = N[:, 2] > UP_NZ_THRESH
    seeds = np.nonzero(up)[0][::SEED_STRIDE]
    if len(seeds) == 0:
        seeds = np.arange(0, len(P), SEED_STRIDE)

    # 宽度范围
    ext = mesh.extents
    min_side = float(np.min(ext))
    if wmin is None: wmin = 0.20 * min_side
    if wmax is None: wmax = 0.95 * min_side
    if wmin >= wmax: wmin, wmax = 0.2*min_side, 0.95*min_side

    cone_half = math.atan(mu)
    kdt = cKDTree(P)
    pq  = build_proximity(mesh)

    out = []
    total = len(seeds)
    for si, i in enumerate(seeds):
        if si % 500 == 0:
            print(f"[{si}/{total}] seeds processed, grasps={len(out)}")
        p, n = P[i], unit(N[i])

        idxs = kdt.query_ball_point(p, wmax)
        if not idxs: 
            continue

        # 距离筛选 + 最近排序
        cand = []
        for j in idxs:
            if j == i: continue
            w = np.linalg.norm(P[j] - p)
            if w < wmin or w > wmax: 
                continue
            cand.append((w, j))
        if not cand:
            continue
        cand.sort(key=lambda x: x[0])
        cand = cand[:MAX_PARTNERS]

        for w, j in cand:
            pj, nj = P[j], unit(N[j])

            # 快速法向相反性筛选（减少计算）
            if np.dot(n, -nj) < OPPOSE_DOT_MIN:
                continue

            axis = unit(pj - p)

            # 摩擦圆锥（带容差）
            a1 = math.acos(np.clip(np.dot(n,  axis), -1, 1))
            a2 = math.acos(np.clip(np.dot(nj,-axis), -1, 1))
            if (a1 > cone_half + angle_tol) or (a2 > cone_half + angle_tol):
                continue

            # 走廊净空
            if not corridor_clear_batch(pq, mesh, p, pj, clearance):
                continue

            # 夹爪坐标系
            y_hat = axis
            sumn = n + nj
            if np.linalg.norm(sumn) > 1e-6:
                z_hat = unit(-sumn)
            else:
                tmp = np.array([0,0,1])
                z_hat = unit(np.cross(y_hat, tmp))
                if z_hat[2] < 0: z_hat = -z_hat
            x_hat = unit(np.cross(y_hat, z_hat))
            z_hat = unit(np.cross(x_hat, y_hat))

            center = 0.5*(p + pj)

            # 上方接近不刮地
            if not approach_not_hit_table(center, z_hat, table_z, finger_len):
                continue

            R = rot_from_xy(x_hat, y_hat)
            quat = mat_to_quat_xyzw(R)
            s = score_grasp(n, nj, axis, cone_half, center, table_z, ext[2])

            out.append({
                "center": center.tolist(),
                "quat_xyzw": quat,
                "width": float(w),
                "q_score": float(s)
            })

    # 排序 + 体素去重
    out.sort(key=lambda g: g["q_score"], reverse=True)
    kept, grid = [], set()
    vox = VOXEL_MM/1000.0
    for g in out:
        key = tuple(np.floor(np.array(g["center"])/vox).astype(int))
        if key in grid: 
            continue
        grid.add(key); kept.append(g)
        if len(kept) >= TOPK: break
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=str, required=True)
    ap.add_argument("--out",  type=str, default="grasps.json")
    ap.add_argument("--scale",type=float, default=0.01, help="把 mesh 单位缩放到米，常见 OBJ=厘米 → 0.01")
    ap.add_argument("--table_z", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=SAMPLES)
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--wmin", type=float, default=-1)
    ap.add_argument("--wmax", type=float, default=-1)
    ap.add_argument("--clear",type=float, default=CLEARANCE)
    ap.add_argument("--finger_len", type=float, default=FINGER_LEN)
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    # 缩放到米 + 网格清理（非 watertight 也可用）
    if args.scale != 1.0:
        mesh.apply_scale(args.scale)
    try:
        repair.fix_normals(mesh)
    except Exception:
        pass
    # 用新的 API，避免 deprecation 警告
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)

    ext = np.round(mesh.extents, 4)
    print(f"[INFO] AABB extents(m): {ext}, watertight={mesh.is_watertight}")

    wmin = None if args.wmin <= 0 else args.wmin
    wmax = None if args.wmax <= 0 else args.wmax

    grasps = synthesize(
        mesh, table_z=args.table_z, mu=args.mu, n_samples=args.n,
        wmin=wmin, wmax=wmax, clearance=args.clear, finger_len=args.finger_len
    )

    with open(args.out, "w") as f:
        json.dump({"mesh": args.mesh, "grasps": grasps}, f, indent=2)
    print(f"[OK] {len(grasps)} grasps → {args.out}")
    if grasps:
        g0 = grasps[0]
        print("[TOP1]", 
              "center=", np.round(g0["center"],4),
              "quat(xyzw)=", np.round(g0["quat_xyzw"],4),
              "width=", round(g0["width"],4),
              "score=", round(g0["q_score"],3))

if __name__ == "__main__":
    main()
