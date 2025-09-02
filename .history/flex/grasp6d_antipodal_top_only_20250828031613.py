#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grasp6d_antipodal_side.py
在地面上生成二指夹爪的侧向对向抓取（antipodal），满足摩擦圆锥约束。
- 只取侧边（法线近水平）的接触
- 法线用KNN高斯平滑，抗三角面量化
- 输出两接触点、抓取宽度、以及可还原6D姿态的夹爪坐标系（x=闭合，z=向下）
"""

import json, argparse, math, random, sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree

# ---------------- 工具 ----------------
def unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else (v / n)

def mat_to_quat(R):
    """3x3 -> xyzw quaternion（右手系，列向量为基）"""
    m = R
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
            w = (m[2,1] - m[1,2]) / s
        elif i == 1:
            s = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
            w = (m[0,2] - m[2,0]) / s
        else:
            s = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
            w = (m[1,0] - m[0,1]) / s
    return [x, y, z, w]

def load_mesh_safely(path, scale=1.0):
    mesh = trimesh.load(path, process=False)
    if scale != 1.0:
        mesh.apply_scale(scale)
    try:
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        trimesh.repair.fill_holes(mesh)
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    return mesh

def face_normals_from_vertex_normals(F, VN):
    """用顶点法线均值当作面法线，便于给采样点赋法线（较平滑）"""
    fn = []
    for tri in F:
        n = unit(VN[tri[0]] + VN[tri[1]] + VN[tri[2]])
        if np.linalg.norm(n) < 1e-12:
            n = np.array([0,0,1.0])
        fn.append(n)
    return np.asarray(fn)

def smooth_normal_at_point(p, pts, Ns, kdt, k=24, sigma=0.005):
    """KNN + 高斯加权 平滑法线，降低三角面量化噪声影响。"""
    if k <= 0 or len(pts) == 0:
        return None
    d, idx = kdt.query(p, k=min(k, len(pts)))
    if np.isscalar(d):
        d = np.array([d]); idx = np.array([idx])
    w = np.exp(-(d**2) / (2.0 * (sigma**2) + 1e-12))
    n = np.average(Ns[idx], axis=0, weights=w)
    return unit(n)

def seed_mask_side(points, normals, table_z=0.0, side_thresh=0.35, min_height=0.003):
    """
    只选“侧边”候选：|n·z| <= side_thresh，并且点必须离地面有一定高度
    side_thresh≈0.35 对应法线与水平夹角<=~20°
    """
    nz = np.abs(normals[:,2])
    side_ok = (nz <= side_thresh)
    high_ok = (points[:,2] >= (table_z + min_height))
    return side_ok & high_ok

# 可选：简易走廊净空（用 signed_distance / closest_point），对单一网格常可跳过
def corridor_clear_segment(p, q, mesh, clearance=4e-4, samples=7):
    ts = np.linspace(0.0, 1.0, samples)
    pts = (1-ts)[:,None]*p[None,:] + ts[:,None]*q[None,:]
    # 取表面最近点距离作为“无符号距离”
    try:
        from trimesh.proximity import closest_point
        cp, dist, tri = closest_point(mesh, pts)
        d = np.asarray(dist)
    except Exception:
        # 退化处理：直接返回 True
        return True
    return np.all(d >= clearance)

# ---------------- 主逻辑 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=str, required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--n', type=int, default=20000, help='采样种子数量上限（面上随机采样）')

    # 夹爪开口宽度范围（m）：按乐高宽~6-16mm做一个合理范围
    ap.add_argument('--wmin', type=float, default=0.003, help='最小抓取宽度')
    ap.add_argument('--wmax', type=float, default=0.020, help='最大抓取宽度')

    # 摩擦与判据（μ=0.6 ~ 橡胶指垫抓 ABS 的保守值）
    ap.add_argument('--mu', type=float, default=0.6, help='摩擦系数（材质相关，质量不影响μ）')
    ap.add_argument('--ang_slack_deg', type=float, default=12.0,
                    help='摩擦圆锥半角额外余量（度），补偿法线噪声')
    ap.add_argument('--opp_slack_deg', type=float, default=35.0,
                    help='两侧法线的“近对向”容忍角度上限（度）')

    # 法线平滑（按 1cm 量级物体，5mm 高斯尺度较合适）
    ap.add_argument('--avg_norm_k', type=int, default=24, help='KNN法线平滑邻居数（0=不用）')
    ap.add_argument('--avg_norm_sigma', type=float, default=0.005, help='高斯尺度（米）')

    # 只抓侧边
    ap.add_argument('--side_thresh', type=float, default=0.35,
                    help='|n·z|阈值（<=此值才算侧边），0.35≈水平±20°')

    # 地面与安全高度
    ap.add_argument('--table_z', type=float, default=0.0, help='地面高度')
    ap.add_argument('--min_contact_z', type=float, default=0.004,
                    help='接触点至少高于地面这么多，防止夹爪或手掌蹭地')

    # 走廊净空（单物体可调小或忽略）
    ap.add_argument('--clear', type=float, default=0.0004, help='走廊净空阈值（m）')
    ap.add_argument('--check_corridor', action='store_true', help='开启走廊净空检查')

    ap.add_argument('--max_per_seed', type=int, default=3, help='每个种子最多收多少 grasp')
    ap.add_argument('--out', type=str, default='side_grasps.json')
    args = ap.parse_args()

    # 质量 10g 仅用于评分偏好（高处更好），与 μ 无直接关系
    mass_kg = 0.010

    mesh = load_mesh_safely(args.mesh, scale=args.scale)
    V = mesh.vertices.view(np.ndarray)
    F = mesh.faces.view(np.ndarray)
    VN = mesh.vertex_normals if (mesh.vertex_normals is not None and len(mesh.vertex_normals)>0) else None
    if VN is None or len(VN) != len(V):
        # 回退：用几何法线
        mesh.rezero()
        VN = mesh.vertex_normals
    FN_from_VN = face_normals_from_vertex_normals(F, VN)

    extents = mesh.extents
    print(f"[INFO] AABB extents(m): {np.array2string(extents, precision=4)}, watertight={mesh.is_watertight}")

    # 面上均匀采样
    M = max(args.n * 2, 40000)
    pts, fids = trimesh.sample.sample_surface(mesh, M)
    Ns = FN_from_VN[fids]

    # 只选侧边种子
    msk = seed_mask_side(pts, Ns, table_z=args.table_z,
                         side_thresh=args.side_thresh, min_height=args.min_contact_z)
    candidates = np.flatnonzero(msk)
    if len(candidates) == 0:
        print("[ERR] 没有满足侧边与最小高度的候选点，调大 --side_thresh 或降低 --min_contact_z")
        sys.exit(1)
    print(f"[INFO] 侧边候选点: {len(candidates)}")

    # KDTree 便于邻域搜索与法线平滑
    kdt = cKDTree(pts)

    # 判据参数：摩擦圆锥 + 近对向
    alpha = math.atan(args.mu) + math.radians(args.ang_slack_deg)
    cos_alpha = math.cos(alpha)
    cos_opp = math.cos(math.radians(args.opp_slack_deg))

    # 随机挑种子
    if len(candidates) > args.n:
        seeds = np.random.choice(candidates, size=args.n, replace=False)
    else:
        seeds = candidates

    up = np.array([0,0,1.0], dtype=float)
    total = len(seeds)
    report_every = max(1, total // 20)

    grasps = []
    seen = set()

    for si, idx in enumerate(seeds):
        p  = pts[idx]
        n1_raw = unit(Ns[idx])
        # 平滑后的法线
        n1 = smooth_normal_at_point(p, pts, Ns, kdt,
                                    k=args.avg_norm_k, sigma=args.avg_norm_sigma) or n1_raw

        # 搜索抓取宽度范围候选
        idxs = kdt.query_ball_point(p, r=args.wmax + 1e-4)
        cand = []
        for j in idxs:
            if j == idx: continue
            v = pts[j] - p
            d = np.linalg.norm(v)
            if d < args.wmin or d > args.wmax: 
                continue
            cand.append(j)

        random.shuffle(cand)
        added = 0

        for j in cand:
            q  = pts[j]
            if (q[2] < args.table_z + args.min_contact_z):  # 对侧也要离地足够
                continue

            n2_raw = unit(Ns[j])
            n2 = smooth_normal_at_point(q, pts, Ns, kdt,
                                        k=args.avg_norm_k, sigma=args.avg_norm_sigma) or n2_raw

            # 仍需是侧边
            if abs(n2[2]) > args.side_thresh: 
                continue

            v  = q - p
            d  = np.linalg.norm(v)
            if d < 1e-9:
                continue
            u  = v / d  # 闭合方向（从 p 指向 q）

            # 近对向容忍：n1·(-n2) >= cos_opp
            if np.dot(n1, -n2) < cos_opp:
                continue

            # 摩擦圆锥(带余量)：u 必须落在 n1 的圆锥内；-u 落在 n2 的圆锥内
            if np.dot(n1,  u) < cos_alpha:    continue
            if np.dot(n2, -u) < cos_alpha:    continue

            # 走廊净空（单物体常可跳过；开启则检查）
            if args.check_corridor and (not corridor_clear_segment(p, q, mesh, clearance=args.clear)):
                continue

            # 夹爪坐标系（用于还原6D位姿）：
            # x轴：闭合方向 u
            # z轴：世界 -Z（自上而下接近）
            # y轴：z × x（右手系），再正交化
            x_axis = unit(u)
            z_axis = np.array([0,0,-1.0])
            y_axis = unit(np.cross(z_axis, x_axis))
            x_axis = unit(np.cross(y_axis, z_axis))
            R = np.column_stack([x_axis, y_axis, z_axis])
            quat = mat_to_quat(R)

            mid = 0.5 * (p + q)

            # 去重（位置+方向量化）
            key = (tuple(np.round(mid, 4)), tuple(np.round(u, 3)))
            if key in seen:
                continue
            seen.add(key)

            # 评分：摩擦余量 + 近对向 + 宽度居中 + 高度
            opp_score = (np.dot(n1, -n2) - cos_opp) / (1.0 - cos_opp + 1e-12)
            score = float(
                0.45 * (np.dot(n1, u) + np.dot(n2, -u)) +
                0.25 * np.clip(opp_score, 0.0, 1.0) +
                0.20 * (1.0 - abs((d - 0.5*(args.wmin+args.wmax)) / (args.wmax-args.wmin + 1e-9))) +
                0.10 * (mid[2] / max(extents[2], 1e-6))
            )

            grasps.append({
                "position": mid.tolist(),                  # 抓取中心
                "width": float(d),                         # 夹爪开口
                "contacts": [p.tolist(), q.tolist()],      # 两接触点
                "normals":  [n1.tolist(), n2.tolist()],    # 两侧法线（已平滑）
                "closing_direction": x_axis.tolist(),      # 夹爪x轴
                "approach_direction": z_axis.tolist(),     # 接近方向（向下）
                "R": R.tolist(),                           # 旋转矩阵（列向量）
                "quaternion_xyzw": quat,                   # 四元数（xyzw）
                "score": score
            })
            added += 1
            if added >= args.max_per_seed:
                break

        if si % report_every == 0:
            print(f"[{si}/{total}] seeds processed, grasps={len(grasps)}")

    grasps.sort(key=lambda g: g["score"], reverse=True)
    out = {
        "mesh": args.mesh,
        "scale": args.scale,
        "mass_kg": 0.010,                 # 10g（仅作记录与后续评估参考）
        "friction_coefficient": args.mu,  # μ=0.6（橡胶指垫抓ABS的保守值）
        "assumptions": {
            "side_only": True,
            "world_down_as_approach": True,
            "antipodal_with_friction_cone": True
        },
        "grasps": grasps
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[SUCCESS] {len(grasps)} grasps → {args.out}")

    if grasps:
        best = grasps[0]
        print(f"[BEST] pos={np.round(best['position'],4)}, width={best['width']:.4f} m, score={best['score']:.3f}")

if __name__ == "__main__":
    main()
