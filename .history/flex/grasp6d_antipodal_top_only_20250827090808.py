#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# grasp6d_antipodal_top_only.py
# 生成二指夹爪的 6D 抓取（只能从上方抓取）
import json, argparse, math, random, sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree

# ------------------ 工具 ------------------
def unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps: return v
    return v / n

def rot_from_xy(x, y):
    """给定 x轴 和 近似 y轴，构造正交的旋转矩阵（列向量为基）"""
    ex = unit(x)
    # 去掉 y 在 x 上的分量
    y = y - np.dot(y, ex) * ex
    ey = unit(y) if np.linalg.norm(y) > 1e-12 else np.array([0,1,0], dtype=float)
    ez = np.cross(ex, ey)
    ey = np.cross(ez, ex)  # 保证正交
    return np.stack([ex, ey, unit(ez)], axis=1)

def mat_to_quat(R):
    """3x3 -> xyzw quaternion"""
    m = R
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t+1.0) * 2
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        i = np.argmax(np.diag(m))
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

# ------------------ 网格加载与法线 ------------------
def load_mesh_safely(path, scale=1.0):
    mesh = trimesh.load(path, process=False)
    if scale != 1.0:
        mesh.apply_scale(scale)

    # 轻修复：尽可能改善法线一致性，但不强行 watertight
    try:
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        trimesh.repair.fill_holes(mesh)
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    V = mesh.vertices.view(np.ndarray)
    F = mesh.faces.view(np.ndarray)
    VN_mesh = mesh.vertex_normals.copy() if (mesh.vertex_normals is not None and len(mesh.vertex_normals)>0) else None

    # PCA 法线（非 watertight 时更稳）
    kdt = cKDTree(V)
    def pca_normals(k=24):
        N = np.zeros_like(V)
        C0 = V.mean(0)
        for i, p in enumerate(V):
            _, idx = kdt.query(p, k=min(k, len(V)))
            Q = V[idx] - V[i]
            C = Q.T @ Q
            eigw, eigv = np.linalg.eigh(C)
            n = eigv[:, 0]
            # 朝外：与质心方向外指
            if np.dot(n, V[i] - C0) < 0:
                n = -n
            N[i] = unit(n)
        return N
    VN_pca = pca_normals(k=24)

    return mesh, V, F, VN_mesh, VN_pca

def face_normals_from_vertex_normals(F, VN):
    """用顶点法线平均出面法线（适配 PCA 顶点法线）"""
    fn = []
    for tri in F:
        n = unit(VN[tri[0]] + VN[tri[1]] + VN[tri[2]])
        if np.linalg.norm(n) < 1e-12:
            n = np.array([0,0,1.0])
        fn.append(n)
    return np.asarray(fn)

# ------------------ 走廊净空（无符号距离，更稳） ------------------
def make_proximity(mesh):
    try:
        from trimesh.proximity import ProximityQuery
        return ProximityQuery(mesh)
    except Exception as e:
        print("[WARN] ProximityQuery 不可用：", e)
        return None

def corridor_clear_segment(p, q, pq, clearance, samples=7):
    if pq is None:  # 没有距离查询，直接放行（不建议）
        return True
    ts = np.linspace(0.0, 1.0, samples)
    pts = p[None,:] * (1-ts)[:,None] + q[None,:] * ts[:,None]
    d = pq.distance(pts)
    return np.all(d >= clearance)

# ------------------ 采样：只在严格朝上的面上取种子 ------------------
def seed_mask(points, normals, up=np.array([0,0,1.0]), up_thresh=0.8, outward_thresh=0.3):
    """
    只选择严格朝上的面作为种子点
    up_thresh: 法线与上方向的点积阈值（0.8 = 约37度以内）
    outward_thresh: 确保法线朝外（避免内部面）
    """
    C = points.mean(0)
    to_out = points - C
    to_out = to_out / (np.linalg.norm(to_out, axis=1, keepdims=True) + 1e-12)
    
    # 严格朝上：法线必须主要朝上
    up_ok = (normals @ up) > up_thresh
    
    # 确保法线朝外（避免内部面）
    outward = (np.sum(normals * to_out, axis=1) > outward_thresh)
    
    return up_ok & outward

# ------------------ 检查抓取高度是否安全 ------------------
def check_grasp_height_safe(grasp_center_z, gripper_len, table_z, safety_margin=0.005):
    """检查抓取器是否会碰到地面"""
    return grasp_center_z - gripper_len >= table_z + safety_margin

# ------------------ 主逻辑 ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=str, required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--n', type=int, default=20000, help='采样种子数量（上限）')
    ap.add_argument('--mu', type=float, default=0.8, help='摩擦系数')
    ap.add_argument('--wmin', type=float, default=0.003)
    ap.add_argument('--wmax', type=float, default=0.016)
    ap.add_argument('--clear', type=float, default=0.0004, help='走廊净空（m）')
    ap.add_argument('--max_per_seed', type=int, default=3, help='每个种子最多收多少 grasp')
    ap.add_argument('--use_pca_normals', action='store_true', help='用 PCA 顶点法线')
    ap.add_argument('--up_thresh', type=float, default=0.8, help='法线朝上阈值（0.8=约37度内）')
    ap.add_argument('--table_z', type=float, default=0.0, help='桌面高度')
    ap.add_argument('--gripper_len', type=float, default=0.05, help='夹爪长度')
    ap.add_argument('--safety_margin', type=float, default=0.005, help='安全距离')
    ap.add_argument('--out', type=str, default='top_grasps.json')
    args = ap.parse_args()

    mesh, V, F, VN_mesh, VN_pca = load_mesh_safely(args.mesh, scale=args.scale)
    extents = mesh.extents
    print(f"[INFO] AABB extents(m): {np.array2string(extents, precision=4)}, watertight={mesh.is_watertight}")

    # 顶点法线选择
    VNv = VN_pca if args.use_pca_normals or (VN_mesh is None) else VN_mesh
    # 用顶点法线合成"面法线"（配合面采样）
    FN_from_VNv = face_normals_from_vertex_normals(F, VNv)

    # 表面均匀采样（点+face id），用面法线（更平滑）
    M = max(args.n * 2, 30000)
    pts, fids = trimesh.sample.sample_surface(mesh, M)
    Ns = FN_from_VNv[fids]
    
    # 只保留严格朝上的种子点
    msk = seed_mask(pts, Ns, up=np.array([0,0,1.0]), 
                   up_thresh=args.up_thresh, outward_thresh=0.3)
    candidates = np.flatnonzero(msk)
    
    if len(candidates) == 0:
        print("[ERR] 没有朝上的候选点；请降低 --up_thresh 或检查 mesh 方向")
        sys.exit(1)

    print(f"[INFO] 找到 {len(candidates)} 个朝上的候选点")

    # KDTree 供邻域查询（基于采样点）
    kdt = cKDTree(pts)
    pq  = make_proximity(mesh)

    # 摩擦圆锥角阈值
    cos_phi = 1.0 / math.sqrt(1.0 + args.mu * args.mu)
    # 方向/对齐额外限制
    cos_align = math.cos(math.radians(25.0))

    grasps = []
    seen = set()

    # 随机选种子
    if len(candidates) > args.n:
        seeds = np.random.choice(candidates, size=args.n, replace=False)
    else:
        seeds = candidates

    up = np.array([0,0,1.0], dtype=float)
    total = len(seeds)
    report_every = max(1, total // 20)

    for si, idx in enumerate(seeds):
        p  = pts[idx]
        n1 = unit(Ns[idx])

        # 在 [wmin, wmax] 范围内找邻居作为对侧接触
        idxs = kdt.query_ball_point(p, r=args.wmax + 1e-4)
        # 粗过滤：距离要 >= wmin
        cand = []
        for j in idxs:
            if j == idx: continue
            v = pts[j] - p
            d = np.linalg.norm(v)
            if d < args.wmin or d > args.wmax: continue
            cand.append(j)

        random.shuffle(cand)
        added = 0
        
        for j in cand:
            q  = pts[j]
            n2 = unit(Ns[j])
            v  = q - p
            d  = np.linalg.norm(v)
            if d < 1e-9: continue
            u  = v / d  # 闭合方向（finger 夹向）

            # 对偶 + 摩擦圆锥判据
            if np.dot(n1, u) < cos_phi:    continue
            if np.dot(-n2, u) < cos_phi:   continue

            # 方向对齐（更稳）
            if np.dot(n1,  u) < cos_align: continue
            if np.dot(n2, -u) < cos_align: continue

            # 走廊净空
            if not corridor_clear_segment(p, q, pq, clearance=args.clear, samples=7):
                continue

            # 生成 6D 姿态：强制从上方向下接近
            mid = 0.5 * (p + q)
            
            # 检查高度安全
            if not check_grasp_height_safe(mid[2], args.gripper_len, args.table_z, args.safety_margin):
                continue

            # 强制 approach 方向为世界坐标系 -Z（向下）
            approach_dir = np.array([0, 0, -1.0])
            
            # 夹爪坐标系：
            # x轴：夹持方向 (u)
            # z轴：approach 方向（向下）
            # y轴：x × z（确保右手系）
            x_axis = unit(u)
            z_axis = unit(approach_dir)
            
            # 确保正交
            y_axis = unit(np.cross(z_axis, x_axis))
            x_axis = unit(np.cross(y_axis, z_axis))
            
            R = np.column_stack([x_axis, y_axis, z_axis])
            quat = mat_to_quat(R)

            # 去重：以 (mid 量化, u 的方位) 为键
            key = (tuple(np.round(mid, 4)), tuple(np.round(u, 3)))
            if key in seen:
                continue
            seen.add(key)

            # 评分：考虑摩擦余量、宽度适宜度、高度优势
            score = float(
                0.5 * (np.dot(n1, u) + np.dot(-n2, u)) +  # 摩擦余量
                0.3 * (1.0 - abs((d - 0.5*(args.wmin+args.wmax)) / (args.wmax-args.wmin + 1e-9))) +  # 宽度居中
                0.2 * (mid[2] / extents[2])  # 高度优势（越高越好）
            )

            grasps.append({
                "position": mid.tolist(),
                "quaternion_xyzw": quat,
                "width": float(d),
                "contacts": [p.tolist(), q.tolist()],
                "score": score,
                "approach_direction": [0, 0, -1]  # 明确标注接近方向
            })
            added += 1
            if added >= args.max_per_seed:
                break

        if si % report_every == 0:
            print(f"[{si}/{total}] seeds processed, grasps={len(grasps)}")

    # 按评分排序
    grasps.sort(key=lambda x: x['score'], reverse=True)
    
    # 保存
    out = {
        "mesh": args.mesh,
        "scale": args.scale,
        "table_height": args.table_z,
        "gripper_length": args.gripper_len,
        "grasps": grasps
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[SUCCESS] {len(grasps)} 个上方抓取姿态 → {args.out}")
    
    if grasps:
        best = grasps[0]
        print(f"[BEST] 位置: {np.round(best['position'], 4)}, 宽度: {best['width']:.4f}, 评分: {best['score']:.3f}")

if __name__ == '__main__':
    main()