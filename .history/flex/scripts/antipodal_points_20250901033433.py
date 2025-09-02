#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import trimesh
from tqdm import trange, tqdm
from scipy.optimize import minimize, nnls
from dataclasses import dataclass, asdict

# ===================== 基础工具 =====================

def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def _orthonormal_tangent_basis(axis):
    axis = _unit(axis)
    up = np.array([0., 0., 1.]) if abs(np.dot(axis, [0,0,1])) < 0.95 else np.array([0., 1., 0.])
    t1 = _unit(np.cross(axis, up))
    t2 = _unit(np.cross(axis, t1))
    return t1, t2

def _build_ray_engine(mesh):
    # 优先 embree
    return (trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
            if trimesh.ray.has_embree else
            trimesh.ray.ray_triangle.RayMeshIntersector(mesh))

# ===================== 几何判据与几何分 =====================

def is_antipodal(p1, n1, p2, n2, mu):
    """反对称-摩擦锥判据：沿 ±u 是否都落在两侧摩擦锥内"""
    u = p2 - p1
    d = np.linalg.norm(u)
    if d < 1e-9:
        return False
    u /= d
    theta = np.arctan(mu)
    cos_th = np.cos(theta)
    return (np.dot(-u, n1) >= cos_th) and (np.dot(u, n2) >= cos_th)

def geom_score_pair(p, q, n_p, n_q, mu, width_min, width_max):
    """
    几何打分（越大越好）：
      + 两侧摩擦锥“裕度”（softplus）
      + 反法向程度 (-n·n)
      - 宽度惩罚（区间外二次罚）
    """
    v = q - p; d = np.linalg.norm(v)
    if d < 1e-9:
        return -1e9
    u = v / d
    theta = np.arctan(mu)
    cos_th = np.cos(theta)

    # 锥内裕度（值越大表示越“楔入”）
    m_p = (-cos_th - np.dot(n_p, u))      # p 侧看 -u
    m_q = ( np.dot(n_q, u) - cos_th)      # q 侧看 +u
    sp = lambda x: np.log1p(np.exp(x))    # softplus
    margin_term = sp(m_p) + sp(m_q)

    # 反法向
    anti_term = -float(np.dot(n_p, n_q))

    # 宽度惩罚
    if d < width_min:
        w_pen = (width_min - d)**2
    elif d > width_max:
        w_pen = (d - width_max)**2
    else:
        mid = 0.5*(width_min + width_max)
        w_pen = 0.25*((d - mid)/(0.5*(width_max - width_min)+1e-9))**2

    return 1.0*margin_term + 0.3*anti_term - 0.3*w_pen

# ===================== 力闭合（扳手空间） =====================

def _contact_wrench_rays(p, n, mu, m_dirs, com, torque_scale):
    """
    在切平面取 m_dirs 个方向，离散摩擦锥：
      f_dir = normalize(n + mu * t)
    返回 (6, m_dirs) 列向量，每列 [f; (r×f)/torque_scale]
    """
    t1, t2 = _orthonormal_tangent_basis(n)
    cols = []
    r = p - com
    for k in range(m_dirs):
        phi = 2*np.pi*k/m_dirs
        t = np.cos(phi)*t1 + np.sin(phi)*t2
        f = _unit(n + mu*t)
        tau = np.cross(r, f) / (torque_scale + 1e-12)
        cols.append(np.hstack([f, tau]))
    return np.column_stack(cols)

def _epsilon_qp(W):
    """
    Ferrari–Canny ε 质量的凸包近似：
       min ||W λ||  s.t. λ≥0, 1^T λ = 1
    先用 SLSQP，失败则退回 NNLS 近似。
    """
    m = W.shape[1]
    H = W.T @ W

    fun = lambda lam: 0.5 * lam @ H @ lam
    jac = lambda lam: H @ lam
    cons = [{'type': 'eq', 'fun': lambda lam: np.sum(lam) - 1.0,
             'jac':  lambda lam: np.ones_like(lam)}]
    bnds = [(0.0, None)] * m
    x0 = np.ones(m) / m

    res = minimize(fun, x0, method='SLSQP', jac=jac,
                   bounds=bnds, constraints=cons,
                   options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
    if res.success:
        lam = res.x
        return float(np.linalg.norm(W @ lam)), lam, True

    # 退路：带正则的 NNLS
    rho = 10.0
    A = np.vstack([W, np.sqrt(rho) * np.ones((1, m))])
    b = np.zeros(A.shape[0]); b[-1] = np.sqrt(rho)
    lam, _ = nnls(A, b)
    s = np.sum(lam)
    if s > 1e-12:
        lam = lam / s
    return float(np.linalg.norm(W @ lam)), lam, False

def wrench_quality_for_pair(p, q, n_p, n_q, mesh, mu=0.6, m_dirs=8):
    """
    返回 (eps, wscore)
      eps 越小越好（越接近力闭合）
      wscore = 1/(eps+1e-6)
    """
    # 量纲归一：力矩除以 Lc
    com = mesh.center_mass if mesh.is_watertight else 0.5*(mesh.bounds[0] + mesh.bounds[1])
    Lc  = float(np.linalg.norm(mesh.extents))
    W1 = _contact_wrench_rays(p, n_p, mu, m_dirs, com, Lc)
    W2 = _contact_wrench_rays(q, n_q, mu, m_dirs, com, Lc)
    W  = np.concatenate([W1, W2], axis=1)  # (6, 2*m_dirs)
    eps, lam, ok = _epsilon_qp(W)
    wscore = 1.0 / (eps + 1e-6)
    return eps, wscore

# ===================== 数据结构 =====================

@dataclass
class PairScored:
    p1: np.ndarray
    p2: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    width: float
    gscore: float
    eps: float
    wscore: float
    total: float


def generate_grasp_pairs_with_force_closure(
    mesh,
    mu=0.6,
    num_samples=30000,
    wmin=None, wmax=None,
    opp_angle_deg=30.0,
    cone_half_deg=12.0,
    rays_per_point=10,
    require_through_com=True,
    use_nearest_exit_if_closed=True,
    m_dirs=8,
    max_keep=8000,
    topk=4000,
    bottom_ratio=0.05,
    top_ratio=0.05,
    eps_thresh=None,           #力闭合 ε 硬阈值（如 0.01）
    wscore_thresh=None,        # 或者用 wscore 阈值（如 100）
    seed=0
):
    rng = np.random.default_rng(seed)
    ex, ey, ez = mesh.extents
    if wmin is None: wmin = 0.2 * float(min(ex, ey))
    if wmax is None: wmax = 1.2 * float(max(ex, ey))

    # 射线起点的小偏移（避免自相交触发里面）
    scale = float(np.linalg.norm(mesh.extents))
    eps_offset = 1e-5 * scale

    cos_opp = np.cos(np.deg2rad(opp_angle_deg))
    theta    = np.deg2rad(cone_half_deg)
    ray_engine = _build_ray_engine(mesh)
    com = mesh.center_mass if mesh.is_watertight else 0.5*(mesh.bounds[0]+mesh.bounds[1])

    # 侧壁高度窗口（排除上下帽檐）
    zmin, zmax = float(mesh.bounds[0,2]), float(mesh.bounds[1,2])
    height = zmax - zmin
    side_zmin = zmin + bottom_ratio * height
    side_zmax = zmax - top_ratio * height

    # 采样点与法向
    points, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
    normals = mesh.face_normals[face_idx]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12, None)

    # 统计拒绝原因
    rej = {
        'z_window':0, 'opp_angle':0, 'width':0,
        'antipodal':0, 'through_com':0, 'inside_mid':0, 'gscore<=0':0
    }

    candidates = []
    USE_EXIT_NEAREST = (mesh.is_watertight and use_nearest_exit_if_closed)

    for i in trange(num_samples, desc="生成候选(射线)"):
        p1 = points[i]
        if not (side_zmin < float(p1[2]) < side_zmax):
            rej['z_window'] += 1
            continue

        n1 = normals[i]
        axis = -n1
        t1, t2 = _orthonormal_tangent_basis(axis)
        phis = rng.uniform(0.0, 2.0*np.pi, size=rays_per_point)
        dirs = []
        for phi in phis:
            d = axis*np.cos(theta) + (np.cos(phi)*t1 + np.sin(phi)*t2)*np.sin(theta)
            dirs.append(_unit(d))
        origins = np.repeat((p1 - eps_offset*n1)[None, :], len(dirs), axis=0)

        # 多命中处理
        locs, idx_ray, tri_ids = ray_engine.intersects_location(origins, np.array(dirs), multiple_hits=True)
        if len(locs) == 0 or len(idx_ray) == 0:
            continue

        by_ray = {}
        for hit_idx, rid in enumerate(idx_ray):
            by_ray.setdefault(rid, []).append(hit_idx)

        early_break = False
        for rid, hit_list in by_ray.items():
            hit_pts  = locs[hit_list]
            hit_tris = [tri_ids[h] for h in hit_list]
            dists    = np.linalg.norm(hit_pts - p1, axis=1)

            if USE_EXIT_NEAREST:
                k_local = int(np.argmin(dists))  # watertight：最近出口
            else:
                t_min = max(5.0*eps_offset, 0.25*wmin)  # 厚度下限
                mask = dists >= t_min
                if not np.any(mask):
                    continue
                idxs = np.where(mask)[0]
                k_local = idxs[np.argmax(dists[idxs])]

            p2 = hit_pts[k_local]
            if not (side_zmin < float(p2[2]) < side_zmax):
                rej['z_window'] += 1
                continue

            tri_k = hit_tris[k_local]
            n2 = _unit(mesh.face_normals[tri_k])

            # 反法向粗过滤
            if np.dot(n1, n2) > -cos_opp:
                rej['opp_angle'] += 1
                continue

            # 宽度
            width = float(np.linalg.norm(p2 - p1))
            if width < wmin or width > wmax:
                rej['width'] += 1
                continue

            # 摩擦锥严格检查（antipodal）
            if not is_antipodal(p1, n1, p2, n2, mu):
                rej['antipodal'] += 1
                continue

            # 质心附近穿越
            if require_through_com:
                u = (p2 - p1) / (width + 1e-12)
                v = com - p1
                dist_line = np.linalg.norm(np.cross(u, v))
                if dist_line > 0.1 * max(ex, ey, ez):
                    rej['through_com'] += 1
                    continue

            # 中点在体内（仅 watertight）
            if mesh.is_watertight:
                mid = 0.5*(p1 + p2)
                if not mesh.contains([mid])[0]:
                    rej['inside_mid'] += 1
                    continue

            gscore = geom_score_pair(p1, p2, n1, n2, mu, wmin, wmax)
            if gscore <= 0:
                rej['gscore<=0'] += 1
                continue

            candidates.append((p1, p2, n1, n2, width, gscore))
            if len(candidates) >= max_keep:
                early_break = True
                break
        if early_break:
            break

    if len(candidates) == 0:
        print("⚠️  没有通过几何过滤的候选。建议：放宽 mu/opp_angle/cone_half_deg 或窗口比例。")
        return [], rej

    # 力闭合评分 + 综合排序 +（可选）阈值过滤
    results = []
    passed_fc = 0
    for (p1, p2, n1, n2, width, gscore) in tqdm(candidates, desc="力闭合打分"):
        eps_fc, wscore = wrench_quality_for_pair(p1, p2, n1, n2, mesh, mu=mu, m_dirs=m_dirs)

        # 可选硬阈值（任一方式满足即可）
        pass_eps = True if eps_thresh is None else (eps_fc <= float(eps_thresh))
        pass_wsc = True if wscore_thresh is None else (wscore >= float(wscore_thresh))
        if pass_eps and pass_wsc:
            passed_fc += 1
            total = 0.6*gscore + 0.4*np.log1p(wscore)
            results.append(PairScored(p1=p1, p2=p2, n1=n1, n2=n2,
                                      width=width, gscore=gscore, eps=eps_fc,
                                      wscore=wscore, total=total))
    # 排序 & 截断
    results.sort(key=lambda r: -r.total)
    if topk is not None and topk > 0:
        results = results[:topk]

    print("\n—— 统计汇总 ——")
    print(f"  采样总数           : {num_samples}")
    print(f"  通过几何过滤候选数 : {len(candidates)} / {num_samples}")
    print(f"  通过力闭合阈值数   : {passed_fc} / {len(candidates)}")
    print("  几何拒绝原因：")
    for k,v in rej.items():
        print(f"    - {k:12s}: {v}")

    return results, rej

# ===================== CLI & I/O =====================

def main():
    ap = argparse.ArgumentParser(description="生成两指抓取候选（几何+力闭合评分）")
    # IO
    ap.add_argument("--mesh", type=str, default="../lego.obj", help="网格路径（OBJ/STL/PLY 等）")
    ap.add_argument("--scale", type=float, default=0.01, help="统一尺度比例（例如 0.01 把 cm 变 m）")
    ap.add_argument("--out_dir", type=str, default="../results/antipodal_pairs", help="输出目录")
    ap.add_argument("--out_name", type=str, default="antipodal_pairs_ray", help="输出文件名（不含后缀）")
    ap.add_argument("--save_full", action="store_true", help="额外保存包含分数的结构化结果")

    # 采样/过滤
    ap.add_argument("--num_samples", type=int, default=30000)
    ap.add_argument("--max_keep", type=int, default=8000)
    ap.add_argument("--topk", type=int, default=4000)
    ap.add_argument("--mu", type=float, default=0.6)
    ap.add_argument("--opp_angle_deg", type=float, default=30.0)
    ap.add_argument("--cone_half_deg", type=float, default=12.0)
    ap.add_argument("--rays_per_point", type=int, default=10)
    ap.add_argument("--m_dirs", type=int, default=8)
    ap.add_argument("--bottom_ratio", type=float, default=0.01)
    ap.add_argument("--top_ratio", type=float, default=0.01)
    ap.add_argument("--wmin", type=float, default=None)
    ap.add_argument("--wmax", type=float, default=None)
    ap.add_argument("--require_through_com", action="store_true", default=True)
    ap.add_argument("--no_require_through_com", dest="require_through_com", action="store_false")

    # 力闭合阈值（可二选一或都不用）
    ap.add_argument("--eps_thresh", type=float, default=None, help="力闭合 ε 硬阈值，例如 0.01")
    ap.add_argument("--wscore_thresh", type=float, default=None, help="或用 wscore 阈值，例如 100")

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # 加载网格
    mesh_path = os.path.abspath(args.mesh)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    mesh = trimesh.load(mesh_path, force='mesh')

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise RuntimeError("网格没有三角形Empty triangle list。请检查输入文件。")

    if args.scale is not None and args.scale != 1.0:
        mesh.apply_scale(float(args.scale))

    print(f"Loaded mesh: {mesh_path}")
    print(f"  vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, watertight={mesh.is_watertight}")
    print(f"  extents = {mesh.extents}, bounds(z) = [{mesh.bounds[0,2]:.4f}, {mesh.bounds[1,2]:.4f}]")

    # 生成候选
    results, rej = generate_grasp_pairs_with_force_closure(
        mesh=mesh,
        mu=args.mu,
        num_samples=args.num_samples,
        wmin=args.wmin, wmax=args.wmax,
        opp_angle_deg=args.opp_angle_deg,
        cone_half_deg=args.cone_half_deg,
        rays_per_point=args.rays_per_point,
        require_through_com=args.require_through_com,
        use_nearest_exit_if_closed=True,
        m_dirs=args.m_dirs,
        max_keep=args.max_keep,
        topk=args.topk,
        bottom_ratio=args.bottom_ratio,
        top_ratio=args.top_ratio,
        eps_thresh=args.eps_thresh,
        wscore_thresh=args.wscore_thresh,
        seed=args.seed
    )

    # 输出
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1) 与你原先一致：仅保存 (p1,p2,n1,n2)
    pairs = [(r.p1, r.p2, r.n1, r.n2) for r in results]
    save_path_pairs = os.path.join(out_dir, f"{args.out_name}.npy")
    np.save(save_path_pairs, pairs)
    print(f"\n✅ 已保存 {len(pairs)} 对到 {save_path_pairs}")

    if args.save_full:
        payload = []
        for r in results:
            rec = {
                "p1": np.asarray(r.p1), "p2": np.asarray(r.p2),
                "n1": np.asarray(r.n1), "n2": np.asarray(r.n2),
                "width": float(r.width),
                "gscore": float(r.gscore),
                "eps": float(r.eps),
                "wscore": float(r.wscore),
                "total": float(r.total),
            }
            payload.append(rec)
        save_path_full = os.path.join(out_dir, f"{args.out_name}_scored.npy")
        np.save(save_path_full, payload, allow_pickle=True)
        print(f" 已保存带分数的结果到 {save_path_full}")

if __name__ == "__main__":
    main()


# import numpy as np
# import trimesh
# import os
# from tqdm import trange

# def _unit(v, eps=1e-12):
#     n = np.linalg.norm(v)
#     return v if n < eps else v / n

# def is_antipodal(p1, n1, p2, n2, mu):
#     """检查是否满足摩擦锥下的antipodal条件"""
#     u = p2 - p1
#     d = np.linalg.norm(u)
#     if d < 1e-6:
#         return False
#     u /= d

#     theta = np.arctan(mu)
#     cos_th = np.cos(theta)

#     cond1 = np.dot(-u, n1) >= cos_th
#     cond2 = np.dot(u, n2)  >= cos_th
#     return cond1 and cond2

# def sample_antipodal_pairs_ray(mesh,
#                                mu=0.5,
#                                num_samples=20000,
#                                wmin=None, wmax=None,
#                                require_through_com=True):
#     """从表面点出发射线找对置点，生成antipodal抓取对"""
#     points, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
#     normals = mesh.face_normals[face_idx]

#     com = mesh.center_mass
#     pairs = []

#     ex, ey, ez = mesh.extents
#     if wmin is None: wmin = 0.2 * min(ex, ey)
#     if wmax is None: wmax = 1.2 * max(ex, ey)

#     for i in trange(num_samples, desc="射线寻找antipodal对"):
#         p1 = points[i]
#         n1 = _unit(normals[i])

#         # 从p1沿 -n1 射线发射
#         origins = p1[None, :]
#         directions = (-n1)[None, :]
#         locs, idx_ray, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)

#         if len(locs) == 0:
#             continue

#         # 取最远交点作为另一侧
#         p2 = locs[np.argmax(np.linalg.norm(locs - p1, axis=1))]

#         # 查询p2的法向
#         _, _, fid = mesh.nearest.on_surface([p2])
#         n2 = mesh.face_normals[fid[0]]
#         n2 = _unit(n2)

#         w = np.linalg.norm(p2 - p1)
#         if w < wmin or w > wmax:
#             continue

#         # 检查摩擦锥条件
#         if not is_antipodal(p1, n1, p2, n2, mu):
#             continue

#         # 质心检查：连线必须穿过质心附近
#         if require_through_com:
#             u = (p2 - p1)
#             u /= np.linalg.norm(u) + 1e-12
#             v = com - p1
#             dist_line = np.linalg.norm(np.cross(u, v))
#             if dist_line > 0.1 * max(ex, ey, ez):  # 距离太远，认为不稳定
#                 continue

#         pairs.append((p1, p2, n1, n2))

#     return pairs


# if __name__ == "__main__":
#     results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/antipodal_pairs"))
#     os.makedirs(results_dir, exist_ok=True)
#     save_path = os.path.join(results_dir, "antipodal_pairs_ray.npy")

#     mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lego.obj"))
#     mesh = trimesh.load(mesh_path, force='mesh')
#     scale = 0.01
#     mesh.apply_scale(scale) 
#     print(f"Loaded mesh from {mesh_path}, vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

#     pairs = sample_antipodal_pairs_ray(mesh, mu=0.5, num_samples=5000)
#     print(f"实际采样到 {len(pairs)} 对antipodal点 (ray method)")

#     if len(pairs) > 0:
#         np.save(save_path, pairs)
#         print(f"✅ 已保存 {len(pairs)} 对antipodal点到 {save_path}")
#     else:
#         print("❌ 未找到足够的antipodal点对")
# import numpy as np
# import trimesh
# from tqdm import trange
# from scipy.spatial import cKDTree
# import os

# def _unit(v, eps=1e-12):
#     n = np.linalg.norm(v)
#     return v if n < eps else v/n

# def _outward_fix_normals(points, normals):
#     C = points.mean(0)
#     vout = points - C
#     vout = vout / (np.linalg.norm(vout, axis=1, keepdims=True) + 1e-12)
#     flip = (np.sum(normals * vout, axis=1) < 0.0)
#     normals = normals.copy()
#     normals[flip] *= -1.0
#     return normals

# def _pca_vertex_normals(V, k=24):
#     kdt = cKDTree(V)
#     N = np.zeros_like(V)
#     C0 = V.mean(0)
#     for i,p in enumerate(V):
#         _, idx = kdt.query(p, k=min(k, len(V)))
#         Q = V[idx] - p
#         C = Q.T @ Q
#         w, U = np.linalg.eigh(C)
#         n = U[:,0]
#         if np.dot(n, V[i]-C0) < 0: n = -n
#         N[i] = _unit(n)
#     return N

# def is_antipodal(p1, n1, p2, n2, mu):
#     u = p2 - p1
#     d = np.linalg.norm(u)
#     if d < 1e-6: 
#         return False
#     u /= d
#     cos_th = 1.0/np.sqrt(1.0 + mu*mu)
#     c1 = np.dot(u, n1) <= -cos_th
#     c2 = np.dot(u, n2) >=  cos_th
#     return bool(c1 and c2)

# def sample_antipodal_pairs_side(mesh,
#                                 mu=0.5,
#                                 approach_dir=np.array([1.0,0.0,0.0]),
#                                 angle_thresh=np.deg2rad(80),   # 只采侧面，夹角接近90°
#                                 num_pairs=10000,
#                                 num_surface_samples=60000,
#                                 wmin=None, wmax=None,
#                                 use_pca_normals=True):
#     points, fids = trimesh.sample.sample_surface(mesh, num_surface_samples)
#     if use_pca_normals:
#         V = mesh.vertices.view(np.ndarray)
#         VN = _pca_vertex_normals(V, k=24)
#         kdt = cKDTree(V)
#         _, idx = kdt.query(points)
#         FN = VN[idx]
#         FN = _outward_fix_normals(points, FN)
#     else:
#         FN = mesh.face_normals[fids]
#         FN = _outward_fix_normals(points, FN)

#     # 指间距范围
#     ex, ey, ez = mesh.extents
#     if wmin is None: wmin = 0.2 * min(ex, ey)
#     if wmax is None: wmax = 0.95 * max(ex, ey)

#     # 只保留侧面点
#     ad = _unit(np.asarray(approach_dir, float))
#     dot = FN @ ad
#     cos_th = np.cos(angle_thresh)  # angle_thresh接近90°，cos很小
#     side_idx = np.where(np.abs(dot) < cos_th)[0]
#     if len(side_idx) < 2:
#         print("未找到足够的侧面点")
#         return []

#     pairs = []
#     rng = np.random.default_rng()
#     max_trials = num_pairs * 8

#     for _ in trange(max_trials, desc="采样侧面antipodal点对"):
#         i, j = rng.choice(side_idx, size=2, replace=False)
#         p1, n1 = points[i], FN[i]
#         p2, n2 = points[j], FN[j]
#         # 过滤非法法向
#         if (np.linalg.norm(n1) < 1e-8 or np.linalg.norm(n2) < 1e-8 or
#             not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)) or
#             not np.all(np.isfinite(n1)) or not np.all(np.isfinite(n2))):
#             continue
#         w = np.linalg.norm(p2 - p1)
#         if w < wmin or w > wmax:
#             continue
#         if is_antipodal(p1, n1, p2, n2, mu):
#             pairs.append((p1, p2, n1, n2))
#             if len(pairs) >= num_pairs:
#                 break

#     if len(pairs) == 0:
#         T = mesh.bounding_box_oriented.primitive.transform
#         R = T[:3,:3]; t = T[:3,3]
#         ext = mesh.bounding_box_oriented.primitive.extents
#         grid = 10
#         axes = [(0, ext[0]), (1, ext[1])]
#         for ax, L in axes:
#             if len(pairs) >= num_pairs: break
#             for u in np.linspace(-0.5, 0.5, grid):
#                 for v in np.linspace(-0.5, 0.5, grid):
#                     loc = np.zeros(3)
#                     loc[ax] = +0.5*ext[ax]
#                     oth = [0,1,2]
#                     oth.remove(ax)
#                     loc[oth[0]] = u * ext[oth[0]]
#                     loc[oth[1]] = v * ext[oth[1]]
#                     pR = R@loc + t
#                     nR = R[:,ax]
#                     loc[ax] = -0.5*ext[ax]
#                     pL = R@loc + t
#                     nL = -R[:,ax]
#                     w = np.linalg.norm(pR - pL)
#                     if wmin <= w <= wmax and is_antipodal(pL, nL, pR, nR, mu):
#                         pairs.append((pL, pR, nL, nR))
#                         if len(pairs) >= num_pairs:
#                             break
#                 if len(pairs) >= num_pairs:
#                     break

#     return pairs

# if __name__ == "__main__":
#     results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/antipodal_pairs"))
#     os.makedirs(results_dir, exist_ok=True)
#     save_path = os.path.join(results_dir, "antipodal_pairs_side.npy")

#     mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lego.obj"))
#     mesh = trimesh.load(mesh_path)
#     print(f"Loaded mesh from {mesh_path}")

#     pairs = sample_antipodal_pairs_side(mesh, mu=0.5, num_pairs=10000, num_surface_samples=60000)
#     print(f"实际采样到 {len(pairs)} 对侧面antipodal点")
#     if len(pairs) > 0:
#         np.save(save_path, pairs)
#         print(f"已保存 {len(pairs)} 对侧面antipodal点到 {save_path}")
#     else:
#         print("未找到足够的侧面antipodal点对")