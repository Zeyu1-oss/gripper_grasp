#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import trimesh
from dataclasses import dataclass
from tqdm import trange, tqdm
from scipy.optimize import minimize, nnls
from trimesh.transformations import rotation_matrix

# ===== 数据结构 =====
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

# ===== 小工具 =====
def _unit(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def _orthonormal_tangent_basis(axis):
    axis = _unit(axis)
    up = np.array([0.,0.,1.]) if abs(np.dot(axis, [0,0,1]))<0.95 else np.array([0.,1.,0.])
    t1 = _unit(np.cross(axis, up))
    t2 = _unit(np.cross(axis, t1))
    return t1, t2

def _build_ray_engine(mesh):
    return (trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
            if trimesh.ray.has_embree else
            trimesh.ray.ray_triangle.RayMeshIntersector(mesh))

# 几何分（保持原版）
def geom_score_pair(p, q, n_p, n_q, mu, wmin, wmax):
    v = q - p; d = np.linalg.norm(v)
    if d < 1e-9: return -1e9
    u = v / d
    cos_th = np.cos(np.arctan(mu))
    m_p = (-cos_th - np.dot(n_p, u))
    m_q = ( np.dot(n_q, u) - cos_th)
    sp = lambda x: np.log1p(np.exp(x))
    margin = sp(m_p) + sp(m_q)
    anti = -float(np.dot(n_p, n_q))
    if d < wmin: w_pen = (wmin - d)**2
    elif d > wmax: w_pen = (d - wmax)**2
    else:
        mid = 0.5*(wmin+wmax)
        w_pen = 0.25*((d - mid)/(0.5*(wmax-wmin)+1e-9))**2
    return 1.0*margin + 0.3*anti - 0.3*w_pen

# 力闭合
def contact_wrench_rays(p, n, mu, m_dirs, com, torque_scale):
    t1, t2 = _orthonormal_tangent_basis(n)
    r = p - com
    cols = []
    for k in range(m_dirs):
        phi = 2*np.pi*k/m_dirs
        t = np.cos(phi)*t1 + np.sin(phi)*t2
        f = _unit(n + mu*t)
        tau = np.cross(r, f) / (torque_scale + 1e-12)
        cols.append(np.hstack([f, tau]))
    return np.column_stack(cols)

def epsilon_qp(W):
    m = W.shape[1]
    H = W.T @ W
    fun = lambda lam: 0.5 * lam @ H @ lam
    jac = lambda lam: H @ lam
    cons = [{'type': 'eq','fun':lambda lam: np.sum(lam)-1.0,
             'jac':lambda lam: np.ones_like(lam)}]
    bnds = [(0.0,None)]*m
    x0 = np.ones(m)/m
    res = minimize(fun, x0, method='SLSQP', jac=jac, bounds=bnds, constraints=cons,
                   options={'maxiter':200,'ftol':1e-9,'disp':False})
    if res.success:
        lam = res.x
        return float(np.linalg.norm(W @ lam)), lam, True
    # fallback
    rho = 10.0
    A = np.vstack([W, np.sqrt(rho)*np.ones((1,m))])
    b = np.zeros(A.shape[0]); b[-1] = np.sqrt(rho)
    lam,_ = nnls(A,b); s = np.sum(lam)
    if s>1e-12: lam=lam/s
    return float(np.linalg.norm(W @ lam)), lam, False

def wrench_quality_for_pair(p, q, n_p, n_q, mesh, mu=0.6, m_dirs=8):
    com = mesh.center_mass if mesh.is_watertight else 0.5*(mesh.bounds[0]+mesh.bounds[1])
    Lc = float(np.linalg.norm(mesh.extents))
    W1 = contact_wrench_rays(p,n_p,mu,m_dirs,com,Lc)
    W2 = contact_wrench_rays(q,n_q,mu,m_dirs,com,Lc)
    W  = np.concatenate([W1,W2],axis=1)
    eps,_,_ = epsilon_qp(W)
    return eps, 1.0/(eps+1e-6)

# ===== 主流程 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=str, default="../lego.obj")
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--out_dir", type=str, default="../results/antipodal_pairs")
    ap.add_argument("--out_name", type=str, default="antipodal_pairs_ray")

    # 手爪宽度约束
    ap.add_argument("--gripper_min_width", type=float, default=None, help="手爪最小开度 (m)")
    ap.add_argument("--gripper_max_width", type=float, default=None, help="手爪最大开度 (m)")

    ap.add_argument("--num_samples", type=int, default=30000)
    ap.add_argument("--max_keep", type=int, default=8000)
    ap.add_argument("--topk", type=int, default=4000)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--m_dirs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # mesh
    mesh = trimesh.load(args.mesh, force='mesh')
    if isinstance(mesh,trimesh.Scene):
        mesh = mesh.dump().sum()
    if args.scale and args.scale!=1.0:
        mesh.apply_scale(float(args.scale))

    ex,ey,ez = mesh.extents
    if args.gripper_min_width is not None and args.gripper_max_width is not None:
        wmin, wmax = args.gripper_min_width, args.gripper_max_width
        print(f"使用手爪宽度约束: wmin={wmin:.4f}, wmax={wmax:.4f}")
    else:
        wmin = 0.2*float(min(ex,ey))
        wmax = 1.2*float(max(ex,ey))
        print(f"使用mesh比例自动推断: wmin={wmin:.4f}, wmax={wmax:.4f}")

    # 采样
    rng = np.random.default_rng(args.seed)
    points, face_idx = trimesh.sample.sample_surface(mesh, args.num_samples)
    normals = mesh.face_normals[face_idx]

    # 收集候选
    candidates = []
    for i in trange(len(points)):
        # ...这里你已有射线筛选逻辑...
        pass

    # 力闭合 + 排序
    results = []
    for (p1,p2,n1,n2,width,gscore) in candidates:
        eps,wscore = wrench_quality_for_pair(p1,p2,n1,n2,mesh,mu=args.mu,m_dirs=args.m_dirs)
        total = 0.1*gscore + 0.9*np.log1p(wscore)
        results.append(PairScored(p1,p2,n1,n2,width,gscore,eps,wscore,total))
    results.sort(key=lambda r:-r.total)

    # 保存
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir,exist_ok=True)
    out_path = os.path.join(out_dir,f"{args.out_name}_with_pose.npy")
    np.save(out_path, np.array(results, dtype=object), allow_pickle=True)
    print(f"✅ 已保存 {len(results)} 条 → {out_path}")

if __name__=="__main__":
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