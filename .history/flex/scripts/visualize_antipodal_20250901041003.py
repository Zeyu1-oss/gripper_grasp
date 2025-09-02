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

# ======= 数据结构 =======

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

# ======= 工具函数 =======

def _unit(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def _orthonormal_tangent_basis(axis):
    axis = _unit(axis)
    up = np.array([0., 0., 1.]) if abs(np.dot(axis, [0,0,1])) < 0.95 else np.array([0., 1., 0.])
    t1 = _unit(np.cross(axis, up))
    t2 = _unit(np.cross(axis, t1))
    return t1, t2

def _build_ray_engine(mesh):
    return (trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
            if trimesh.ray.has_embree else
            trimesh.ray.ray_triangle.RayMeshIntersector(mesh))

def is_antipodal(p1, n1, p2, n2, mu):
    u = p2 - p1
    d = np.linalg.norm(u)
    if d < 1e-9: return False
    u /= d
    theta = np.arctan(mu)
    cos_th = np.cos(theta)
    return (np.dot(-u, n1) >= cos_th) and (np.dot(u, n2) >= cos_th)

def geom_score_pair(p, q, n_p, n_q, mu, wmin, wmax):
    v = q - p
    d = np.linalg.norm(v)
    if d < 1e-9: return -1e9
    u = v / d
    theta = np.arctan(mu); cos_th = np.cos(theta)
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

def contact_wrench_rays(p, n, mu, m_dirs, com, torque_scale):
    t1, t2 = _orthonormal_tangent_basis(n)
    cols = []
    r = p - com
    for k in range(m_dirs):
        phi = 2*np.pi*k/m_dirs
        t = np.cos(phi)*t1 + np.sin(phi)*t2
        f = _unit(n + mu*t)
        tau = np.cross(r, f) / (torque_scale + 1e-12)
        cols.append(np.hstack([f, tau]))
    return np.column_stack(cols)  # (6, m_dirs)

def epsilon_qp(W):
    m = W.shape[1]
    H = W.T @ W
    fun = lambda lam: 0.5*lam @ H @ lam
    jac = lambda lam: H @ lam
    cons = [{'type':'eq','fun':lambda lam: np.sum(lam)-1.0,
             'jac':lambda lam: np.ones_like(lam)}]
    bnds = [(0.0, None)]*m
    x0 = np.ones(m)/m
    res = minimize(fun, x0, method='SLSQP', jac=jac, bounds=bnds, constraints=cons,
                   options={'maxiter':200,'ftol':1e-9,'disp':False})
    if res.success:
        lam = res.x
        return float(np.linalg.norm(W @ lam)), lam, True
    # fallback: NNLS 近似
    rho = 10.0
    A = np.vstack([W, np.sqrt(rho)*np.ones((1,m))])
    b = np.zeros(A.shape[0]); b[-1] = np.sqrt(rho)
    lam, _ = nnls(A, b)
    s = np.sum(lam)
    if s>1e-12: lam = lam/s
    return float(np.linalg.norm(W @ lam)), lam, False

def wrench_quality_for_pair(p, q, n_p, n_q, mesh, mu=0.6, m_dirs=8):
    com = mesh.center_mass if mesh.is_watertight else 0.5*(mesh.bounds[0] + mesh.bounds[1])
    Lc  = float(np.linalg.norm(mesh.extents))
    W1 = contact_wrench_rays(p, n_p, mu, m_dirs, com, Lc)
    W2 = contact_wrench_rays(q, n_q, mu, m_dirs, com, Lc)
    W  = np.concatenate([W1, W2], axis=1)
    eps, lam, ok = epsilon_qp(W)
    return eps, 1.0/(eps+1e-6)

def pair_to_pose(p1, p2, n1, n2):
    """把 (p1,p2,n1,n2) 变成手爪 pose: center、R(3x3)、quat_wxyz。"""
    x = _unit(p2 - p1)
    nz = n1 + n2
    if np.linalg.norm(nz) < 1e-8:
        tmp = np.array([0.,0.,1.]) if abs(x[2])<0.9 else np.array([0.,1.,0.])
        z = _unit(np.cross(tmp, x))
    else:
        z = -_unit(nz)
        if abs(np.dot(z, x)) > 0.99:
            tmp = np.array([0.,0.,1.]) if abs(x[2])<0.9 else np.array([0.,1.,0.])
            z = _unit(np.cross(tmp, x))
    y = _unit(np.cross(z, x))
    z = _unit(np.cross(x, y))
    R = np.column_stack([x, y, z])
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        xq = (R[2,1]-R[1,2]) / S
        yq = (R[0,2]-R[2,0]) / S
        zq = (R[1,0]-R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w  = (R[2,1]-R[1,2]) / S
        xq = 0.25 * S
        yq = (R[0,1]+R[1,0]) / S
        zq = (R[0,2]+R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w  = (R[0,2]-R[2,0]) / S
        xq = (R[0,1]+R[1,0]) / S
        yq = 0.25 * S
        zq = (R[1,2]+R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w  = (R[1,0]-R[0,1]) / S
        xq = (R[0,2]+R[2,0]) / S
        yq = (R[1,2]+R[2,1]) / S
        zq = 0.25 * S
    quat = np.array([w, xq, yq, zq], float)
    quat /= (np.linalg.norm(quat) + 1e-12)
    center = 0.5*(p1+p2)
    return center, R, quat

def rotate_mesh_in_place(mesh, rot_x_deg=0.0, rot_y_deg=0.0, rot_z_deg=0.0,
                         order='xyz', about='origin', pivot=None):
    if about == 'origin':
        p = np.array([0.,0.,0.], float)
    elif about == 'centroid':
        p = mesh.center_mass if mesh.is_watertight else mesh.centroid
    elif about == 'custom':
        if pivot is None: raise ValueError("about=custom 需要 pivot")
        p = np.asarray(pivot, float)
    else:
        raise ValueError("rot_about ∈ {origin, centroid, custom}")
    ax = {'x':np.array([1.,0.,0.]),
          'y':np.array([0.,1.,0.]),
          'z':np.array([0.,0.,1.])}
    deg = {'x':rot_x_deg, 'y':rot_y_deg, 'z':rot_z_deg}
    # 累乘用于写 meta
    T_total = np.eye(4)
    for ch in order.lower():
        if ch not in 'xyz': continue
        ang = float(deg[ch])
        if abs(ang) < 1e-12: continue
        T = rotation_matrix(np.deg2rad(ang), ax[ch], point=p)
        mesh.apply_transform(T)
        T_total = T @ T_total
    return T_total  # 4x4，总的 mesh 旋转(绕同一 pivot)

# ======= 主流程 =======

def main():
    ap = argparse.ArgumentParser("Generate antipodal pairs with per-sample rotation, after rotating mesh")
    # IO
    ap.add_argument("--mesh", type=str, default="../lego.obj")
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--out_dir", type=str, default="../results/antipodal_pairs")
    ap.add_argument("--out_name", type=str, default="antipodal_pairs_ray")

    # 旋转
    ap.add_argument("--rot_x_deg", type=float, default=0.0)
    ap.add_argument("--rot_y_deg", type=float, default=0.0)
    ap.add_argument("--rot_z_deg", type=float, default=0.0)
    ap.add_argument("--rot_order", type=str, default="xyz")
    ap.add_argument("--rot_about", type=str, default="origin", choices=["origin","centroid","custom"])
    ap.add_argument("--rot_pivot", type=str, default="0,0,0")

    # 采样/过滤/打分
    ap.add_argument("--num_samples", type=int, default=30000)
    ap.add_argument("--max_keep", type=int, default=8000)
    ap.add_argument("--topk", type=int, default=4000)
    ap.add_argument("--mu", type=float, default=0.6)
    ap.add_argument("--opp_angle_deg", type=float, default=30.0)
    ap.add_argument("--cone_half_deg", type=float, default=12.0)
    ap.add_argument("--rays_per_point", type=int, default=10)
    ap.add_argument("--m_dirs", type=int, default=8)
    ap.add_argument("--bottom_ratio", type=float, default=0.05)
    ap.add_argument("--top_ratio", type=float, default=0.05)
    ap.add_argument("--wmin", type=float, default=None)
    ap.add_argument("--wmax", type=float, default=None)
    ap.add_argument("--require_through_com", action="store_true", default=True)
    ap.add_argument("--no_require_through_com", dest="require_through_com", action="store_false")
    ap.add_argument("--through_tol_ratio", type=float, default=0.1)

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--nearest_exit", action="store_true")
    g.add_argument("--farthest_exit", action="store_true")

    ap.add_argument("--eps_thresh", type=float, default=None)
    ap.add_argument("--wscore_thresh", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 加载 mesh
    mesh_path = os.path.abspath(args.mesh)
    if not os.path.exists(mesh_path): raise FileNotFoundError(mesh_path)
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if mesh.vertices.shape[0]==0 or mesh.faces.shape[0]==0:
        raise RuntimeError("网格没有三角形（Empty triangle list）")

    # 尺度
    if args.scale and args.scale!=1.0:
        mesh.apply_scale(float(args.scale))

    # 旋转（生成前应用）
    pivot = np.array([float(x) for x in args.rot_pivot.split(',')]) if args.rot_about=='custom' else None
    T_mesh = rotate_mesh_in_place(mesh,
                                  rot_x_deg=args.rot_x_deg,
                                  rot_y_deg=args.rot_y_deg,
                                  rot_z_deg=args.rot_z_deg,
                                  order=args.rot_order,
                                  about=args.rot_about,
                                  pivot=pivot)

    print(f"Loaded mesh: {mesh_path}")
    print(f"  vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, watertight={mesh.is_watertight}")
    print(f"  extents={mesh.extents}, z-bounds=[{mesh.bounds[0,2]:.4f},{mesh.bounds[1,2]:.4f}]")
    print(f"  rotated: order={args.rot_order}, about={args.rot_about}, "
          f"deg=({args.rot_x_deg},{args.rot_y_deg},{args.rot_z_deg})")

    # 采样
    rng = np.random.default_rng(args.seed)
    ex, ey, ez = mesh.extents
    wmin = args.wmin if args.wmin is not None else 0.2*float(min(ex,ey))
    wmax = args.wmax if args.wmax is not None else 1.2*float(max(ex,ey))
    eps_offset = 1e-5 * float(np.linalg.norm(mesh.extents))
    cos_opp = np.cos(np.deg2rad(args.opp_angle_deg))
    theta_ray = np.deg2rad(args.cone_half_deg)
    ray_engine = _build_ray_engine(mesh)
    zmin, zmax = float(mesh.bounds[0,2]), float(mesh.bounds[1,2])
    height = zmax - zmin
    side_zmin = zmin + args.bottom_ratio * height
    side_zmax = zmax - args.top_ratio * height
    if args.nearest_exit and args.farthest_exit:
        raise ValueError("不能同时指定 --nearest_exit 和 --farthest_exit")
    use_nearest_exit = mesh.is_watertight if (not args.nearest_exit and not args.farthest_exit) else args.nearest_exit

    points, face_idx = trimesh.sample.sample_surface(mesh, args.num_samples)
    normals = mesh.face_normals[face_idx]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12,None)

    # 收集候选
    candidates = []
    com = mesh.center_mass if mesh.is_watertight else 0.5*(mesh.bounds[0]+mesh.bounds[1])
    for i in trange(args.num_samples, desc="生成候选(射线)"):
        p1 = points[i]; n1 = normals[i]
        if not (side_zmin < float(p1[2]) < side_zmax): continue

        axis = -n1
        t1, t2 = _orthonormal_tangent_basis(axis)
        phis = rng.uniform(0.0, 2.0*np.pi, size=args.rays_per_point)
        dirs = []
        for phi in phis:
            d = axis*np.cos(theta_ray) + (np.cos(phi)*t1 + np.sin(phi)*t2)*np.sin(theta_ray)
            dirs.append(_unit(d))
        origins = np.repeat((p1 - eps_offset*n1)[None,:], len(dirs), axis=0)

        locs, idx_ray, tri_ids = ray_engine.intersects_location(origins, np.array(dirs), multiple_hits=True)
        if len(locs)==0 or len(idx_ray)==0: continue

        by_ray = {}
        for hit_idx, rid in enumerate(idx_ray):
            by_ray.setdefault(rid, []).append(hit_idx)

        for rid, hit_list in by_ray.items():
            hit_pts  = locs[hit_list]
            hit_tris = [tri_ids[h] for h in hit_list]
            dists    = np.linalg.norm(hit_pts - p1, axis=1)

            if use_nearest_exit:
                k_local = int(np.argmin(dists))
            else:
                t_min = max(5.0*eps_offset, 0.25*wmin)
                mask = dists >= t_min
                if not np.any(mask): continue
                idxs = np.where(mask)[0]
                k_local = idxs[np.argmax(dists[idxs])]

            p2 = hit_pts[k_local]
            if not (side_zmin < float(p2[2]) < side_zmax): continue

            tri_k = hit_tris[k_local]
            n2 = _unit(mesh.face_normals[tri_k])

            if np.dot(n1, n2) > -cos_opp: continue

            width = float(np.linalg.norm(p2 - p1))
            if width < wmin or width > wmax: continue

            if not is_antipodal(p1, n1, p2, n2, args.mu): continue

            if args.require_through_com:
                u = (p2 - p1) / (width + 1e-12)
                v = com - p1
                dist_line = np.linalg.norm(np.cross(u, v))
                if dist_line > args.through_tol_ratio * max(ex,ey,ez):
                    continue

            if mesh.is_watertight:
                mid = 0.5*(p1+p2)
                if not mesh.contains([mid])[0]: continue

            gscore = geom_score_pair(p1, p2, n1, n2, args.mu, wmin, wmax)
            if gscore <= 0: continue

            candidates.append((p1, p2, n1, n2, width, gscore))
            if len(candidates) >= args.max_keep:
                break
        if len(candidates) >= args.max_keep:
            break

    if len(candidates)==0:
        print("⚠️ 没有几何可行的候选。可放宽 cone_half/opp_angle/mu 或高度窗口。")
        return

    # 力闭合评分 + 总分 + 阈值
    results = []
    for (p1,p2,n1,n2,width,gscore) in tqdm(candidates, desc="力闭合打分"):
        eps_fc, wscore = wrench_quality_for_pair(p1, p2, n1, n2, mesh, mu=args.mu, m_dirs=args.m_dirs)
        pass_eps = True if args.eps_thresh is None else (eps_fc <= float(args.eps_thresh))
        pass_wsc = True if args.wscore_thresh is None else (wscore >= float(args.wscore_thresh))
        if pass_eps and pass_wsc:
            total = 0.6*gscore + 0.4*np.log1p(wscore)
            results.append(PairScored(p1=p1, p2=p2, n1=n1, n2=n2,
                                      width=width, gscore=gscore, eps=eps_fc,
                                      wscore=wscore, total=total))
    results.sort(key=lambda r: -r.total)
    if args.topk and args.topk>0:
        results = results[:args.topk]

    # 保存（每条自带 rotation；同时写入一个 meta 记录 mesh 的整体旋转）
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 只保存 (p1,p2,n1,n2, center, R, quat) —— 便于后续可视化/下游
    pairs_with_pose = []
    for r in results:
        center, R_pose, quat = pair_to_pose(r.p1, r.p2, r.n1, r.n2)
        pairs_with_pose.append({
            "p1": r.p1, "p2": r.p2, "n1": r.n1, "n2": r.n2,
            "center": center, "R": R_pose, "quat_wxyz": quat
        })
    # 在最后追加一条 meta（可选），记录 mesh 的整体旋转变换，方便可视化时复现
    meta = {
        "__meta__": True,
        "T_mesh_4x4": T_mesh,  # 生成前对 mesh 施加的总变换
        "rot_order": args.rot_order,
        "rot_about": args.rot_about,
        "rot_deg": (args.rot_x_deg, args.rot_y_deg, args.rot_z_deg),
        "scale": args.scale
    }
    payload = np.array(pairs_with_pose + [meta], dtype=object)

    out_path = os.path.join(out_dir, f"{args.out_name}_with_pose.npy")
    np.save(out_path, payload, allow_pickle=True)
    print(f"\n✅ 已保存 {len(pairs_with_pose)} 条（含每条 rotation）→ {out_path}")

if __name__ == "__main__":
    main()
