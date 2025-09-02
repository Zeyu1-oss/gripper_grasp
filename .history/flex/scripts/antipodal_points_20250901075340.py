# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# Antipodal Grasp Generation Pipeline

# 流程概览：
# 1. 输入物体网格 (mesh)
#    - 读取并缩放 STL/OBJ/PLY
#    - 根据参数对 mesh 进行旋转 (rot_x, rot_y, rot_z)

# 2. 表面采样
#    - 在 mesh 上随机采样点 p1
#    - 获得法向 n1

# 3. 对射线找对偶点 (antipodal pair)
#    - 从 p1 沿 -n1 发射射线，找到另一侧交点 p2
#    - 获取法向 n2
#    - 过滤条件：
#        * 宽度在 [wmin, wmax] 范围内
#        * n1 与 n2 近似相反 (对向)
#        * 可选：抓取线穿过质心 (com)

# 4. 几何打分 (_geom_score_pair)
#    - 考虑：
#        * 摩擦锥 margin（点对进入摩擦锥的裕度）
#        * 反法向程度 (n1 · n2)
#        * 抓取宽度是否合理
# 5. 生成接触 wrenches (_contact_wrench_rays)
#    - 在每个接触点离散化摩擦锥方向：
#        f = normalize(n + μ t), 其中 t 在切平面一圈
#    - 计算对应力矩：
#        τ = (p - com) × f
#    - 每列向量：[f; τ] ∈ R^6

# 6. 构造 W 矩阵
#    - W = [w1, w2, …, wM] ∈ R^(6×M)
#    - 包含两个接触点所有摩擦锥方向的 wrenches

# 7. 力闭合评估 (_epsilon_qp)
#    - 解 Ferrari–Canny ε-metric：
#        ε = min_{λ≥0, ∑λ=1} || Wλ ||_2
#    - ε > 0 ⇒ 力闭合
#    - ε 越大，抓取越稳健
#    - 同时计算 wscore = 1/(ε+1e-6) 作为代价指标

# 8. 评分
#    - total = 0.1 * gscore + 0.9 * log(1+wscore)
#    - 按 total 排序，选前 topk 个候选

# 9. **输出结果**
#    - 每个抓取保存：
#        * 接触点 (p1, p2)
#        * 法向 (n1, n2)
#        * 宽度 width
#        * 抓取中心 center
#        * 姿态矩阵 R
#    - 附带 meta 信息：旋转矩阵 T_mesh, scale, 参数配置
#    - 保存.npy 文件
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Antipodal Grasp Generation Pipeline (with edge sampling)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.optimize import minimize, nnls


class AntipodalGraspGenerator:
    def __init__(self, mu=0.4, opp_angle_deg=30.0, m_dirs=8, seed=42,
                 require_through_com=True, through_tol_ratio=0.2):
        self.mu = mu
        self.opp_angle_deg = opp_angle_deg
        self.m_dirs = m_dirs
        self.rng = np.random.default_rng(seed)
        self.require_through_com = require_through_com
        self.through_tol_ratio = through_tol_ratio

    # ========== 工具 ==========
    def _unit(self, v, eps=1e-12):
        v = np.asarray(v, float)
        n = np.linalg.norm(v)
        return v if n < eps else v / n

    def _geom_score_pair(self, p1, p2, n1, n2, wmin, wmax):
        v = p2 - p1
        d = np.linalg.norm(v)
        if d < 1e-9: return -1e9
        u = v / d
        cos_th = np.cos(np.arctan(self.mu))
        m_p = (-cos_th - np.dot(n1, u))
        m_q = ( np.dot(n2, u) - cos_th)
        sp = lambda x: np.log1p(np.exp(x))
        margin = sp(m_p) + sp(m_q)
        anti = -float(np.dot(n1, n2))
        if d < wmin: w_pen = (wmin - d)**2
        elif d > wmax: w_pen = (d - wmax)**2
        else:
            mid = 0.5*(wmin+wmax)
            w_pen = 0.25*((d-mid)/(0.5*(wmax-wmin)+1e-9))**2
        return margin + 0.3*anti - 0.3*w_pen

    def _epsilon_qp(self, W):
        m = W.shape[1]
        H = W.T @ W
        fun = lambda lam: 0.5 * lam @ H @ lam
        jac = lambda lam: H @ lam
        cons = [{'type': 'eq', 'fun': lambda lam: np.sum(lam) - 1.0,
                 'jac': lambda lam: np.ones_like(lam)}]
        bnds = [(0.0, None)] * m
        x0 = np.ones(m)/m
        res = minimize(fun, x0, method='SLSQP', jac=jac, bounds=bnds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
        if res.success:
            return float(np.linalg.norm(W @ res.x))
        # fallback
        rho = 10.0
        A = np.vstack([W, np.sqrt(rho)*np.ones((1, m))])
        b = np.zeros(A.shape[0]); b[-1] = np.sqrt(rho)
        lam, _ = nnls(A, b)
        s = np.sum(lam)
        if s > 1e-12: lam = lam/s
        return float(np.linalg.norm(W @ lam))

    # ========== 均匀采样 ==========
    def _uniform_surface_samples(self, mesh, num_samples):
        """直接在 mesh 表面均匀采样"""
        points, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
        normals = mesh.face_normals[face_idx]
        return np.array(points), np.array(normals)

    def generate_for_mesh(self, mesh_path, scale=0.01,
                          num_samples=20000, topk=1000,
                          out_path=None):
        mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()
        if scale and scale != 1.0:
            mesh.apply_scale(scale)

        ex, ey, ez = mesh.extents
        wmin=0.005
        wmax=0.05
        cos_opp = np.cos(np.deg2rad(self.opp_angle_deg))
        eps_offset = 1e-5 * np.linalg.norm(mesh.extents)
        com = mesh.center_mass if mesh.is_watertight else mesh.centroid

        # 均匀表面采样
        points, normals = self._uniform_surface_samples(mesh, num_samples)
        ray_engine = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

        candidates=[]
        for i in tqdm(range(len(points)), desc=f"候选生成 {os.path.basename(mesh_path)}"):
            p1, n1 = points[i], normals[i]

            axis = -n1
            origins = np.array([p1 - eps_offset*n1])
            dirs = np.array([self._unit(axis)])
            locs, _, tri_ids = ray_engine.intersects_location(origins, dirs, multiple_hits=True)
            if len(locs)==0: continue
            p2 = locs[0]
            n2 = mesh.face_normals[tri_ids[0]]

            # 条件：对向 + 宽度
            if np.dot(n1,n2) > -cos_opp: continue
            width = float(np.linalg.norm(p2-p1))
            if width < wmin or width > wmax: continue

            # 可选：穿过质心
            if self.require_through_com:
                u = (p2 - p1) / (width + 1e-12)
                v = com - p1
                dist_line = np.linalg.norm(np.cross(u, v))
                if dist_line > self.through_tol_ratio * max(ex,ey,ez):
                    continue

            gscore = self._geom_score_pair(p1,p2,n1,n2,wmin,wmax)
            W = np.column_stack([np.hstack([n1,[0,0,0]]), np.hstack([n2,[0,0,0]])])
            eps = self._epsilon_qp(W)
            total = 0.4*gscore + 0.6*np.log1p(eps)

            candidates.append({
                "p1": p1, "p2": p2,
                "n1": n1, "n2": n2,
                "width": width,
                "gscore": gscore,
                "eps": eps,
                "total": total,
                "center": 0.5*(p1+p2)
            })

        candidates.sort(key=lambda r: -r["total"])
        results = candidates[:topk]

        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, np.array(results, dtype=object), allow_pickle=True)
            print(f"✅ 保存 {len(results)} 条抓取到 {out_path}")

        return results


# ========== main ==========
def main():
    ap = argparse.ArgumentParser("Antipodal grasp generator (均匀覆盖采样)")
    ap.add_argument("--meshes", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, default="../results/antipodal_pairs")
    ap.add_argument("--scale", type=float, default=1)
    ap.add_argument("--num_samples", type=int, default=20000)
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()

    gen = AntipodalGraspGenerator()
    os.makedirs(args.out_dir, exist_ok=True)

    for mesh_path in args.meshes:
        name = os.path.splitext(os.path.basename(mesh_path))[0]
        out_path = os.path.join(args.out_dir, f"{name}_pairs.npy")
        gen.generate_for_mesh(mesh_path,
                              scale=args.scale,
                              num_samples=args.num_samples,
                              topk=args.topk,
                              out_path=out_path)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import os
# import argparse
# import numpy as np
# import trimesh
# from dataclasses import dataclass
# from tqdm import trange, tqdm
# from scipy.optimize import minimize, nnls
# from trimesh.transformations import rotation_matrix

# # ======= 数据结构 =======

# @dataclass
# class PairScored:
#     p1: np.ndarray
#     p2: np.ndarray
#     n1: np.ndarray
#     n2: np.ndarray
#     width: float
#     gscore: float
#     eps: float
#     wscore: float
#     total: float

# # ======= 小工具 =======

# def _unit(v, eps=1e-12):
#     v = np.asarray(v, float)
#     n = np.linalg.norm(v)
#     return v if n < eps else v / n

# def _orthonormal_tangent_basis(axis):
#     axis = _unit(axis)
#     up = np.array([0.,0.,1.]) if abs(np.dot(axis, [0,0,1])) < 0.95 else np.array([0.,1.,0.])
#     t1 = _unit(np.cross(axis, up))
#     t2 = _unit(np.cross(axis, t1))
#     return t1, t2

# def _build_ray_engine(mesh):
#     return (trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
#             if trimesh.ray.has_embree else
#             trimesh.ray.ray_triangle.RayMeshIntersector(mesh))

# # —— 几何与打分 ——

# def geom_score_pair_single(p1, p2, n1, n2, mu, wmin, wmax):
#     """单射线模式下的几何分：只用 p2 侧的摩擦裕度 + 反法向 + 宽度贴合"""
#     v = p2 - p1; d = np.linalg.norm(v)
#     if d < 1e-9: return -1e9
#     u = v / d
#     cos_th = np.cos(np.arctan(mu))
#     # p1 侧天然满足，这里只算 p2 侧的 margin
#     m_q = (np.dot(n2, u) - cos_th)
#     sp = lambda x: np.log1p(np.exp(x))
#     margin = sp(m_q)
#     anti = -float(np.dot(n1, n2))  # 越“反”越大
#     if d < wmin: w_pen = (wmin - d)**2
#     elif d > wmax: w_pen = (d - wmax)**2
#     else:
#         mid = 0.5*(wmin+wmax)
#         w_pen = 0.25*((d - mid)/(0.5*(wmax-wmin)+1e-9))**2
#     return 1.0*margin + 0.3*anti - 0.3*w_pen

# def geom_score_pair_cone(p1, p2, n1, n2, mu, wmin, wmax):
#     """原扇形模式的几何分：两侧摩擦裕度 + 反法向 + 宽度贴合"""
#     v = p2 - p1; d = np.linalg.norm(v)
#     if d < 1e-9: return -1e9
#     u = v / d
#     cos_th = np.cos(np.arctan(mu))
#     m_p = (-cos_th - np.dot(n1, u))
#     m_q = ( np.dot(n2, u) - cos_th)
#     sp = lambda x: np.log1p(np.exp(x))
#     margin = sp(m_p) + sp(m_q)
#     anti = -float(np.dot(n1, n2))
#     if d < wmin: w_pen = (wmin - d)**2
#     elif d > wmax: w_pen = (d - wmax)**2
#     else:
#         mid = 0.5*(wmin+wmax)
#         w_pen = 0.25*((d - mid)/(0.5*(wmax-wmin)+1e-9))**2
#     return 1.0*margin + 0.3*anti - 0.3*w_pen

# # —— 力闭合评分（epsilon metric 近似） ——

# def contact_wrench_rays(p, n, mu, m_dirs, com, torque_scale):
#     """离散摩擦锥:f_dir = normalize(n + mu * t),t 在切平面一圈"""
#     t1, t2 = _orthonormal_tangent_basis(n)
#     r = p - com
#     cols = []
#     for k in range(m_dirs):
#         phi = 2*np.pi*k/m_dirs
#         t = np.cos(phi)*t1 + np.sin(phi)*t2
#         f = _unit(n + mu*t)
#         tau = np.cross(r, f) / (torque_scale + 1e-12)
#         cols.append(np.hstack([f, tau]))
#     return np.column_stack(cols)  # (6, m_dirs)

# def epsilon_qp(W):
#     m = W.shape[1]
#     H = W.T @ W
#     fun = lambda lam: 0.5 * lam @ H @ lam
#     jac = lambda lam: H @ lam
#     cons = [{'type': 'eq', 'fun': lambda lam: np.sum(lam) - 1.0,
#              'jac':  lambda lam: np.ones_like(lam)}]
#     bnds = [(0.0, None)] * m
#     x0 = np.ones(m)/m
#     res = minimize(fun, x0, method='SLSQP', jac=jac, bounds=bnds, constraints=cons,
#                    options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
#     if res.success:
#         lam = res.x
#         return float(np.linalg.norm(W @ lam)), lam, True
#     # fallback: NNLS
#     rho = 10.0
#     A = np.vstack([W, np.sqrt(rho)*np.ones((1, m))])
#     b = np.zeros(A.shape[0]); b[-1] = np.sqrt(rho)
#     lam, _ = nnls(A, b)
#     s = np.sum(lam)
#     if s > 1e-12: lam = lam/s
#     return float(np.linalg.norm(W @ lam)), lam, False

# def wrench_quality_for_pair(p, q, n_p, n_q, mesh, mu=0.6, m_dirs=8):
#     com = mesh.center_mass if mesh.is_watertight else 0.5*(mesh.bounds[0] + mesh.bounds[1])
#     Lc  = float(np.linalg.norm(mesh.extents))
#     W1 = contact_wrench_rays(p, n_p, mu, m_dirs, com, Lc)
#     W2 = contact_wrench_rays(q, n_q, mu, m_dirs, com, Lc)
#     W  = np.concatenate([W1, W2], axis=1)
#     eps, _, _ = epsilon_qp(W)
#     return eps, 1.0/(eps + 1e-6)

# # —— 将点对转抓取位姿（给下游用） ——

# def pair_to_pose(p1, p2, n1, n2):
#     x = _unit(p2 - p1)
#     nz = n1 + n2
#     if np.linalg.norm(nz) < 1e-8:
#         tmp = np.array([0.,0.,1.]) if abs(x[2]) < 0.9 else np.array([0.,1.,0.])
#         z = _unit(np.cross(tmp, x))
#     else:
#         z = -_unit(nz)
#         if abs(np.dot(z, x)) > 0.99:
#             tmp = np.array([0.,0.,1.]) if abs(x[2]) < 0.9 else np.array([0.,1.,0.])
#             z = _unit(np.cross(tmp, x))
#     y = _unit(np.cross(z, x))
#     z = _unit(np.cross(x, y))
#     R = np.column_stack([x, y, z])
#     # rot -> quat (wxyz)
#     tr = np.trace(R)
#     if tr > 0:
#         S = np.sqrt(tr + 1.0) * 2
#         w = 0.25 * S
#         xq = (R[2,1]-R[1,2]) / S
#         yq = (R[0,2]-R[2,0]) / S
#         zq = (R[1,0]-R[0,1]) / S
#     elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
#         S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
#         w  = (R[2,1]-R[1,2]) / S; xq = 0.25 * S
#         yq = (R[0,1]+R[1,0]) / S; zq = (R[0,2]+R[2,0]) / S
#     elif R[1,1] > R[2,2]:
#         S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
#         w  = (R[0,2]-R[2,0]) / S
#         xq = (R[0,1]+R[1,0]) / S; yq = 0.25 * S
#         zq = (R[1,2]+R[2,1]) / S
#     else:
#         S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
#         w  = (R[1,0]-R[0,1]) / S
#         xq = (R[0,2]+R[2,0]) / S; yq = (R[1,2]+R[2,1]) / S; zq = 0.25 * S
#     quat = np.array([w, xq, yq, zq], float)
#     quat /= (np.linalg.norm(quat) + 1e-12)
#     center = 0.5*(p1 + p2)
#     return center, R, quat

# # —— mesh 旋转

# def rotate_mesh_in_place(mesh, rot_x_deg=0.0, rot_y_deg=0.0, rot_z_deg=0.0,
#                          order='xyz', about='origin', pivot=None):
#     if about == 'origin':
#         p = np.array([0.,0.,0.], float)
#     elif about == 'centroid':
#         p = mesh.center_mass if mesh.is_watertight else mesh.centroid
#     elif about == 'custom':
#         if pivot is None: raise ValueError("about=custom 需要 pivot")
#         p = np.asarray(pivot, float)
#     else:
#         raise ValueError("rot_about ∈ {origin, centroid, custom}")
#     ax = {'x':np.array([1.,0.,0.]),
#           'y':np.array([0.,1.,0.]),
#           'z':np.array([0.,0.,1.])}
#     deg = {'x':rot_x_deg, 'y':rot_y_deg, 'z':rot_z_deg}
#     T_total = np.eye(4)
#     for ch in order.lower():
#         if ch not in 'xyz': continue
#         ang = float(deg[ch])
#         if abs(ang) < 1e-12: continue
#         T = rotation_matrix(np.deg2rad(ang), ax[ch], point=p)
#         mesh.apply_transform(T)
#         T_total = T @ T_total
#     return T_total  # 4x4


# def main():
#     ap = argparse.ArgumentParser("Generate antipodal pairs; support 'normal' (single ray) or 'cone' modes")
#     # IO
#     ap.add_argument("--mesh", type=str, default="../lego.obj")
#     ap.add_argument("--scale", type=float, default=0.01)
#     ap.add_argument("--out_dir", type=str, default="../results/lego_pairs")
#     ap.add_argument("--out_name", type=str, default="antipodal_pairs")

#     # 旋转
#     ap.add_argument("--rot_x_deg", type=float, default=0.0)
#     ap.add_argument("--rot_y_deg", type=float, default=90.0)
#     ap.add_argument("--rot_z_deg", type=float, default=0.0)
#     ap.add_argument("--rot_order", type=str, default="xyz")
#     ap.add_argument("--rot_about", type=str, default="origin", choices=["origin","centroid","custom"])
#     ap.add_argument("--rot_pivot", type=str, default="0,0,0")

#     # 采样/过滤/打分
#     ap.add_argument("--num_samples", type=int, default=50000)
#     ap.add_argument("--max_keep", type=int, default=50000)
#     ap.add_argument("--topk", type=int, default=10000)
#     ap.add_argument("--mu", type=float, default=0.4)
#     ap.add_argument("--opp_angle_deg", type=float, default=30.0, help="保留反法向粗过滤")
#     ap.add_argument("--cone_half_deg", type=float, default=12.0, help="仅 cone 模式使用")
#     ap.add_argument("--rays_per_point", type=int, default=10, help="仅 cone 模式使用")
#     ap.add_argument("--m_dirs", type=int, default=8)

#     ap.add_argument("--bottom_ratio", type=float, default=0.05)
#     ap.add_argument("--top_ratio", type=float, default=0.05)
#     ap.add_argument("--wmin", type=float, default=None)
#     ap.add_argument("--wmax", type=float, default=None)

#     # 线穿质心
#     ap.add_argument("--require_through_com", action="store_true", default=True)
#     ap.add_argument("--no_require_through_com", dest="require_through_com", action="store_false")
#     ap.add_argument("--through_tol_ratio", type=float, default=0.2)

#     # 出口选择
#     g = ap.add_mutually_exclusive_group()
#     g.add_argument("--nearest_exit", action="store_true")
#     g.add_argument("--farthest_exit", action="store_true")

#     # 模式：normal=单射线（推荐），cone=原扇形
#     ap.add_argument("--ray_mode", type=str, choices=["normal","cone"], default="normal")

#     # 可选阈值
#     ap.add_argument("--eps_thresh", type=float, default=None)
#     ap.add_argument("--wscore_thresh", type=float, default=None)

#     # 其他
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()

#     # 加载 mesh
#     mesh_path = os.path.abspath(args.mesh)
#     if not os.path.exists(mesh_path): raise FileNotFoundError(mesh_path)
#     mesh = trimesh.load(mesh_path, force='mesh')
#     if isinstance(mesh, trimesh.Scene):
#         mesh = mesh.dump().sum()
#     if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
#         raise RuntimeError("Empty triangle list")

#     # 尺度
#     if args.scale and args.scale != 1.0:
#         mesh.apply_scale(float(args.scale))

#     # 旋转（生成前应用）
#     pivot = np.array([float(x) for x in args.rot_pivot.split(',')]) if args.rot_about=='custom' else None
#     T_mesh = rotate_mesh_in_place(mesh,
#                                   rot_x_deg=args.rot_x_deg,
#                                   rot_y_deg=args.rot_y_deg,
#                                   rot_z_deg=args.rot_z_deg,
#                                   order=args.rot_order,
#                                   about=args.rot_about,
#                                   pivot=pivot)

#     print(f"Loaded mesh: {mesh_path}")
#     print(f"  vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, watertight={mesh.is_watertight}")
#     print(f"  extents={mesh.extents}, z-bounds=[{mesh.bounds[0,2]:.4f},{mesh.bounds[1,2]:.4f}]")
#     print(f"  rotated: order={args.rot_order}, about={args.rot_about}, "
#           f"deg=({args.rot_x_deg},{args.rot_y_deg},{args.rot_z_deg}), scale={args.scale}")

#     # 采样准备
#     rng = np.random.default_rng(args.seed)
#     ex, ey, ez = mesh.extents
#     wmin = args.wmin if args.wmin is not None else 0.2*float(min(ex,ey))
#     wmax = args.wmax if args.wmax is not None else 1.2*float(max(ex,ey))
#     eps_offset = 1e-5 * float(np.linalg.norm(mesh.extents))
#     cos_opp = np.cos(np.deg2rad(args.opp_angle_deg))
#     theta_ray = np.deg2rad(args.cone_half_deg)
#     ray_engine = _build_ray_engine(mesh)
#     zmin, zmax = float(mesh.bounds[0,2]), float(mesh.bounds[1,2])
#     height = zmax - zmin
#     side_zmin = zmin + args.bottom_ratio * height
#     side_zmax = zmax - args.top_ratio * height
#     if args.nearest_exit and args.farthest_exit:
#         raise ValueError("不能同时指定 --nearest_exit 和 --farthest_exit")
#     use_nearest_exit = mesh.is_watertight if (not args.nearest_exit and not args.farthest_exit) else args.nearest_exit

#     # 采样点与法向
#     points, face_idx = trimesh.sample.sample_surface(mesh, args.num_samples)
#     normals = mesh.face_normals[face_idx]
#     normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12, None)

#     # 收集候选
#     candidates = []
#     com = mesh.center_mass if mesh.is_watertight else 0.5*(mesh.bounds[0] + mesh.bounds[1])

#     for i in trange(args.num_samples, desc="生成候选(射线)"):
#         p1 = points[i]; n1 = normals[i]
#         if not (side_zmin < float(p1[2]) < side_zmax):
#             continue

#         if args.ray_mode == "normal":
#             # —— 单射线：沿 -n1 打
#             axis = -n1
#             origins = np.array([p1 - eps_offset*n1])
#             dirs    = np.array([_unit(axis)])
#             locs, idx_ray, tri_ids = ray_engine.intersects_location(origins, dirs, multiple_hits=True)
#             if len(locs) == 0 or len(idx_ray) == 0:
#                 continue

#             # 选择最近“出口”，但要有最小厚度
#             hit_pts  = locs
#             hit_tris = [tri_ids[h] for h in range(len(tri_ids))]
#             dists    = np.linalg.norm(hit_pts - p1, axis=1)
#             t_min = max(5.0*eps_offset, 0.25*wmin)
#             mask = dists >= t_min
#             if not np.any(mask):
#                 continue
#             k_local = int(np.argmin(dists[mask]))
#             # 还原到原索引
#             valid_idxs = np.where(mask)[0]
#             k_local = valid_idxs[k_local]

#             p2 = hit_pts[k_local]
#             if not (side_zmin < float(p2[2]) < side_zmax):
#                 continue
#             tri_k = hit_tris[k_local]
#             n2 = _unit(mesh.face_normals[tri_k])

#             width = float(np.linalg.norm(p2 - p1))
#             if width < wmin or width > wmax:
#                 continue

#             u = (p2 - p1) / (width + 1e-12)
#             # 只检查 p2 侧摩擦锥：<u, n2> >= cos(arctan(mu))
#             if np.dot(u, n2) < np.cos(np.arctan(args.mu)):
#                 continue
#             # 可选：仍保留反法向粗过滤（更干净）
#             if np.dot(n1, n2) > -cos_opp:
#                 continue

#             gscore = geom_score_pair_single(p1, p2, n1, n2, args.mu, wmin, wmax)

#         else:
#             # —— 原扇形：围绕 -n1 多条
#             axis = -n1
#             t1, t2 = _orthonormal_tangent_basis(axis)
#             phis = rng.uniform(0.0, 2.0*np.pi, size=args.rays_per_point)
#             dirs = []
#             for phi in phis:
#                 d = axis*np.cos(theta_ray) + (np.cos(phi)*t1 + np.sin(phi)*t2)*np.sin(theta_ray)
#                 dirs.append(_unit(d))
#             origins = np.repeat((p1 - eps_offset*n1)[None, :], len(dirs), axis=0)

#             locs, idx_ray, tri_ids = ray_engine.intersects_location(origins, np.array(dirs), multiple_hits=True)
#             if len(locs) == 0 or len(idx_ray) == 0:
#                 continue

#             by_ray = {}
#             for hit_idx, rid in enumerate(idx_ray):
#                 by_ray.setdefault(rid, []).append(hit_idx)

#             p2 = None; n2 = None; width = None; gscore = None
#             for rid, hit_list in by_ray.items():
#                 hit_pts  = locs[hit_list]
#                 hit_tris = [tri_ids[h] for h in hit_list]
#                 dists    = np.linalg.norm(hit_pts - p1, axis=1)

#                 if use_nearest_exit:
#                     k_local = int(np.argmin(dists))
#                 else:
#                     t_min = max(5.0*eps_offset, 0.25*wmin)
#                     mask = dists >= t_min
#                     if not np.any(mask):
#                         continue
#                     idxs = np.where(mask)[0]
#                     k_local = idxs[np.argmax(dists[idxs])]

#                 cand_p2 = hit_pts[k_local]
#                 if not (side_zmin < float(cand_p2[2]) < side_zmax):
#                     continue

#                 tri_k = hit_tris[k_local]
#                 cand_n2 = _unit(mesh.face_normals[tri_k])

#                 if np.dot(n1, cand_n2) > -cos_opp:
#                     continue

#                 cand_width = float(np.linalg.norm(cand_p2 - p1))
#                 if cand_width < wmin or cand_width > wmax:
#                     continue

#                 # 双侧摩擦锥
#                 u = (cand_p2 - p1) / (cand_width + 1e-12)
#                 cos_th = np.cos(np.arctan(args.mu))
#                 if not ((np.dot(-u, n1) >= cos_th) and (np.dot(u, cand_n2) >= cos_th)):
#                     continue

#                 cand_g = geom_score_pair_cone(p1, cand_p2, n1, cand_n2, args.mu, wmin, wmax)
#                 # 取这个 rid 的最佳
#                 if (gscore is None) or (cand_g > gscore):
#                     p2, n2, width, gscore = cand_p2, cand_n2, cand_width, cand_g

#             if p2 is None:
#                 continue

#         # 线穿质心
#         if args.require_through_com:
#             u = (p2 - p1) / (width + 1e-12)
#             v = com - p1
#             dist_line = np.linalg.norm(np.cross(u, v))
#             if dist_line > args.through_tol_ratio * max(ex, ey, ez):
#                 continue

#         # 中点在体内
#         if mesh.is_watertight:
#             mid = 0.5*(p1 + p2)
#             if not mesh.contains([mid])[0]:
#                 continue

#         candidates.append((p1, p2, n1, n2, width, gscore))
#         if len(candidates) >= args.max_keep:
#             break

#     if len(candidates) == 0:
#         print("⚠️ 没有几何可行的候选。可以尝试：放宽 opp_angle / wmin,wmax / 取消 through_com。")
#         return

#     # 力闭合 + 总分 + 阈值
#     results = []
#     for (p1, p2, n1, n2, width, gscore) in tqdm(candidates, desc="力闭合打分"):
#         eps_fc, wscore = wrench_quality_for_pair(p1, p2, n1, n2, mesh, mu=args.mu, m_dirs=args.m_dirs)
#         if (args.eps_thresh is not None) and (eps_fc > float(args.eps_thresh)):
#             continue
#         if (args.wscore_thresh is not None) and (wscore < float(args.wscore_thresh)):
#             continue
#         total = 0.1*gscore + 0.9*np.log1p(wscore)
#         results.append(PairScored(p1=p1, p2=p2, n1=n1, n2=n2,
#                                   width=width, gscore=gscore, eps=eps_fc,
#                                   wscore=wscore, total=total))
#     results.sort(key=lambda r: -r.total)
#     if args.topk and args.topk > 0:
#         results = results[:args.topk]

#     # 保存（每条带 center / R / quat），最后追加 meta（mesh 旋转）
#     out_dir = os.path.abspath(args.out_dir); os.makedirs(out_dir, exist_ok=True)
#     pairs_with_pose = []
#     for r in results:
#         center, R_pose, quat = pair_to_pose(r.p1, r.p2, r.n1, r.n2)
#         pairs_with_pose.append({
#             "p1": r.p1, "p2": r.p2, "n1": r.n1, "n2": r.n2,
#             "center": center, "R": R_pose, "quat_wxyz": quat
#         })
#     meta = {
#         "__meta__": True,
#         "T_mesh_4x4": T_mesh,
#         "rot_order": args.rot_order,
#         "rot_about": args.rot_about,
#         "rot_deg": (args.rot_x_deg, args.rot_y_deg, args.rot_z_deg),
#         "scale": args.scale,
#         "ray_mode": args.ray_mode
#     }
#     payload = np.array(pairs_with_pose + [meta], dtype=object)
#     out_path = os.path.join(out_dir, f"{args.out_name}.npy")
#     np.save(out_path, payload, allow_pickle=True)
#     print(f"\n✅ 已保存 {len(pairs_with_pose)} 条（含每条 rotation/pose)→ {out_path}")

# if __name__ == "__main__":
#     main()

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