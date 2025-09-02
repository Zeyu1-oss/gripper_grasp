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


class AntipodalGraspGenerator:
    def __init__(self, mu=0.4, seed=42):
        self.mu = mu
        self.rng = np.random.default_rng(seed)

    # ---------- 工具函数 ----------
    def _unit(self, v, eps=1e-12):
        v = np.asarray(v, float)
        n = np.linalg.norm(v)
        return v if n < eps else v / n

    def _orthonormal_tangent_basis(self, axis):
        axis = self._unit(axis)
        up = np.array([0.,0.,1.]) if abs(np.dot(axis,[0,0,1]))<0.95 else np.array([0.,1.,0.])
        t1 = self._unit(np.cross(axis, up))
        t2 = self._unit(np.cross(axis, t1))
        return t1, t2

    def _build_ray_engine(self, mesh):
        return (trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
                if trimesh.ray.has_embree else
                trimesh.ray.ray_triangle.RayMeshIntersector(mesh))

    def _geom_score(self, p1, p2, n1, n2, wmin, wmax):
        v = p2 - p1; d = np.linalg.norm(v)
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

    def _contact_wrench_rays(self, p, n, m_dirs, com, torque_scale):
        t1, t2 = self._orthonormal_tangent_basis(n)
        r = p - com
        cols = []
        for k in range(m_dirs):
            phi = 2*np.pi*k/m_dirs
            t = np.cos(phi)*t1 + np.sin(phi)*t2
            f = self._unit(n + self.mu*t)
            tau = np.cross(r, f) / (torque_scale + 1e-12)
            cols.append(np.hstack([f, tau]))
        return np.column_stack(cols)

    def _epsilon_qp(self, W):
        m = W.shape[1]
        H = W.T @ W
        fun = lambda lam: 0.5*lam @ H @ lam
        jac = lambda lam: H @ lam
        cons = [{'type': 'eq', 'fun': lambda lam: np.sum(lam) - 1.0,
                 'jac': lambda lam: np.ones_like(lam)}]
        bnds = [(0.0, None)] * m
        x0 = np.ones(m)/m
        res = minimize(fun, x0, method='SLSQP', jac=jac, bounds=bnds, constraints=cons,
                       options={'maxiter':200,'ftol':1e-9,'disp':False})
        if res.success:
            lam = res.x
            return float(np.linalg.norm(W @ lam))
        # fallback
        rho = 10.0
        A = np.vstack([W, np.sqrt(rho)*np.ones((1,m))])
        b = np.zeros(A.shape[0]); b[-1] = np.sqrt(rho)
        lam, _ = nnls(A,b); s = np.sum(lam)
        if s > 1e-12: lam = lam/s
        return float(np.linalg.norm(W @ lam))

    def _wrench_quality(self, p1, p2, n1, n2, mesh, m_dirs=8):
        com = mesh.center_mass if mesh.is_watertight else 0.5*(mesh.bounds[0]+mesh.bounds[1])
        Lc = float(np.linalg.norm(mesh.extents))
        W1 = self._contact_wrench_rays(p1, n1, m_dirs, com, Lc)
        W2 = self._contact_wrench_rays(p2, n2, m_dirs, com, Lc)
        W = np.concatenate([W1,W2], axis=1)
        eps = self._epsilon_qp(W)
        return eps, 1.0/(eps+1e-6)

    def _pair_to_pose(self, p1, p2, n1, n2):
        x = self._unit(p2 - p1)
        nz = n1 + n2
        if np.linalg.norm(nz) < 1e-8:
            tmp = np.array([0.,0.,1.]) if abs(x[2]) < 0.9 else np.array([0.,1.,0.])
            z = self._unit(np.cross(tmp, x))
        else:
            z = -self._unit(nz)
            if abs(np.dot(z, x)) > 0.99:
                tmp = np.array([0.,0.,1.]) if abs(x[2]) < 0.9 else np.array([0.,1.,0.])
                z = self._unit(np.cross(tmp, x))
        y = self._unit(np.cross(z, x))
        z = self._unit(np.cross(x, y))
        R = np.column_stack([x, y, z])
        # rot -> quat
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            xq = (R[2,1]-R[1,2]) / S
            yq = (R[0,2]-R[2,0]) / S
            zq = (R[1,0]-R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w  = (R[2,1]-R[1,2]) / S; xq = 0.25 * S
            yq = (R[0,1]+R[1,0]) / S; zq = (R[0,2]+R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w  = (R[0,2]-R[2,0]) / S
            xq = (R[0,1]+R[1,0]) / S; yq = 0.25 * S
            zq = (R[1,2]+R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w  = (R[1,0]-R[0,1]) / S
            xq = (R[0,2]+R[2,0]) / S; yq = (R[1,2]+R[2,1]) / S; zq = 0.25 * S
        quat = np.array([w, xq, yq, zq], float)
        quat /= (np.linalg.norm(quat) + 1e-12)
        center = 0.5*(p1 + p2)
        return center, R, quat

    def _rotate_mesh_in_place(self, mesh, rot_x=0.0, rot_y=0.0, rot_z=0.0,
                              order='xyz', about='origin', pivot=None):
        if about == 'origin':
            p = np.array([0.,0.,0.], float)
        elif about == 'centroid':
            p = mesh.center_mass if mesh.is_watertight else mesh.centroid
        elif about == 'custom':
            if pivot is None: raise ValueError("custom 需要 pivot")
            p = np.asarray(pivot, float)
        else:
            raise ValueError("rot_about ∈ {origin, centroid, custom}")
        ax = {'x':np.array([1.,0.,0.]),
              'y':np.array([0.,1.,0.]),
              'z':np.array([0.,0.,1.])}
        deg = {'x':rot_x, 'y':rot_y, 'z':rot_z}
        T_total = np.eye(4)
        for ch in order.lower():
            if ch not in 'xyz': continue
            ang = float(deg[ch])
            if abs(ang) < 1e-12: continue
            T = rotation_matrix(np.deg2rad(ang), ax[ch], point=p)
            mesh.apply_transform(T)
            T_total = T @ T_total
        return T_total

    # ---------- 主接口 ----------
    def generate_for_mesh(self, mesh_path, scale=0.01,
                          num_samples=50000, topk=1000,
                          rot=(0,0,0), rot_order='xyz',
                          out_path=None):
        mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(mesh, trimesh.Scene): mesh = mesh.dump().sum()
        if scale and scale != 1.0: mesh.apply_scale(scale)

        T_mesh = self._rotate_mesh_in_place(mesh, *rot, order=rot_order)

        ex,ey,ez = mesh.extents
        wmin, wmax = 0.2*min(ex,ey), 1.2*max(ex,ey)
        eps_offset = 1e-5*np.linalg.norm(mesh.extents)
        cos_opp = np.cos(np.deg2rad(30.0))

        points, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
        normals = mesh.face_normals[face_idx]
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12,None)

        ray_engine = self._build_ray_engine(mesh)
        candidates=[]
        for i in trange(num_samples, desc=f"候选 {os.path.basename(mesh_path)}"):
            p1,n1=points[i], normals[i]
            axis=-n1
            origins=np.array([p1 - eps_offset*n1])
            dirs=np.array([self._unit(axis)])
            locs, _, tri_ids=ray_engine.intersects_location(origins, dirs, multiple_hits=True)
            if len(locs)==0: continue
            p2=locs[0]; n2=self._unit(mesh.face_normals[tri_ids[0]])
            if np.dot(n1,n2)>-cos_opp: continue
            width=float(np.linalg.norm(p2-p1))
            if width<wmin or width>wmax: continue
            gscore=self._geom_score(p1,p2,n1,n2,wmin,wmax)
            candidates.append((p1,p2,n1,n2,width,gscore))
            if len(candidates)>=5000: break

        results=[]
        for (p1,p2,n1,n2,width,gscore) in tqdm(candidates, desc="打分"):
            eps_fc,wscore=self._wrench_quality(p1,p2,n1,n2,mesh)
            total=0.1*gscore+0.9*np.log1p(wscore)
            results.append(PairScored(p1,p2,n1,n2,width,gscore,eps_fc,wscore,total))
        results.sort(key=lambda r: -r.total)
        results=results[:topk]

        pairs=[{
            "p1": r.p1, "p2": r.p2, "n1": r.n1, "n2": r.n2,
            "center": self._pair_to_pose(r.p1,r.p2,r.n1,r.n2)[0],
            "R": self._pair_to_pose(r.p1,r.p2,r.n1,r.n2)[1],
            "quat_wxyz": self._pair_to_pose(r.p1,r.p2,r.n1,r.n2)[2]
        } for r in results]

        meta={"__meta__":True,"T_mesh_4x4":T_mesh,"scale":scale}
        payload=np.array(pairs+[meta],dtype=object)

        if out_path:
            np.save(out_path,payload,allow_pickle=True)
            print(f"✅ 保存 {len(pairs)} 条抓取到 {out_path}")

        return payload


# ---------- main ----------
def main():
    ap=argparse.ArgumentParser("Batch antipodal grasp generator")
    ap.add_argument("--meshes", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, default="../results/antipodal_pairs")
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--num_samples", type=int, default=50000)
    ap.add_argument("--topk", type=int, default=1000)
    args=ap.parse_args()

    gen=AntipodalGraspGenerator(mu=0.4)
    os.makedirs(args.out_dir, exist_ok=True)

    for mesh_path in args.meshes:
        name=os.path.splitext(os.path.basename(mesh_path))[0]
        out_path=os.path.join(args.out_dir,f"{name}_pairs.npy")
        gen.generate_for_mesh(mesh_path, scale=args.scale,
                              num_samples=args.num_samples,
                              topk=args.topk,
                              out_path=out_path)


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