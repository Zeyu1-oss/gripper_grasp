#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import trimesh
from dataclasses import dataclass
from tqdm import trange, tqdm
from scipy.optimize import minimize, nnls


# 数据结构
@dataclass
class PairScored:
    """单个抓取候选及其打分"""
    p1: np.ndarray
    p2: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    width: float
    gscore: float
    eps: float
    wscore: float
    total: float



class GraspPairGenerator:
    """
    两指抓取候选生成（几何过滤 + 力闭合打分）

    主要步骤：
    1) 在表面均匀采样 p1, n1;按 -n1 轴的锥形扇区打射线，找对侧点 p2, n2
    2) 几何硬过滤：反法向、宽度、摩擦锥、（可选）穿越质心、（可选）中点在体内
    3) 几何软评分 gscore(裕度 + 反法向 - 宽度罚）
    4) 力闭合评分：离散摩擦锥→扳手矩阵 W→最小凸包距离 ;wscore=1/(ε+1e-6)
    5) 综合总分 total = 0.6*gscore + 0.4*log1p(wscore)，排序取 topk
    """


    @staticmethod
    def _unit(v, eps=1e-12):
        n = np.linalg.norm(v)
        return v if n < eps else v / n

    @staticmethod
    def _orthonormal_tangent_basis(axis):
        axis = GraspPairGenerator._unit(axis)
        up = np.array([0., 0., 1.]) if abs(np.dot(axis, [0, 0, 1])) < 0.95 else np.array([0., 1., 0.])
        t1 = GraspPairGenerator._unit(np.cross(axis, up))
        t2 = GraspPairGenerator._unit(np.cross(axis, t1))
        return t1, t2

    @staticmethod
    def _build_ray_engine(mesh):
        return (trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
                if trimesh.ray.has_embree else
                trimesh.ray.ray_triangle.RayMeshIntersector(mesh))

    @staticmethod
    def is_antipodal(p1, n1, p2, n2, mu):
        """反对称-摩擦锥判据：沿 ±u 是否落在两侧摩擦锥内"""
        u = p2 - p1
        d = np.linalg.norm(u)
        if d < 1e-9:
            return False
        u /= d
        theta = np.arctan(mu)
        cos_th = np.cos(theta)
        return (np.dot(-u, n1) >= cos_th) and (np.dot(u, n2) >= cos_th)

    @staticmethod
    def geom_score_pair(p, q, n_p, n_q, mu, width_min, width_max):
        """
        几何打分（越大越好）：
          + 两侧摩擦锥“裕度”(softplus)
          + 反法向程度 (-n·n)
          - 宽度惩罚（区间外二次罚）
        """
        v = q - p
        d = np.linalg.norm(v)
        if d < 1e-9:
            return -1e9
        u = v / d
        theta = np.arctan(mu)
        cos_th = np.cos(theta)

        # 锥内裕度（值越大表示越“楔入”）
        m_p = (-cos_th - np.dot(n_p, u))     # p 侧看 -u
        m_q = ( np.dot(n_q, u) - cos_th)     # q 侧看 +u
        sp = lambda x: np.log1p(np.exp(x))   # softplus
        margin_term = sp(m_p) + sp(m_q)

        # 反法向
        anti_term = -float(np.dot(n_p, n_q))

        # 宽度惩罚
        if d < width_min:
            w_pen = (width_min - d) ** 2
        elif d > width_max:
            w_pen = (d - width_max) ** 2
        else:
            mid = 0.5 * (width_min + width_max)
            w_pen = 0.25 * ((d - mid) / (0.5 * (width_max - width_min) + 1e-9)) ** 2

        return 1.0 * margin_term + 0.3 * anti_term - 0.3 * w_pen

    #  力闭合wrench
    def _contact_wrench_rays(self, p, n, mu, m_dirs, com, torque_scale):
        """
        在切平面取 m_dirs 个方向，离散摩擦锥：
          f_dir = normalize(n + mu * t)
        返回 (6, m_dirs) 列向量，每列 [f; (r*f)/torque_scale]
        """
        t1, t2 = self._orthonormal_tangent_basis(n)
        cols = []
        r = p - com
        for k in range(m_dirs):
            phi = 2 * np.pi * k / m_dirs
            t = np.cos(phi) * t1 + np.sin(phi) * t2
            f = self._unit(n + mu * t)
            tau = np.cross(r, f) / (torque_scale + 1e-12)
            cols.append(np.hstack([f, tau]))
        return np.column_stack(cols)

    @staticmethod
    def _epsilon_qp(W):
        """
        Ferrari-Canny ε 质量的凸包近似：
           min ||W λ||  s.t. λ≥0, 1^T λ = 1
        先用 SLSQP,失败则退回 NNLS 近似。
        """
        m = W.shape[1]
        H = W.T @ W

        fun = lambda lam: 0.5 * lam @ H @ lam
        jac = lambda lam: H @ lam
        cons = [{'type': 'eq', 'fun': lambda lam: np.sum(lam) - 1.0,
                 'jac': lambda lam: np.ones_like(lam)}]
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

    def wrench_quality_for_pair(self, p, q, n_p, n_q, mesh, mu=0.6, m_dirs=8):
        """
        返回 (eps, wscore)
          eps 越小越好（越接近力闭合）
          wscore = 1/(eps+1e-6)
        """
        com = mesh.center_mass if mesh.is_watertight else 0.5 * (mesh.bounds[0] + mesh.bounds[1])
        Lc = float(np.linalg.norm(mesh.extents))
        W1 = self._contact_wrench_rays(p, n_p, mu, m_dirs, com, Lc)
        W2 = self._contact_wrench_rays(q, n_q, mu, m_dirs, com, Lc)
        W = np.concatenate([W1, W2], axis=1)  # (6, 2*m_dirs)
        eps, lam, ok = self._epsilon_qp(W)
        return eps, 1.0 / (eps + 1e-6)

    # ---------- 构造 & 运行 ----------

    def __init__(self,
                 mu=0.6,
                 num_samples=30000,
                 wmin=None, wmax=None,
                 opp_angle_deg=30.0,
                 cone_half_deg=12.0,
                 rays_per_point=10,
                 m_dirs=8,
                 bottom_ratio=0.05,
                 top_ratio=0.05,
                 side_only_norm_thresh=None,    # e.g., 0.3: 仅保留“近水平法向”的面（|n·z| < 0.3）
                 require_through_com=True,
                 through_tol_ratio=0.1,         # 线到质心距离阈值（物体最长边的比例）
                 nearest_exit=None,             # None: watertight->True else False；也可显式 True/False
                 eps_thresh=None,
                 wscore_thresh=None,
                 max_keep=8000,
                 topk=4000,
                 seed=42):
        self.mu = mu
        self.num_samples = num_samples
        self.wmin = wmin
        self.wmax = wmax
        self.opp_angle_deg = opp_angle_deg
        self.cone_half_deg = cone_half_deg
        self.rays_per_point = rays_per_point
        self.m_dirs = m_dirs
        self.bottom_ratio = bottom_ratio
        self.top_ratio = top_ratio
        self.side_only_norm_thresh = side_only_norm_thresh
        self.require_through_com = require_through_com
        self.through_tol_ratio = through_tol_ratio
        self.nearest_exit = nearest_exit
        self.eps_thresh = eps_thresh
        self.wscore_thresh = wscore_thresh
        self.max_keep = max_keep
        self.topk = topk
        self.rng = np.random.default_rng(seed)

    def generate(self, mesh):
        """
        生成候选(PairScored 列表）与拒绝统计 rej
        """
        ex, ey, ez = mesh.extents
        wmin = self.wmin if self.wmin is not None else 0.2 * float(min(ex, ey))
        wmax = self.wmax if self.wmax is not None else 1.2 * float(max(ex, ey))

        # 射线起点的小偏移（避免从面内起射）
        scale = float(np.linalg.norm(mesh.extents))
        eps_offset = 1e-5 * scale

        cos_opp = np.cos(np.deg2rad(self.opp_angle_deg))
        theta_ray = np.deg2rad(self.cone_half_deg)
        ray_engine = self._build_ray_engine(mesh)

        # 侧壁高度窗口（排除上下帽檐）
        zmin, zmax = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
        height = zmax - zmin
        side_zmin = zmin + self.bottom_ratio * height
        side_zmax = zmax - self.top_ratio * height

        # watertight 的默认路线
        if self.nearest_exit is None:
            use_nearest_exit = bool(mesh.is_watertight)
        else:
            use_nearest_exit = bool(self.nearest_exit)

        # 采样点与法向
        points, face_idx = trimesh.sample.sample_surface(mesh, self.num_samples)
        normals = mesh.face_normals[face_idx]
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12, None)

        # 统计拒绝原因
        rej = {
            'z_window': 0, 'side_norm': 0, 'opp_angle': 0, 'width': 0,
            'antipodal': 0, 'through_com': 0, 'inside_mid': 0, 'gscore<=0': 0
        }

        candidates = []
        com = mesh.center_mass if mesh.is_watertight else 0.5 * (mesh.bounds[0] + mesh.bounds[1])

        for i in trange(self.num_samples, desc="生成候选(射线)"):
            p1 = points[i]
            n1 = normals[i]

            # 高度窗口
            if not (side_zmin < float(p1[2]) < side_zmax):
                rej['z_window'] += 1
                continue

            # 侧面法向过滤（可选）
            if self.side_only_norm_thresh is not None:
                if abs(n1[2]) > float(self.side_only_norm_thresh):
                    rej['side_norm'] += 1
                    continue

            # 构造锥形射线
            axis = -n1
            t1, t2 = self._orthonormal_tangent_basis(axis)
            phis = self.rng.uniform(0.0, 2.0 * np.pi, size=self.rays_per_point)
            dirs = []
            for phi in phis:
                d = axis * np.cos(theta_ray) + (np.cos(phi) * t1 + np.sin(phi) * t2) * np.sin(theta_ray)
                dirs.append(self._unit(d))
            origins = np.repeat((p1 - eps_offset * n1)[None, :], len(dirs), axis=0)

            # 多命中处理
            locs, idx_ray, tri_ids = ray_engine.intersects_location(origins, np.array(dirs), multiple_hits=True)
            if len(locs) == 0 or len(idx_ray) == 0:
                continue

            by_ray = {}
            for hit_idx, rid in enumerate(idx_ray):
                by_ray.setdefault(rid, []).append(hit_idx)

            early_break = False
            for rid, hit_list in by_ray.items():
                hit_pts = locs[hit_list]
                hit_tris = [tri_ids[h] for h in hit_list]
                dists = np.linalg.norm(hit_pts - p1, axis=1)

                if use_nearest_exit:
                    k_local = int(np.argmin(dists))
                else:
                    t_min = max(5.0 * eps_offset, 0.25 * wmin)  # 厚度下限
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
                n2 = self._unit(mesh.face_normals[tri_k])

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
                if not self.is_antipodal(p1, n1, p2, n2, self.mu):
                    rej['antipodal'] += 1
                    continue

                # 质心附近穿越（可选）
                if self.require_through_com:
                    u = (p2 - p1) / (width + 1e-12)
                    v = com - p1
                    dist_line = np.linalg.norm(np.cross(u, v))
                    if dist_line > self.through_tol_ratio * max(ex, ey, ez):
                        rej['through_com'] += 1
                        continue

                # 中点在体内（仅 watertight）
                if mesh.is_watertight:
                    mid = 0.5 * (p1 + p2)
                    if not mesh.contains([mid])[0]:
                        rej['inside_mid'] += 1
                        continue

                # 几何分
                gscore = self.geom_score_pair(p1, p2, n1, n2, self.mu, wmin, wmax)
                if gscore <= 0:
                    rej['gscore<=0'] += 1
                    continue

                candidates.append((p1, p2, n1, n2, width, gscore))
                if len(candidates) >= self.max_keep:
                    early_break = True
                    break

            if early_break:
                break

        if len(candidates) == 0:
            print("⚠️  没有通过几何过滤的候选。建议：放宽 mu/opp_angle/cone_half_deg 或高度/法向窗口。")
            return [], rej

        # 力闭合评分 + 综合排序 +（可选）阈值过滤
        results = []
        passed_fc = 0
        for (p1, p2, n1, n2, width, gscore) in tqdm(candidates, desc="力闭合打分"):
            eps_fc, wscore = self.wrench_quality_for_pair(p1, p2, n1, n2, mesh, mu=self.mu, m_dirs=self.m_dirs)

            pass_eps = True if self.eps_thresh is None else (eps_fc <= float(self.eps_thresh))
            pass_wsc = True if self.wscore_thresh is None else (wscore >= float(self.wscore_thresh))
            if pass_eps and pass_wsc:
                passed_fc += 1
                total = 0.6 * gscore + 0.4 * np.log1p(wscore)
                results.append(PairScored(p1=p1, p2=p2, n1=n1, n2=n2,
                                          width=width, gscore=gscore, eps=eps_fc,
                                          wscore=wscore, total=total))

        results.sort(key=lambda r: -r.total)
        if self.topk is not None and self.topk > 0:
            results = results[:self.topk]

        print("\n—— 统计汇总 ——")
        print(f"  采样总数           : {self.num_samples}")
        print(f"  通过几何过滤候选数 : {len(candidates)} / {self.num_samples}")
        print(f"  通过力闭合阈值数   : {passed_fc} / {len(candidates)}")
        print("  几何拒绝原因：")
        for k, v in rej.items():
            print(f"    - {k:12s}: {v}")

        return results, rej

    # ---------- I/O 工具 ----------

    @staticmethod
    def save_pairs_only(results, path):
        """仅保存 (p1,p2,n1,n2)"""
        pairs = [(r.p1, r.p2, r.n1, r.n2) for r in results]
        np.save(path, pairs)
        return len(pairs)

    @staticmethod
    def save_scored(results, path):
        """保存带分数的结构化结果"""
        payload = []
        for r in results:
            payload.append({
                "p1": np.asarray(r.p1), "p2": np.asarray(r.p2),
                "n1": np.asarray(r.n1), "n2": np.asarray(r.n2),
                "width": float(r.width),
                "gscore": float(r.gscore),
                "eps": float(r.eps),
                "wscore": float(r.wscore),
                "total": float(r.total),
            })
        np.save(path, payload, allow_pickle=True)
        return len(payload)

    # ----------（可选）点对→6D位姿 ----------

    @staticmethod
    def pair_to_pose(p1, p2, n1, n2):
        """
        将 (p1,p2,n1,n2) 转成平行夹爪的 6D 位姿：
          - 位置：center = 0.5*(p1+p2)
          - 朝向：x轴= u=(p2-p1)/||·||（闭合轴）
                  z轴= -normalize(n1+n2)（接近轴；若极小则取与 u 垂直的稳定向量）
                  y轴= z × x
        返回：center (3,), R (3x3), quat_wxyz (4,)
        """
        u = GraspPairGenerator._unit(p2 - p1)
        nz = n1 + n2
        if np.linalg.norm(nz) < 1e-8:
            # 退化：取任一与 u 垂直的稳定方向
            tmp = np.array([0., 0., 1.]) if abs(u[2]) < 0.9 else np.array([0., 1., 0.])
            z = GraspPairGenerator._unit(np.cross(tmp, u))
        else:
            z = -GraspPairGenerator._unit(nz)
            # 保证与 u 不共线
            if abs(np.dot(z, u)) > 0.99:
                tmp = np.array([0., 0., 1.]) if abs(u[2]) < 0.9 else np.array([0., 1., 0.])
                z = GraspPairGenerator._unit(np.cross(tmp, u))
        x = u
        y = np.cross(z, x); y = GraspPairGenerator._unit(y)
        z = np.cross(x, y); z = GraspPairGenerator._unit(z)
        Rm = np.column_stack([x, y, z])  # 列为基
        # R -> quat (wxyz)
        # scipy 在你环境中已装，这里避免依赖：手写转换（数值稳定处理）
        tr = Rm[0,0] + Rm[1,1] + Rm[2,2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            xq = (Rm[2,1] - Rm[1,2]) / S
            yq = (Rm[0,2] - Rm[2,0]) / S
            zq = (Rm[1,0] - Rm[0,1]) / S
        elif (Rm[0,0] > Rm[1,1]) and (Rm[0,0] > Rm[2,2]):
            S = np.sqrt(1.0 + Rm[0,0] - Rm[1,1] - Rm[2,2]) * 2
            w = (Rm[2,1] - Rm[1,2]) / S
            xq = 0.25 * S
            yq = (Rm[0,1] + Rm[1,0]) / S
            zq = (Rm[0,2] + Rm[2,0]) / S
        elif Rm[1,1] > Rm[2,2]:
            S = np.sqrt(1.0 + Rm[1,1] - Rm[0,0] - Rm[2,2]) * 2
            w = (Rm[0,2] - Rm[2,0]) / S
            xq = (Rm[0,1] + Rm[1,0]) / S
            yq = 0.25 * S
            zq = (Rm[1,2] + Rm[2,1]) / S
        else:
            S = np.sqrt(1.0 + Rm[2,2] - Rm[0,0] - Rm[1,1]) * 2
            w = (Rm[1,0] - Rm[0,1]) / S
            xq = (Rm[0,2] + Rm[2,0]) / S
            yq = (Rm[1,2] + Rm[2,1]) / S
            zq = 0.25 * S
        center = 0.5 * (p1 + p2)
        quat_wxyz = np.array([w, xq, yq, zq], dtype=float)
        quat_wxyz = quat_wxyz / (np.linalg.norm(quat_wxyz) + 1e-12)
        return center, Rm, quat_wxyz


# ===================== CLI =====================

def parse_args():
    ap = argparse.ArgumentParser(description="两指抓取候选（Class 版本）")
    # IO
    ap.add_argument("--mesh", type=str, default="../lego.obj", help="网格路径（OBJ/STL/PLY 等）")
    ap.add_argument("--scale", type=float, default=0.01, help="统一尺度比例（例如 0.01 把 cm 变 m）")
    ap.add_argument("--out_dir", type=str, default="../results/antipodal_pairs", help="输出目录")
    ap.add_argument("--out_name", type=str, default="antipodal_pairs_ray", help="输出文件名（不含后缀）")
    ap.add_argument("--save_full", action="store_true", help="额外保存包含分数的结构化结果")

    # 采样/过滤
    ap.add_argument("--num_samples", type=int, default=30000)
    ap.add_argument("--max_keep", type=int, default=30000)
    ap.add_argument("--topk", type=int, default=10000)
    ap.add_argument("--mu", type=float, default=0.6)
    ap.add_argument("--opp_angle_deg", type=float, default=30.0)
    ap.add_argument("--cone_half_deg", type=float, default=12.0)
    ap.add_argument("--rays_per_point", type=int, default=10)
    ap.add_argument("--m_dirs", type=int, default=8)
    ap.add_argument("--bottom_ratio", type=float, default=0.05)
    ap.add_argument("--top_ratio", type=float, default=0.05)
    ap.add_argument("--wmin", type=float, default=None)
    ap.add_argument("--wmax", type=float, default=None)
    ap.add_argument("--side_only_norm_thresh", type=float, default=None,
                    help="仅保留侧壁：|n·z| < 阈值（如 0.3);None 表示不启用")

    ap.add_argument("--require_through_com", action="store_true", default=True)
    ap.add_argument("--no_require_through_com", dest="require_through_com", action="store_false")
    ap.add_argument("--through_tol_ratio", type=float, default=0.1)

    # 射线出口策略
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--nearest_exit", action="store_true", help="总是最近出口")
    g.add_argument("--farthest_exit", action="store_true", help="总是最远命中（带厚度阈值）")

    # 力闭合阈值
    ap.add_argument("--eps_thresh", type=float, default=None, help="力闭合 ε 硬阈值，例如 0.01")
    ap.add_argument("--wscore_thresh", type=float, default=None, help="或用 wscore 阈值，例如 100")

    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


def main():
    args = parse_args()

    # 加载网格
    mesh_path = os.path.abspath(args.mesh)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    mesh = trimesh.load(mesh_path, force='mesh')

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise RuntimeError("网格没有三角形（Empty triangle list）。请检查输入文件。")

    # 统一尺度（米）
    if args.scale is not None and args.scale != 1.0:
        mesh.apply_scale(float(args.scale))

    print(f"Loaded mesh: {mesh_path}")
    print(f"  vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, watertight={mesh.is_watertight}")
    print(f"  extents = {mesh.extents}, bounds(z) = [{mesh.bounds[0,2]:.4f}, {mesh.bounds[1,2]:.4f}]")

    # 选择出口策略（None=按 watertight 自动）
    if args.nearest_exit and args.farthest_exit:
        raise ValueError("不能同时指定 --nearest_exit 和 --farthest_exit")
    nearest_exit = None
    if args.nearest_exit:
        nearest_exit = True
    elif args.farthest_exit:
        nearest_exit = False

    gen = GraspPairGenerator(
        mu=args.mu,
        num_samples=args.num_samples,
        wmin=args.wmin, wmax=args.wmax,
        opp_angle_deg=args.opp_angle_deg,
        cone_half_deg=args.cone_half_deg,
        rays_per_point=args.rays_per_point,
        m_dirs=args.m_dirs,
        bottom_ratio=args.bottom_ratio,
        top_ratio=args.top_ratio,
        side_only_norm_thresh=args.side_only_norm_thresh,
        require_through_com=args.require_through_com,
        through_tol_ratio=args.through_tol_ratio,
        nearest_exit=nearest_exit,
        eps_thresh=args.eps_thresh,
        wscore_thresh=args.wscore_thresh,
        max_keep=args.max_keep,
        topk=args.topk,
        seed=args.seed
    )

    results, rej = gen.generate(mesh)

    # 输出
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pairs_path = os.path.join(out_dir, f"{args.out_name}.npy")
    n_pairs = gen.save_pairs_only(results, pairs_path)
    print(f"\n✅ 已保存 {n_pairs} 对 (p1,p2,n1,n2) → {pairs_path}")

    if args.save_full:
        full_path = os.path.join(out_dir, f"{args.out_name}_scored.npy")
        n_full = gen.save_scored(results, full_path)
        print(f"📝 已保存 {n_full} 条带分数候选 → {full_path}")


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