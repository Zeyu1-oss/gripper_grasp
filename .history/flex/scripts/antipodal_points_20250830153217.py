import numpy as np
import trimesh
import os
from tqdm import trange
from scipy.optimize import minimize, nnls
from dataclasses import dataclass

# =================== 基础工具 ===================
def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def _orthonormal_tangent_basis(axis):
    axis = _unit(axis)
    up = np.array([0., 0., 1.]) if abs(np.dot(axis, [0,0,1])) < 0.95 else np.array([0., 1., 0.])
    t1 = np.cross(axis, up); t1 = _unit(t1)
    t2 = np.cross(axis, t1); t2 = _unit(t2)
    return t1, t2

def _build_ray_engine(mesh):
    return (trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
            if trimesh.ray.has_embree else
            trimesh.ray.ray_triangle.RayMeshIntersector(mesh))

# =================== 几何判定 ===================
def is_antipodal(p1, n1, p2, n2, mu):
    """摩擦锥下的两点 antipodal 判定（与你原逻辑一致）"""
    u = p2 - p1
    d = np.linalg.norm(u)
    if d < 1e-6:
        return False
    u /= d
    theta = np.arctan(mu)
    cos_th = np.cos(theta)
    cond1 = np.dot(-u, n1) >= cos_th
    cond2 = np.dot(u, n2)  >= cos_th
    return cond1 and cond2

def geom_score_pair(p, q, n_p, n_q, mu, width_min, width_max):
    """几何打分：摩擦锥裕度 + 反法向 + 宽度贴合（越大越好）"""
    v = q - p; d = np.linalg.norm(v)
    if d < 1e-9: return -1e9
    u = v / d
    alpha = np.arctan(mu); cos_a = np.cos(alpha)
    m_p = (-cos_a - np.dot(n_p, u))      # p侧锥裕度
    m_q = ( np.dot(n_q, u) - cos_a)      # q侧锥裕度
    anti = -(np.dot(n_p, n_q))           # 反法向程度
    if d < width_min: w_pen = (width_min - d)**2
    elif d > width_max: w_pen = (d - width_max)**2
    else:
        mid = 0.5*(width_min + width_max)
        w_pen = 0.25*((d - mid)/(0.5*(width_max - width_min)+1e-9))**2
    sp = lambda x: np.log1p(np.exp(x))
    return 1.0*sp(m_p) + 1.0*sp(m_q) + 0.3*anti - 0.3*w_pen

# =================== 力空间（wrench）评分 ===================
def _contact_wrench_rays(p, n, mu, m_dirs, com, torque_scale):
    """
    离散摩擦锥 m_dirs 条射线：f_dir = normalize(n + mu * t)，t 在切平面均匀一圈
    返回 (6, m_dirs) 的扳手矩阵，每列为 [f; tau/torque_scale]
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
    解决：min 0.5 ||W λ||^2  s.t. λ>=0, sum λ = 1
    成功返回 (eps, λ, True)，失败用 NNLS 近似退路
    """
    m = W.shape[1]
    H = W.T @ W
    fun = lambda lam: 0.5 * lam @ H @ lam
    jac = lambda lam: H @ lam
    cons = [{'type': 'eq', 'fun': lambda lam: np.sum(lam) - 1.0,
             'jac':  lambda lam: np.ones_like(lam)}]
    bnds = [(0.0, None)] * m
    x0 = np.ones(m) / m
    res = minimize(fun, x0, method='SLSQP', jac=jac, bounds=bnds, constraints=cons,
                   options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
    if res.success:
        lam = res.x
        return float(np.linalg.norm(W @ lam)), lam, True
    # 退回 NNLS 近似：min || [W; sqrt(rho)*1^T] λ - [0; sqrt(rho)] ||, λ>=0
    rho = 10.0
    A = np.vstack([W, np.sqrt(rho) * np.ones((1, m))])
    b = np.zeros(A.shape[0]); b[-1] = np.sqrt(rho)
    lam, _ = nnls(A, b)
    s = np.sum(lam)
    if s > 1e-12: lam = lam / s
    return float(np.linalg.norm(W @ lam)), lam, False

def wrench_quality_for_pair(p, q, n_p, n_q, mesh, mu=0.6, m_dirs=8):
    """
    返回：(eps, quality, total_score)
    - eps 越小越好（→力闭合）
    - quality = 1/(eps+1e-6)
    - total_score：与几何分结合在外层做
    """
    # 量纲归一
    com = mesh.center_mass if mesh.is_watertight else 0.5*(mesh.bounds[0] + mesh.bounds[1])
    Lc  = float(np.linalg.norm(mesh.extents))
    # 两接触的摩擦锥离散
    W1 = _contact_wrench_rays(p, n_p, mu, m_dirs, com, Lc)
    W2 = _contact_wrench_rays(q, n_q, mu, m_dirs, com, Lc)
    W  = np.concatenate([W1, W2], axis=1)  # (6, 2*m_dirs)
    eps, lam, ok = _epsilon_qp(W)
    quality = 1.0 / (eps + 1e-6)
    return eps, quality

# =================== 主流程：一步到位 ===================
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
    num_samples=30000,          # 多采样，多产点
    wmin=None, wmax=None,        # 夹爪开度
    opp_angle_deg=30.0,          # 反法向粗过滤
    cone_half_deg=12.0,          # -n1 周围圆锥半角
    rays_per_point=10,           # 每点多射线
    require_through_com=True,    # 质心附近穿越
    use_nearest_exit_if_closed=True,  # watertight 时用最近“出口”
    m_dirs=8,                    # 摩擦锥离散
    max_keep=8000,               # 最多保留这么多候选（几何过筛后）
    topk=4000,                   # 最终按总分取前 K（保存的数量）
    seed=0
):
    rng = np.random.default_rng(seed)
    ex, ey, ez = mesh.extents
    if wmin is None: wmin = 0.2 * float(min(ex, ey))
    if wmax is None: wmax = 1.2 * float(max(ex, ey))
    scale = float(np.linalg.norm(mesh.extents))
    eps = 1e-5 * scale
    cos_opp = np.cos(np.deg2rad(opp_angle_deg))
    theta = np.deg2rad(cone_half_deg)
    ray_engine = _build_ray_engine(mesh)
    com = mesh.center_mass
    # 采样点与法向
    points, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
    normals = mesh.face_normals[face_idx]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12, None)

    candidates = []
    USE_EXIT_NEAREST = (mesh.is_watertight and use_nearest_exit_if_closed)

    for i in trange(num_samples, desc="生成候选(射线)"):
        p1 = points[i]
        n1 = normals[i]
        axis = -n1
        t1, t2 = _orthonormal_tangent_basis(axis)
        phis = rng.uniform(0.0, 2.0*np.pi, size=rays_per_point)
        dirs = []
        for phi in phis:
            d = axis*np.cos(theta) + (np.cos(phi)*t1 + np.sin(phi)*t2)*np.sin(theta)
            dirs.append(_unit(d))
        origins = np.repeat((p1 - eps*n1)[None, :], len(dirs), axis=0)
        locs, idx_ray, tri_ids = ray_engine.intersects_location(origins, np.array(dirs), multiple_hits=True)
        if len(locs) == 0:
            continue

        # 每条发射的命中里挑一个“对侧”命中：闭合网格取最近出口；否则取最远并跳过太薄的
        # 为简单稳妥，这里把本次发出的所有命中按 ray 分组后，各选一个 best，再逐一验算
        # 先按 idx_ray 分桶
        if len(idx_ray) == 0:
            continue
        # 构造每条 ray 的命中索引列表
        by_ray = {}
        for hit_idx, rid in enumerate(idx_ray):
            by_ray.setdefault(rid, []).append(hit_idx)

        for rid, hit_list in by_ray.items():
            hit_pts = locs[hit_list]
            hit_tris = [tri_ids[h] for h in hit_list]
            dists = np.linalg.norm(hit_pts - p1, axis=1)

            if USE_EXIT_NEAREST:
                k_local = int(np.argmin(dists))  # 最近“出口”
                p2 = hit_pts[k_local]
                tri_k = hit_tris[k_local]
            else:
                t_min = max(5.0*eps, 0.25*wmin)  # 自适应最小厚度阈值
                mask = dists >= t_min
                if not np.any(mask):
                    continue
                idxs = np.where(mask)[0]
                # 取最远（穿越另一层）
                k_local = idxs[np.argmax(dists[idxs])]
                p2 = hit_pts[k_local]
                tri_k = hit_tris[k_local]

            n2 = _unit(mesh.face_normals[tri_k])
            # 早期角度过滤
            if np.dot(n1, n2) > -cos_opp:
                continue

            width = float(np.linalg.norm(p2 - p1))
            if width < wmin or width > wmax:
                continue

            # 摩擦锥严格检查
            if not is_antipodal(p1, n1, p2, n2, mu):
                continue

            # 质心附近穿越（粗判）
            if require_through_com:
                u = (p2 - p1) / (width + 1e-12)
                v = com - p1
                dist_line = np.linalg.norm(np.cross(u, v))
                if dist_line > 0.1 * max(ex, ey, ez):
                    continue

            # （可选）中点在体内（仅闭合网格）
            if mesh.is_watertight:
                mid = 0.5*(p1 + p2)
                if not mesh.contains([mid])[0]:
                    continue

            # 初步几何分
            gscore = geom_score_pair(p1, p2, n1, n2, mu, wmin, wmax)
            if gscore <= 0:
                continue

            candidates.append((p1, p2, n1, n2, width, gscore))
            if len(candidates) >= max_keep:
                break
        if len(candidates) >= max_keep:
            break

    if not candidates:
        return []

    # 力闭合评分 + 综合排序
    results = []
    for (p1, p2, n1, n2, width, gscore) in trange(len(candidates), desc="力闭合打分"):
        eps_fc, wscore = wrench_quality_for_pair(p1, p2, n1, n2, mesh, mu=mu, m_dirs=m_dirs)
        total = 0.6*gscore + 0.4*np.log1p(wscore)  # 几何+力学混合
        results.append(PairScored(p1=p1, p2=p2, n1=n1, n2=n2,
                                  width=width, gscore=gscore, eps=eps_fc,
                                  wscore=wscore, total=total))
    results.sort(key=lambda r: -r.total)
    return results[:topk]

# =================== 脚本入口（只保存点对，不改你的保存类型） ===================
if __name__ == "__main__":
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/antipodal_pairs"))
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "antipodal_pairs_ray.npy")

    mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lego.obj"))
    mesh = trimesh.load(mesh_path, force='mesh')

    # 需要的话可缩放到米制
    scale = 0.01
    mesh.apply_scale(scale)
    print(f"Loaded mesh from {mesh_path}, vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

    pairs_scored = generate_grasp_pairs_with_force_closure(
        mesh,
        mu=0.6,
        num_samples=30000,         # 多采样
        wmin=None, wmax=None,      # 默认基于 extents 自适应
        opp_angle_deg=30.0,
        cone_half_deg=12.0,
        rays_per_point=10,
        require_through_com=True,
        use_nearest_exit_if_closed=True,
        m_dirs=8,
        max_keep=8000,             # 先留较多候选
        topk=4000,                 # 最终导出较多（你要“多生成一点数据”）
        seed=42
    )
    print(f"✅ 力闭合打分后，得到 {len(pairs_scored)} 对候选（已排序）")

    # —— 只保存点对（与你原来一致的保存类型）——
    pairs = [(r.p1, r.p2, r.n1, r.n2) for r in pairs_scored]
    np.save(save_path, pairs)
    print(f"✅ 已保存 {len(pairs)} 对 (p1,p2,n1,n2) 到 {save_path}")

    # 如果你想临时看一下分数，取消下行注释另存一份（不影响原有类型）
    # np.save(save_path.replace(".npy", "_scores.npy"),
    #         np.array([(r.width, r.gscore, r.eps, r.wscore, r.total) for r in pairs_scored], dtype=float))
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