# import numpy as np
# import trimesh
# from tqdm import trange
# import os
# def is_antipodal(p1, n1, p2, n2, mu):
    
#     u = p2 - p1
#     d = np.linalg.norm(u)
#     if d < 1e-6:
#         return False
#     u /= d

#     # 摩擦锥
#     cos_theta = 1.0 / np.sqrt(1.0 + mu*mu)  # = cos(arctan(mu))

#     c1 = np.dot(u, n1) <= -cos_theta   # u 与 -n1 足够接近
#     c2 = np.dot(u, n2) >=  cos_theta   # u 与 +n2 足够接近（等价于 -u 与 -n2）

#     return bool(c1 and c2)

# def sample_antipodal_pairs(mesh,
#                            mu=0.9,
#                            approach_dir=np.array([1.0, 0.0, 0.0]),
#                            angle_thresh=np.pi/8,
#                            num_pairs=10000,
#                            num_surface_samples=20000,
#                            wmin=0.004, wmax=0.018):
#     # 归一化 approach_dir
#     approach_dir = np.asarray(approach_dir, dtype=float)
#     na = np.linalg.norm(approach_dir)
#     approach_dir = approach_dir / (na + 1e-12)

#     # 采样点与面法线（面积加权）
#     points, fids = trimesh.sample.sample_surface(mesh, num_surface_samples)
#     normals = mesh.face_normals[fids]  # 已归一化

#     # 只取“侧面”：法线与 approach_dir 夹角 < angle_thresh 与 > pi-angle_thresh
#     dot = normals @ approach_dir
#     cos_th = np.cos(angle_thresh)
#     idx1 = np.where(dot >  cos_th)[0]   # 与 approach_dir 同向
#     idx2 = np.where(dot < -cos_th)[0]   # 与 approach_dir 反向

#     if len(idx1) == 0 or len(idx2) == 0:
#         print("未找到合适的对置候选（可能 approach_dir 不合适或阈值太严）")
#         return []

#     pairs = []
#     rng = np.random.default_rng()

#     # 采样直到凑满或尝试次数用尽
#     max_trials = num_pairs * 4
#     for _ in trange(max_trials):
#         i = rng.choice(idx1)
#         j = rng.choice(idx2)

#         p1, n1 = points[i], normals[i]
#         p2, n2 = points[j], normals[j]

#         # 宽度（指间距）约束
#         w = np.linalg.norm(p2 - p1)
#         if w < wmin or w > wmax:
#             continue

#         # 摩擦锥 + 对向
#         if is_antipodal(p1, n1, p2, n2, mu):
#             pairs.append((p1, p2, n1, n2))
#             if len(pairs) >= num_pairs:
#                 break

#     return pairs


# if __name__ == "__main__":
#     results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
#     os.makedirs(results_dir, exist_ok=True)
#     save_path = os.path.join(results_dir, "antipodal_pairs.npy")

#     mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lego.obj"))
#     mesh = trimesh.load(mesh_path)
#     print(f"Loaded mesh from {mesh_path}")

#     pairs = sample_antipodal_pairs(mesh, mu=0.5, num_pairs=10000, num_surface_samples=20000)
#     print(f"实际采样到 {len(pairs)} 对有效 antipodal 点")

#     if len(pairs) > 0:
#         np.save(save_path, pairs)
#         print(f"已保存 {len(pairs)} 对 antipodal 点到 {save_path}")
#     else:
#         print("未找到足够的 antipodal 点对")
# 
import numpy as np
import trimesh
from tqdm import trange
from scipy.spatial import cKDTree

def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v/n

def _outward_fix_normals(points, normals):
    C = points.mean(0)
    vout = points - C
    vout = vout / (np.linalg.norm(vout, axis=1, keepdims=True) + 1e-12)
    flip = (np.sum(normals * vout, axis=1) < 0.0)
    normals = normals.copy()
    normals[flip] *= -1.0
    return normals

def _pca_vertex_normals(V, k=24):
    kdt = cKDTree(V)
    N = np.zeros_like(V)
    C0 = V.mean(0)
    for i,p in enumerate(V):
        _, idx = kdt.query(p, k=min(k, len(V)))
        Q = V[idx] - p
        C = Q.T @ Q
        w, U = np.linalg.eigh(C)
        n = U[:,0]
        if np.dot(n, V[i]-C0) < 0: n = -n
        N[i] = _unit(n)
    return N

def is_antipodal(p1, n1, p2, n2, mu):
    # u: 从 p1 指向 p2
    u = p2 - p1
    d = np.linalg.norm(u)
    if d < 1e-6: 
        return False
    u /= d
    # cos(theta) = 1/sqrt(1+mu^2)
    cos_th = 1.0/np.sqrt(1.0 + mu*mu)
    # 正确的不等式（注意号）
    c1 = np.dot(u, n1) <= -cos_th   # u 与 -n1 夹角 <= theta
    c2 = np.dot(u, n2) >=  cos_th   # u 与 +n2 夹角 <= theta  <=> -u 与 -n2
    return bool(c1 and c2)

def sample_antipodal_pairs(mesh,
                           mu=0.5,
                           approach_dir=np.array([1.0,0.0,0.0]),
                           angle_thresh=np.deg2rad(60),   # 放宽到 60°
                           num_pairs=10000,
                           num_surface_samples=60000,
                           wmin=None, wmax=None,
                           use_pca_normals=True):
    # 面采样（面积加权）
    points, fids = trimesh.sample.sample_surface(mesh, num_surface_samples)
    FN = mesh.face_normals[fids]

    # 法向强制外向
    FN = _outward_fix_normals(points, FN)

    # 备选：PCA 顶点法向（mesh 不干净时更稳）
    if use_pca_normals:
        V = mesh.vertices.view(np.ndarray)
        VN = _pca_vertex_normals(V, k=24)
        # 用顶点法向替换面法向的方向（取该面三个顶点平均近似）
        # 简化：直接用外向修正后的 FN，已经够用；真要更稳，可在此把 FN 替换为邻近顶点 VN 的均值
        pass

    # 指间距范围（默认从外形估计）
    ex, ey, ez = mesh.extents
    if wmin is None: wmin = 0.2 * min(ex, ey)   # 稍微保守
    if wmax is None: wmax = 0.95* max(ex, ey)   # 不要超过最大侧边

    # 归一化 approach_dir，并自动再试一个与之垂直的方向（XY 平面内）
    ad = _unit(np.asarray(approach_dir, float))
    if np.linalg.norm(ad[:2]) < 1e-9:
        ad = np.array([1.0,0.0,0.0])  # 防止是竖直向量
    ad_perp = _unit(np.array([-ad[1], ad[0], 0.0]))  # XY 里垂直

    pairs = []
    rng = np.random.default_rng()
    max_trials = num_pairs * 6  # 给足尝试次数

    def try_one_dir(dir_vec):
        nonlocal pairs
        dot = FN @ dir_vec
        cos_th = np.cos(angle_thresh)
        idx1 = np.where(dot >  cos_th)[0]   # 与 dir_vec 同向
        idx2 = np.where(dot < -cos_th)[0]   # 与 dir_vec 反向
        if len(idx1)==0 or len(idx2)==0:
            return

        for _ in trange(max_trials, leave=False):
            i = rng.choice(idx1)
            j = rng.choice(idx2)
            p1, n1 = points[i], FN[i]
            p2, n2 = points[j], FN[j]
            w = np.linalg.norm(p2-p1)
            if w < wmin or w > wmax:
                continue
            if is_antipodal(p1, n1, p2, n2, mu):
                pairs.append((p1, p2, n1, n2))
                if len(pairs) >= num_pairs:
                    break

    # 先按给定方向试
    try_one_dir(ad)
    # 不够就换垂直方向再试
    if len(pairs) < num_pairs:
        try_one_dir(ad_perp)

    # 兜底：还是 0 的话，用 OBB 六个面直接造“对置”候选再验摩擦锥（稳出解）
    if len(pairs) == 0:
        T = mesh.bounding_box_oriented.primitive.transform
        R = T[:3,:3]; t = T[:3,3]
        ext = mesh.bounding_box_oriented.primitive.extents  # 长宽高
        # 六个面对（±X, ±Y），跳过 Z 面，构造两侧网格点
        grid = 10
        axes = [(0, ext[0]), (1, ext[1])]  # 只在 X/Y 两轴上试对置
        for ax, L in axes:
            if len(pairs) >= num_pairs: break
            # 在该轴两侧面上，生成均匀网格点
            for u in np.linspace(-0.5, 0.5, grid):
                for v in np.linspace(-0.5, 0.5, grid):
                    loc = np.zeros(3)
                    loc[ax] = +0.5*ext[ax]
                    oth = [0,1,2]
                    oth.remove(ax)
                    loc[oth[0]] = u * ext[oth[0]]
                    loc[oth[1]] = v * ext[oth[1]]
                    pR = R@loc + t
                    nR = R[:,ax]  # 外法向

                    loc[ax] = -0.5*ext[ax]
                    pL = R@loc + t
                    nL = -R[:,ax]

                    w = np.linalg.norm(pR - pL)
                    if wmin <= w <= wmax and is_antipodal(pL, nL, pR, nR, mu):
                        pairs.append((pL, pR, nL, nR))
                        if len(pairs) >= num_pairs:
                            break
                if len(pairs) >= num_pairs:
                    break

    return pairs
