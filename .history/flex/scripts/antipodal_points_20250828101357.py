import numpy as np
import trimesh
from tqdm import trange
from scipy.spatial import cKDTree
import os

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
    u = p2 - p1
    d = np.linalg.norm(u)
    if d < 1e-6: 
        return False
    u /= d
    cos_th = 1.0/np.sqrt(1.0 + mu*mu)
    c1 = np.dot(u, n1) <= -cos_th
    c2 = np.dot(u, n2) >=  cos_th
    return bool(c1 and c2)

def sample_antipodal_pairs_side(mesh,
                                mu=0.5,
                                approach_dir=np.array([1.0,0.0,0.0]),
                                angle_thresh=np.deg2rad(80),   # 只采侧面，夹角接近90°
                                num_pairs=10000,
                                num_surface_samples=60000,
                                wmin=None, wmax=None,
                                use_pca_normals=True):
    points, fids = trimesh.sample.sample_surface(mesh, num_surface_samples)
    if use_pca_normals:
        V = mesh.vertices.view(np.ndarray)
        VN = _pca_vertex_normals(V, k=24)
        kdt = cKDTree(V)
        _, idx = kdt.query(points)
        FN = VN[idx]
        FN = _outward_fix_normals(points, FN)
    else:
        FN = mesh.face_normals[fids]
        FN = _outward_fix_normals(points, FN)

    # 指间距范围
    ex, ey, ez = mesh.extents
    if wmin is None: wmin = 0.2 * min(ex, ey)
    if wmax is None: wmax = 0.95 * max(ex, ey)

    # 只保留侧面点
    ad = _unit(np.asarray(approach_dir, float))
    dot = FN @ ad
    cos_th = np.cos(angle_thresh)  # angle_thresh接近90°，cos很小
    side_idx = np.where(np.abs(dot) < cos_th)[0]
    if len(side_idx) < 2:
        print("未找到足够的侧面点")
        return []

    pairs = []
    rng = np.random.default_rng()
    max_trials = num_pairs * 8

    for _ in trange(max_trials, desc="采样侧面antipodal点对"):
        i, j = rng.choice(side_idx, size=2, replace=False)
        p1, n1 = points[i], FN[i]
        p2, n2 = points[j], FN[j]
        # 过滤非法法向
        if (np.linalg.norm(n1) < 1e-8 or np.linalg.norm(n2) < 1e-8 or
            not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)) or
            not np.all(np.isfinite(n1)) or not np.all(np.isfinite(n2))):
            continue
        w = np.linalg.norm(p2 - p1)
        if w < wmin or w > wmax:
            continue
        if is_antipodal(p1, n1, p2, n2, mu):
            pairs.append((p1, p2, n1, n2))
            if len(pairs) >= num_pairs:
                break

    # 兜底策略：如采不到，尝试OBB六面
    if len(pairs) == 0:
        T = mesh.bounding_box_oriented.primitive.transform
        R = T[:3,:3]; t = T[:3,3]
        ext = mesh.bounding_box_oriented.primitive.extents
        grid = 10
        axes = [(0, ext[0]), (1, ext[1])]
        for ax, L in axes:
            if len(pairs) >= num_pairs: break
            for u in np.linspace(-0.5, 0.5, grid):
                for v in np.linspace(-0.5, 0.5, grid):
                    loc = np.zeros(3)
                    loc[ax] = +0.5*ext[ax]
                    oth = [0,1,2]
                    oth.remove(ax)
                    loc[oth[0]] = u * ext[oth[0]]
                    loc[oth[1]] = v * ext[oth[1]]
                    pR = R@loc + t
                    nR = R[:,ax]
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

if __name__ == "__main__":
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/antipodal_pairs"))
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "antipodal_pairs_side.npy")

    mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lego.obj"))
    mesh = trimesh.load(mesh_path)
    print(f"Loaded mesh from {mesh_path}")

    pairs = sample_antipodal_pairs_side(mesh, mu=0.5, num_pairs=10000, num_surface_samples=60000)
    print(f"实际采样到 {len(pairs)} 对侧面antipodal点")
    if len(pairs) > 0:
        np.save(save_path, pairs)
        print(f"已保存 {len(pairs)} 对侧面antipodal点到 {save_path}")
    else:
        print("未找到足够的侧面antipodal点对")