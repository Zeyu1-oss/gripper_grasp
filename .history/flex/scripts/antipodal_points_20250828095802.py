import numpy as np
import trimesh
from tqdm import trange
import os
def is_antipodal(p1, n1, p2, n2, mu):
    
    u = p2 - p1
    d = np.linalg.norm(u)
    if d < 1e-6:
        return False
    u /= d

    # 摩擦锥
    cos_theta = 1.0 / np.sqrt(1.0 + mu*mu)  # = cos(arctan(mu))

    c1 = np.dot(u, n1) <= -cos_theta   # u 与 -n1 足够接近
    c2 = np.dot(u, n2) >=  cos_theta   # u 与 +n2 足够接近（等价于 -u 与 -n2）

    return bool(c1 and c2)

def sample_antipodal_pairs(mesh,
                           mu=0.9,
                           approach_dir=np.array([1.0, 0.0, 0.0]),
                           angle_thresh=np.pi/8,
                           num_pairs=10000,
                           num_surface_samples=20000,
                           wmin=0.004, wmax=0.018):
    # 归一化 approach_dir
    approach_dir = np.asarray(approach_dir, dtype=float)
    na = np.linalg.norm(approach_dir)
    approach_dir = approach_dir / (na + 1e-12)

    # 采样点与面法线（面积加权）
    points, fids = trimesh.sample.sample_surface(mesh, num_surface_samples)
    normals = mesh.face_normals[fids]  # 已归一化

    # 只取“侧面”：法线与 approach_dir 夹角 < angle_thresh 与 > pi-angle_thresh
    dot = normals @ approach_dir
    cos_th = np.cos(angle_thresh)
    idx1 = np.where(dot >  cos_th)[0]   # 与 approach_dir 同向
    idx2 = np.where(dot < -cos_th)[0]   # 与 approach_dir 反向

    if len(idx1) == 0 or len(idx2) == 0:
        print("未找到合适的对置候选（可能 approach_dir 不合适或阈值太严）")
        return []

    pairs = []
    rng = np.random.default_rng()

    # 采样直到凑满或尝试次数用尽
    max_trials = num_pairs * 4
    for _ in trange(max_trials):
        i = rng.choice(idx1)
        j = rng.choice(idx2)

        p1, n1 = points[i], normals[i]
        p2, n2 = points[j], normals[j]

        # 宽度（指间距）约束
        w = np.linalg.norm(p2 - p1)
        if w < wmin or w > wmax:
            continue

        # 摩擦锥 + 对向
        if is_antipodal(p1, n1, p2, n2, mu):
            pairs.append((p1, p2, n1, n2))
            if len(pairs) >= num_pairs:
                break

    return pairs


if __name__ == "__main__":
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "antipodal_pairs.npy")

    mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lego.obj"))
    mesh = trimesh.load(mesh_path)
    print(f"Loaded mesh from {mesh_path}")

    pairs = sample_antipodal_pairs(mesh, mu=0.5, num_pairs=10000, num_surface_samples=20000)
    print(f"实际采样到 {len(pairs)} 对有效 antipodal 点")

    if len(pairs) > 0:
        np.save(save_path, pairs)
        print(f"已保存 {len(pairs)} 对 antipodal 点到 {save_path}")
    else:
        print("未找到足够的 antipodal 点对")