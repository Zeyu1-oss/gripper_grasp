import trimesh
import numpy as np
from tqdm import trange
import os

def is_antipodal(p1, n1, p2, n2, mu):
    v = p2 - p1
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-6:
        return False
    v_unit = v / v_norm
    theta = np.arctan(mu)
    cos_theta = np.cos(theta)
    cond1 = np.dot(v_unit, n1) >= cos_theta
    cond2 = np.dot(-v_unit, n2) >= cos_theta
    return cond1 and cond2

def is_valid_pair(p1, p2, n1, n2):
    # 检查是否有NaN/inf/全零法向
    if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2)) and
            np.all(np.isfinite(n1)) and np.all(np.isfinite(n2))):
        return False
    if np.linalg.norm(n1) < 1e-8 or np.linalg.norm(n2) < 1e-8:
        return False
    return True

def sample_antipodal_pairs(mesh, mu=0.5, approach_dir=np.array([1,0,0]), angle_thresh=np.pi/8, num_pairs=10000, num_surface_samples=20000):
    points, face_idx = trimesh.sample.sample_surface(mesh, num_surface_samples)
    normals = mesh.face_normals[face_idx]
    dot = normals @ approach_dir
    idx1 = np.where(dot > np.cos(angle_thresh))[0]
    idx2 = np.where(dot < -np.cos(angle_thresh))[0]
    if len(idx1) == 0 or len(idx2) == 0:
        print("未找到合适的对置点")
        return []
    pairs = []
    rng = np.random.default_rng()
    for _ in trange(num_pairs * 3):  # 多采样一些，防止采样效率低
        i = rng.choice(idx1)
        j = rng.choice(idx2)
        p1, p2, n1, n2 = points[i], points[j], normals[i], normals[j]
        if not is_valid_pair(p1, p2, n1, n2):
            continue
        if is_antipodal(p1, n1, p2, n2, mu):
            pairs.append((p1, p2, n1, n2))
        if len(pairs) >= num_pairs:
            break
    return pairs

if __name__ == "__main__":
    # 自动创建results目录
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