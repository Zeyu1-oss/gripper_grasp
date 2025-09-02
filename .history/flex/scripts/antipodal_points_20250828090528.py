import trimesh
import numpy as np
from tqdm import trange

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

def sample_antipodal_pairs(mesh, mu=0.5, approach_dir=np.array([1,0,0]), angle_thresh=np.pi/8, num_pairs=10000):
    points, face_idx = trimesh.sample.sample_surface(mesh, 5000)
    normals = mesh.face_normals[face_idx]
    dot = normals @ approach_dir
    idx1 = np.where(dot > np.cos(angle_thresh))[0]
    idx2 = np.where(dot < -np.cos(angle_thresh))[0]
    if len(idx1) == 0 or len(idx2) == 0:
        print("未找到合适的对置点")
        return None
    pairs = []
    rng = np.random.default_rng()
    for _ in trange(num_pairs):
        i = rng.choice(idx1)
        j = rng.choice(idx2)
        if is_antipodal(points[i], normals[i], points[j], normals[j], mu):
            pairs.append((points[i], points[j], normals[i], normals[j]))
        # 若采样效率低，可适当放宽条件或增加采样点数
        if len(pairs) >= num_pairs:
            break
    return pairs

# 示例用法
mesh = trimesh.load('../lego.obj')
pairs = sample_antipodal_pairs(mesh, mu=0.5, num_pairs=10000)
if pairs:
    # 保存为npy
    np.save('antipodal_pairs.npy', pairs)
    print(f"已保存{len(pairs)}对antipodal点到 antipodal_pairs.npy")
else:
    print("未找到足够的antipodal点对")