import trimesh
import numpy as np

def is_antipodal(p1, n1, p2, n2, mu):
    v = p2 - p1
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-6:
        return False
    v_unit = v / v_norm
    theta = np.arctan(mu)
    cos_theta = np.cos(theta)
    # p1: v方向与n1夹角
    cond1 = np.dot(v_unit, n1) >= cos_theta
    # p2: -v方向与n2夹角
    cond2 = np.dot(-v_unit, n2) >= cos_theta
    return cond1 and cond2

def find_antipodal_grasp_points(mesh, mu=0.5, approach_dir=np.array([1,0,0]), angle_thresh=np.pi/8):
    points, face_idx = trimesh.sample.sample_surface(mesh, 500)
    normals = mesh.face_normals[face_idx]
    dot = normals @ approach_dir
    idx1 = np.where(dot > np.cos(angle_thresh))[0]
    idx2 = np.where(dot < -np.cos(angle_thresh))[0]
    if len(idx1) == 0 or len(idx2) == 0:
        print("未找到合适的对置点")
        return None
    for i in idx1:
        for j in idx2:
            if is_antipodal(points[i], normals[i], points[j], normals[j], mu):
                return points[i], points[j], normals[i], normals[j]
    print("未找到满足摩擦锥条件的antipodal点对")
    return None

# 示例用法
mesh = trimesh.load('../lego.obj')
result = find_antipodal_grasp_points(mesh, mu=0.5)
if result:
    p1, p2, n1, n2 = result
    print("Antipodal points:", p1, p2)
    print("Normals:", n1, n2)