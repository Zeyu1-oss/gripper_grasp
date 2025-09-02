#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import trimesh
from tqdm import trange
from scipy.spatial import cKDTree
import os

def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v/n

def _outward_fix_normals(points, normals):
    """修正法向朝外"""
    C = points.mean(0)
    vout = points - C
    vout = vout / (np.linalg.norm(vout, axis=1, keepdims=True) + 1e-12)
    flip = (np.sum(normals * vout, axis=1) < 0.0)
    normals = normals.copy()
    normals[flip] *= -1.0
    return normals

def _pca_vertex_normals(V, k=24):
    """PCA 近邻点估计法向"""
    kdt = cKDTree(V)
    N = np.zeros_like(V)
    C0 = V.mean(0)
    for i,p in enumerate(V):
        _, idx = kdt.query(p, k=min(k, len(V)))
        Q = V[idx] - p
        C = Q.T @ Q
        w, U = np.linalg.eigh(C)
        n = U[:,0]
        if np.dot(n, V[i]-C0) < 0: 
            n = -n
        N[i] = _unit(n)
    return N

def is_antipodal(p1, n1, p2, n2, mu):
    """检查两点是否满足摩擦锥下的antipodal条件"""
    u = p2 - p1
    d = np.linalg.norm(u)
    if d < 1e-6:
        return False
    u /= d
    
    theta = np.arctan(mu)  # 摩擦锥角度
    cos_th = np.cos(theta)

    # 条件：夹角在摩擦锥内
    cond1 = np.dot(-u, n1) >= cos_th
    cond2 = np.dot(u, n2)  >= cos_th

    return cond1 and cond2


def sample_antipodal_pairs(mesh,
                           mu=0.5,
                           num_pairs=5000,
                           num_surface_samples=50000,
                           wmin=None, wmax=None,
                           use_pca_normals=True):
    """采样物体的二指antipodal抓取点对"""
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

    pairs = []
    rng = np.random.default_rng()
    max_trials = num_pairs * 20

    for _ in trange(max_trials, desc="采样antipodal点对"):
        i, j = rng.choice(len(points), size=2, replace=False)
        p1, n1 = points[i], FN[i]
        p2, n2 = points[j], FN[j]

        # 过滤非法
        if (np.linalg.norm(n1) < 1e-8 or np.linalg.norm(n2) < 1e-8 or
            not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2))):
            continue

        w = np.linalg.norm(p2 - p1)
        if w < wmin or w > wmax:
            continue

        if is_antipodal(p1, n1, p2, n2, mu):
            pairs.append((p1, p2, n1, n2))
            if len(pairs) >= num_pairs:
                break

    return pairs

if __name__ == "__main__":
    # 输出目录
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/antipodal_pairs"))
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "antipodal_pairs.npy")

    # 输入 mesh
    mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lego.obj"))
    mesh = trimesh.load(mesh_path)
    print(f"Loaded mesh from {mesh_path}")

    # 采样
    pairs = sample_antipodal_pairs(mesh, mu=0.5, num_pairs=2000, num_surface_samples=40000)
    print(f"实际采样到 {len(pairs)} 对 antipodal 点")

    # 保存
    if len(pairs) > 0:
        np.save(save_path, pairs)
        print(f"已保存到 {save_path}")

        # 可视化前几对
        scene = mesh.scene()
        for i, (p1, p2, n1, n2) in enumerate(pairs[:10]):
            scene.add_geometry(trimesh.points.PointCloud([p1], colors=[255,0,0]))
            scene.add_geometry(trimesh.points.PointCloud([p2], colors=[0,0,255]))
        scene.show()
    else:
        print("未找到足够的 antipodal 点对")
