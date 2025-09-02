import numpy as np
import trimesh
import os
from tqdm import trange, tqdm
from scipy.optimize import minimize, nnls
from dataclasses import dataclass

# ------------------ Util functions (同你原来的) ------------------
# ... 这里省略：_unit, _orthonormal_tangent_basis, etc ...
# （可直接复制你已有的工具函数）

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

# ------------------ Main Class ------------------

class GraspPlannerRayBased:
    def __init__(self, mesh_path, scale=1.0, verbose=True):
        self.mesh = trimesh.load(mesh_path, force='mesh')
        self.mesh.apply_scale(scale)
        self.verbose = verbose
        if verbose:
            print(f"✅ Mesh loaded: {mesh_path}")
            print(f"   Vertices: {len(self.mesh.vertices)}, Faces: {len(self.mesh.faces)}")
        self.ray_engine = self._build_ray_engine()

    def _build_ray_engine(self):
        return (trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)
                if trimesh.ray.has_embree else
                trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh))

    def generate(self, *,
        mu=0.3,
        num_samples=20000,
        wmin=None,
        wmax=None,
        opp_angle_deg=30.0,
        cone_half_deg=12.0,
        rays_per_point=10,
        require_through_com=True,
        use_nearest_exit_if_closed=True,
        m_dirs=8,
        max_keep=8000,
        topk=1000,
        bottom_ratio=0.05,
        top_ratio=0.05,
        seed=0
    ):
        rng = np.random.default_rng(seed)
        mesh = self.mesh
        com = mesh.center_mass
        ex, ey, ez = mesh.extents
        scale = float(np.linalg.norm(mesh.extents))
        if wmin is None: wmin = 0.2 * float(min(ex, ey))
        if wmax is None: wmax = 1.2 * float(max(ex, ey))
        eps = 1e-5 * scale
        cos_opp = np.cos(np.deg2rad(opp_angle_deg))
        theta = np.deg2rad(cone_half_deg)

        zmin, zmax = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
        height = zmax - zmin
        side_zmin = zmin + bottom_ratio * height
        side_zmax = zmax - top_ratio * height

        points, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
        normals = mesh.face_normals[face_idx]
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12, None)

        candidates = []
        USE_EXIT_NEAREST = mesh.is_watertight and use_nearest_exit_if_closed

        for i in trange(num_samples, desc="📌 Sampling"):
            p1 = points[i]
            if not (side_zmin < float(p1[2]) < side_zmax): continue
            n1 = normals[i]
            axis = -n1
            t1, t2 = _orthonormal_tangent_basis(axis)
            phis = rng.uniform(0.0, 2.0*np.pi, size=rays_per_point)
            dirs = [_unit(axis*np.cos(theta) + (np.cos(phi)*t1 + np.sin(phi)*t2)*np.sin(theta)) for phi in phis]
            origins = np.repeat((p1 - eps*n1)[None, :], len(dirs), axis=0)

            locs, idx_ray, tri_ids = self.ray_engine.intersects_location(
                origins, np.array(dirs), multiple_hits=True
            )
            if len(idx_ray) == 0: continue

            by_ray = {}
            for hit_idx, rid in enumerate(idx_ray):
                by_ray.setdefault(rid, []).append(hit_idx)

            for rid, hit_list in by_ray.items():
                hit_pts  = locs[hit_list]
                hit_tris = [tri_ids[h] for h in hit_list]
                dists = np.linalg.norm(hit_pts - p1, axis=1)

                if USE_EXIT_NEAREST:
                    k_local = int(np.argmin(dists))
                else:
                    t_min = max(5.0*eps, 0.25*wmin)
                    mask = dists >= t_min
                    if not np.any(mask): continue
                    idxs = np.where(mask)[0]
                    k_local = idxs[np.argmax(dists[idxs])]

                p2 = hit_pts[k_local]
                if not (side_zmin < float(p2[2]) < side_zmax): continue
                tri_k = hit_tris[k_local]
                n2 = _unit(mesh.face_normals[tri_k])
                if np.dot(n1, n2) > -cos_opp: continue

                width = float(np.linalg.norm(p2 - p1))
                if width < wmin or width > wmax: continue
                if not is_antipodal(p1, n1, p2, n2, mu): continue
                if require_through_com:
                    u = (p2 - p1) / (width + 1e-12)
                    v = com - p1
                    dist_line = np.linalg.norm(np.cross(u, v))
                    if dist_line > 0.1 * max(ex, ey, ez): continue
                if mesh.is_watertight:
                    mid = 0.5 * (p1 + p2)
                    if not mesh.contains([mid])[0]: continue

                gscore = geom_score_pair(p1, p2, n1, n2, mu, wmin, wmax)
                if gscore <= 0: continue

                candidates.append((p1, p2, n1, n2, width, gscore))
                if len(candidates) >= max_keep:
                    break
            if len(candidates) >= max_keep:
                break

        if not candidates:
            return []

        results = []
        for (p1, p2, n1, n2, width, gscore) in tqdm(candidates, desc="⚙️ Force closure scoring"):
            eps_fc, wscore = wrench_quality_for_pair(p1, p2, n1, n2, mesh, mu=mu, m_dirs=m_dirs)
            total = 0.6*gscore + 0.4*np.log1p(wscore)
            results.append(PairScored(p1, p2, n1, n2, width, gscore, eps_fc, wscore, total))

        results.sort(key=lambda r: -r.total)
        return results[:topk]

    def save_pairs(self, results, save_path):
        pairs = [(r.p1, r.p2, r.n1, r.n2) for r in results]
        np.save(save_path, pairs)
        if self.verbose:
            print(f"✅ Saved {len(pairs)} pairs to {save_path}")
