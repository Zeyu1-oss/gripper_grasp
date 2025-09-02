import trimesh
import numpy as np
import sys

def check_mesh(mesh_path):
    mesh = trimesh.load(mesh_path)
    print(f"Mesh: {mesh_path}")
    print(f"顶点数: {len(mesh.vertices)}")
    print(f"面数: {len(mesh.faces)}")
    # 检查法向
    face_normals = mesh.face_normals
    zero_normals = np.sum(np.linalg.norm(face_normals, axis=1) < 1e-8)
    print(f"面法向全零数: {zero_normals} / {len(face_normals)}")
    # 检查非流形
    print(f"是否流形: {mesh.is_watertight}")
    # 检查nan/inf
    has_nan = np.any(~np.isfinite(mesh.vertices)) or np.any(~np.isfinite(mesh.faces))
    print(f"包含NaN/Inf: {has_nan}")
    # 检查重复顶点
    unique_verts = np.unique(mesh.vertices, axis=0)
    print(f"重复顶点数: {len(mesh.vertices) - len(unique_verts)}")
    # 检查孤立点
    print(f"孤立点数: {len(mesh.vertices) - np.unique(mesh.faces).size}")
    # 检查边界边和边界环
    if hasattr(mesh, "boundary_edges"):
        print(f"边界边数: {mesh.boundary_edges.shape[0]}")
        try:
            rings = trimesh.graph.connected_edges(mesh.boundary_edges)
            print(f"边界环数: {len(rings)}")
        except Exception as e:
            print(f"边界环统计失败: {e}")
    else:
        print("无边界边信息")
    # 可视化（可选）
    # mesh.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python mesh_checker.py ../lego.obj")
        sys.exit(1)
    check_mesh(sys.argv[1])