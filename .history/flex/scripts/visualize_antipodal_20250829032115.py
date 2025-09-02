import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 路径配置 ===
npy_path = "/home/rose/legograsp/flex/results/antipodal_pairs/antipodal_pairs_side.npy"
mesh_path = "/home/rose/legograsp/meshes/lego.obj"  

# === 读取 antipodal 点对 ===
# 数据格式：[N, 2, 3]，表示 N 对接触点，每对两个三维坐标
pairs = np.load(npy_path)
print(f"Loaded {len(pairs)} antipodal pairs from: {npy_path}")

if len(pairs) == 0:
    print("❌ 未找到任何 antipodal 点对，检查输入数据！")
    exit()

# 取前两对点进行可视化（或更多）
num_to_show = min(2, len(pairs))

# === 读取 mesh ===
mesh = trimesh.load(mesh_path)
if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump().sum()

# === 可视化 ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制 mesh 表面
ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                triangles=mesh.faces, color=(0.7, 0.7, 0.9, 0.4), edgecolor='gray', linewidth=0.2)

# 绘制接触点对
for i in range(num_to_show):
    p1, p2 = pairs[i]
    ax.scatter(*p1, color='red', s=60, label='Contact Point 1' if i == 0 else "")
    ax.scatter(*p2, color='green', s=60, label='Contact Point 2' if i == 0 else "")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linestyle='--')

# 轴标签与图例
ax.set_title("Antipodal Contact Points on LEGO Mesh")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()
