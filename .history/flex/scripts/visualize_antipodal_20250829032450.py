import numpy as np
import trimesh
import matplotlib.pyplot as plt

# === 路径配置 ===
npy_path = "/home/rose/legograsp/flex/results/antipodal_pairs/antipodal_pairs_side.npy"
mesh_path = "../lego.obj"   

# === 读取 antipodal 点对 ===
pairs = np.load(npy_path, allow_pickle=True)
print(f"Loaded {len(pairs)} antipodal pairs from: {npy_path}")
print("每个元素格式: (p1, p2, n1, n2)")

if len(pairs) == 0:
    print(" 未找到任何 antipodal 点对！")
    exit()

# === 读取 mesh ===
mesh = trimesh.load(mesh_path, force='mesh')
if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump().sum()

# === 可视化 ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                triangles=mesh.faces, color=(0.7, 0.7, 0.9, 0.4),
                edgecolor='gray', linewidth=0.2)

# 绘制前几个点对
num_to_show = min(5, len(pairs))
for i in range(num_to_show):
    p1, p2, n1, n2 = pairs[i]
    ax.scatter(*p1, color='red', s=60, label='p1' if i == 0 else "")
    ax.scatter(*p2, color='green', s=60, label='p2' if i == 0 else "")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linestyle='--')
    # 可选：画法向
    ax.quiver(p1[0], p1[1], p1[2], n1[0], n1[1], n1[2], length=0.02, color='red', alpha=0.6)
    ax.quiver(p2[0], p2[1], p2[2], n2[0], n2[1], n2[2], length=0.02, color='green', alpha=0.6)

# 标签与图例
ax.set_title("Antipodal Contact Pairs on LEGO Mesh")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()
