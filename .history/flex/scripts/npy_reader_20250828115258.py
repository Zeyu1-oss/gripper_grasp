import numpy as np

def read_grasps(npy_path, num=5):
    # 加载 .npy 文件
    data = np.load(npy_path)

    # 检查维度
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"Expected shape (N, 6), but got {data.shape}")

    print(f"Total grasps in file: {data.shape[0]}")
    print(f"Showing first {num} grasps:")

    for i in range(min(num, data.shape[0])):
        pos = data[i, :3]
        rot = data[i, 3:]
        print(f"[{i}] Pos: {pos}, Rot: {rot}")

    return data[:num]

# 示例调用
if __name__ == "__main__":
    npy_path = "your_grasp_data.npy"  # 替换为你的路径
    read_grasps(npy_path, num=5)
