import numpy as np

def pair_to_grasp_pose(p1, p2, n1, n2, palm_offset=0.04):
    """
    p1, p2: 两接触点 (3,)
    n1, n2: 两法向 (3,)
    palm_offset: 手掌到指尖的距离（单位：米）
    返回: hand_pose = {'center': (3,), 'rotation': (3,3)}
    """
    center = (p1 + p2) / 2
    x_axis = (p2 - p1)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = (n1 + n2) / 2
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    # 正交化（防止数值误差）
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # 列向量
    palm_center = center - z_axis * palm_offset
    return {'center': palm_center, 'rotation': R}

def pairs_to_grasp_poses(pairs, palm_offset=0.04):
    poses = []
    for p1, p2, n1, n2 in pairs:
        pose = pair_to_grasp_pose(np.array(p1), np.array(p2), np.array(n1), np.array(n2), palm_offset)
        poses.append(pose)
    return poses

# 示例用
if __name__ == "__main__":
    import os
    pairs = np.load("../results/antipodal_pairs.npy", allow_pickle=True)
    poses = pairs_to_grasp_poses(pairs, palm_offset=0.04)
    np.save("../results/grasp_poses.npy", poses)
    print(f"已保存 {len(poses)} 个夹爪6D位姿到 ../results/6d/grasp_poses.npy")