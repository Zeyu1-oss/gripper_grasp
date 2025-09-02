import numpy as np
import os

def pair_to_grasp_pose(p1, p2, n1, n2, palm_offset=0.04):
    center = (p1 + p2) / 2
    x_axis = (p2 - p1)
    if np.linalg.norm(x_axis) < 1e-8:
        raise ValueError("接触点重合，无法定义抓取方向")
    x_axis /= np.linalg.norm(x_axis)
    z_axis = (n1 + n2) / 2
    if np.linalg.norm(z_axis) < 1e-8:
        raise ValueError("法向全零，无法定义手掌法向")
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        raise ValueError("x轴与z轴平行,无法定义y轴")
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    palm_center = center - z_axis * palm_offset
    return {'center': palm_center, 'rotation': R}

def pairs_to_grasp_poses(pairs, palm_offset=0):
    poses = []
    for idx, (p1, p2, n1, n2) in enumerate(pairs):
        try:
            pose = pair_to_grasp_pose(np.array(p1), np.array(p2), np.array(n1), np.array(n2), palm_offset)
            poses.append(pose)
        except Exception as e:
            print(f"第{idx}对转换失败: {e}")
    return poses

if __name__ == "__main__":
    pairs = np.load("../results/antipodal_pairs/antipodal_pairs_ray.npy", allow_pickle=True)
    os.makedirs("../results/6d", exist_ok=True)
    poses = pairs_to_grasp_poses(pairs, palm_offset=0.03)
    np.save("../results/6d/grasp_poses.npy", poses)
    print(f"已保存 {len(poses)} 个夹爪6D位姿到 ../results/6d/grasp_poses.npy")