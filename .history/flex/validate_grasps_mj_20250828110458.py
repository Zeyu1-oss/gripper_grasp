import numpy as np
import os

def pair_to_grasp_pose(p1, p2, n1, n2, palm_offset=0.04):
    center = (p1 + p2) / 2
    
    # 计算夹爪方向（x轴）
    x_axis = p2 - p1
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("接触点重合，无法定义抓取方向")
    x_axis /= x_norm
    
    # 计算手掌法向（z轴）
    z_axis = (n1 + n2) / 2
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("法向全零，无法定义手掌法向")
    z_axis /= z_norm
    
    # 确保z轴与x轴正交（Gram-Schmidt过程）
    z_axis = z_axis - np.dot(z_axis, x_axis) * x_axis
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        # 如果z轴与x轴平行，使用备用方法
        if abs(np.dot(z_axis, x_axis)) > 0.99:
            # 找一个与x轴不平行的小向量
            if abs(x_axis[0]) < 0.9:
                temp_vec = np.array([1.0, 0.0, 0.0])
            else:
                temp_vec = np.array([0.0, 1.0, 0.0])
            z_axis = temp_vec - np.dot(temp_vec, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    
    # 计算y轴（完成右手坐标系）
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        raise ValueError("无法构建正交坐标系")
    y_axis /= y_norm
    
    # 重新正交化x轴（确保坐标系完全正交）
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    # 构建旋转矩阵
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    # 确保是右手坐标系（行列式应为+1）
    if np.linalg.det(R) < 0:
        # 如果是左手坐标系，翻转y轴
        y_axis = -y_axis
        R = np.column_stack([x_axis, y_axis, z_axis])
    
    palm_center = center - z_axis * palm_offset
    
    return {'center': palm_center, 'rotation': R}

def pairs_to_grasp_poses(pairs, palm_offset=0.04):
    poses = []
    success_count = 0
    for idx, pair in enumerate(pairs):
        try:
            p1, p2, n1, n2 = pair
            pose = pair_to_grasp_pose(np.array(p1), np.array(p2), np.array(n1), np.array(n2), palm_offset)
            poses.append(pose)
            success_count += 1
        except Exception as e:
            if idx < 10:  # 只打印前10个错误，避免输出太多
                print(f"第{idx}对转换失败: {e}")
            continue
    
    print(f"成功转换 {success_count}/{len(pairs)} 个抓取位姿")
    return poses

if __name__ == "__main__":
    # 检查文件是否存在
    pairs_path = "../results/antipodal_pairs/antipodal_pairs_side.npy"
    if not os.path.exists(pairs_path):
        # 尝试其他可能的文件名
        alternative_path = "../results/antipodal_pairs.npy"
        if os.path.exists(alternative_path):
            pairs_path = alternative_path
            print(f"使用备用文件路径: {pairs_path}")
        else:
            raise FileNotFoundError(f"找不到antipodal pairs文件，请检查路径: {pairs_path}")
    
    pairs = np.load(pairs_path, allow_pickle=True)
    print(f"加载了 {len(pairs)} 对antipodal点")
    
    # 创建输出目录
    os.makedirs("../results/6d", exist_ok=True)
    
    poses = pairs_to_grasp_poses(pairs, palm_offset=0.03)
    
    if len(poses) > 0:
        np.save("../results/6d/grasp_poses.npy", poses)
        print(f"已保存 {len(poses)} 个夹爪6D位姿到 ../results/6d/grasp_poses.npy")
    else:
        print("警告: 没有成功转换任何抓取位姿")