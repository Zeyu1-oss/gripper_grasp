import numpy as np
import os

def pair_to_grasp_pose(p1, p2, n1, n2, palm_offset=-0.1315, mode="up"):
    """
    将一对antipodal点转换为夹爪palm的6D位姿
    """
    # 确保输入为浮点数
    p1 = p1.astype(np.float64)
    p2 = p2.astype(np.float64)
    n1 = n1.astype(np.float64)
    n2 = n2.astype(np.float64)
    
    # 指尖中心点
    center = (p1 + p2) / 2.0
    
    # x轴：夹爪开合方向（从p1指向p2）
    x_axis = p2 - p1
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("接触点重合，无法定义夹爪开合方向")
    x_axis = x_axis / x_norm

    # z轴：抓取方向（明确指向夹爪）
    if mode == "avg":  # 平均法向的反方向（指向夹爪）
        z_axis = -(n1 + n2) / 2.0
    elif mode == "up":  # 固定朝上
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif mode == "random":  # 随机正交方向
        rand = np.random.randn(3).astype(np.float64)
        z_axis = rand - np.dot(rand, x_axis) * x_axis
    else:
        raise ValueError(f"未知模式: {mode}")
    
    # 归一化z轴
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # fallback
    else:
        z_axis = z_axis / z_norm
    
    # 确保z轴与x轴正交
    z_axis = z_axis - np.dot(z_axis, x_axis) * x_axis
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        # 如果z轴与x轴平行，使用备用方向
        if abs(np.dot(np.array([0.0, 0.0, 1.0]), x_axis)) < 0.9:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            z_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        z_axis = z_axis - np.dot(z_axis, x_axis) * x_axis
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # y轴：完成右手坐标系 (y = z × x)
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        raise ValueError("无法构建正交坐标系")
    y_axis = y_axis / y_norm
    
    # 重新正交化x轴确保完全正交
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 构建旋转矩阵 [x, y, z]
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    # 确保是右手坐标系（行列式应为+1）
    if np.linalg.det(R) < 0:
        y_axis = -y_axis
        R = np.column_stack([x_axis, y_axis, z_axis])
    
    # palm位置 = 指尖中心 - z轴 × offset（z轴指向夹爪）
    palm_center = center - z_axis * palm_offset
    
    return {'center': palm_center, 'rotation': R}

def pairs_to_grasp_poses(pairs, palm_offset=0.1315):
    """
    将多对antipodal点转换为夹爪palm的6D位姿
    """
    poses = []
    success_count = 0
    
    for idx, pair in enumerate(pairs):
        try:
            p1, p2, n1, n2 = pair
            # 确保数据为numpy数组并转换为浮点数
            p1 = np.array(p1, dtype=np.float64)
            p2 = np.array(p2, dtype=np.float64)
            n1 = np.array(n1, dtype=np.float64)
            n2 = np.array(n2, dtype=np.float64)
            
            # 确保法向量是单位向量
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            if n1_norm > 1e-8:
                n1 = n1 / n1_norm
            if n2_norm > 1e-8:
                n2 = n2 / n2_norm
            
            pose = pair_to_grasp_pose(p1, p2, n1, n2, palm_offset)
            poses.append(pose)
            success_count += 1
            
        except Exception as e:
            if idx < 5:  # 只打印前几个错误
                print(f"第{idx}对转换失败: {e}")
            continue
    
    print(f"成功转换 {success_count}/{len(pairs)} 个夹爪palm位姿")
    return poses

if __name__ == "__main__":
    # 检查文件是否存在
    pairs_path = "../results/antipodal_pairs/antipodal_pairs_ray.npy"
    if not os.path.exists(pairs_path):
        # 尝试其他可能的文件名
        alternative_path = "../results/antipodal_pairs.npy"
        if os.path.exists(alternative_path):
            pairs_path = alternative_path
            print(f"使用备用文件路径: {pairs_path}")
        else:
            raise FileNotFoundError(f"找不到antipodal pairs文件: {pairs_path}")
    
    pairs = np.load(pairs_path, allow_pickle=True)
    print(f"加载了 {len(pairs)} 对antipodal点")
    
    # 检查第一对数据的类型
    if len(pairs) > 0:
        p1, p2, n1, n2 = pairs[0]
        print(f"数据类型: p1={type(p1)}, n1={type(n1)}")
        print(f"数据形状: p1={np.array(p1).shape}, n1={np.array(n1).shape}")
    
    # 创建输出目录
    os.makedirs("../results/6d", exist_ok=True)
    
    # 转换为夹爪palm的6D位姿（使用正确的offset）
    poses = pairs_to_grasp_poses(pairs, palm_offset=-0.1315)
    
    if len(poses) > 0:
        np.save("../results/6d/grasp_poses.npy", poses)
        print(f"已保存 {len(poses)} 个夹爪palm 6D位姿到 ../results/6d/grasp_poses.npy")
        
        # 打印第一个位姿作为示例
        first_pose = poses[0]
        print("\n第一个夹爪palm位姿示例:")
        print(f"中心点: {first_pose['center']}")
        print(f"x轴(开合方向): {first_pose['rotation'][:, 0]}")
        print(f"y轴: {first_pose['rotation'][:, 1]}")
        print(f"z轴(抓取方向): {first_pose['rotation'][:, 2]}")
        print(f"旋转矩阵:\n{first_pose['rotation']}")
    else:
        print("警告: 没有成功转换任何夹爪位姿")
# import numpy as np
# import os
# def pair_to_grasp_pose(p1, p2, n1, n2, palm_offset=0.1315, mode="random"):
#     center = (p1 + p2) / 2
#     x_axis = (p2 - p1)
#     if np.linalg.norm(x_axis) < 1e-8:
#         raise ValueError("接触点重合，无法定义抓取方向")
#     x_axis /= np.linalg.norm(x_axis)

#     if mode == "avg":  # 平均法向
#         z_axis = (n1 + n2)
#         if np.linalg.norm(z_axis) < 1e-8:
#             z_axis = np.array([0,0,1])  # fallback
#     elif mode == "up":  # 固定向上
#         z_axis = np.array([0,0,1])
#         if abs(np.dot(z_axis, x_axis)) > 0.9:  # 防止平行
#             z_axis = np.array([0,1,0])
#     elif mode == "random":  # 随机正交
#         rand = np.random.randn(3)
#         z_axis = rand - np.dot(rand, x_axis) * x_axis
#     else:
#         raise ValueError(f"未知模式: {mode}")

#     z_axis /= np.linalg.norm(z_axis)
#     y_axis = np.cross(z_axis, x_axis)
#     y_axis /= np.linalg.norm(y_axis)
#     x_axis = np.cross(y_axis, z_axis)
#     x_axis /= np.linalg.norm(x_axis)

#     R = np.stack([x_axis, y_axis, z_axis], axis=1)
#     palm_center = center - z_axis * palm_offset
#     return {'center': palm_center, 'rotation': R}


# def pairs_to_grasp_poses(pairs, palm_offset=0):
#     poses = []
#     for idx, (p1, p2, n1, n2) in enumerate(pairs):
#         try:
#             pose = pair_to_grasp_pose(np.array(p1), np.array(p2), np.array(n1), np.array(n2), palm_offset)
#             poses.append(pose)
#         except Exception as e:
#             print(f"第{idx}对转换失败: {e}")
#     return poses

# if __name__ == "__main__":
#     pairs = np.load("../results/antipodal_pairs/antipodal_pairs_ray.npy", allow_pickle=True)
#     os.makedirs("../results/6d", exist_ok=True)
#     poses = pairs_to_grasp_poses(pairs, palm_offset=0.03)
#     np.save("../results/6d/grasp_poses.npy", poses)
#     if len(poses) > 0:
#         np.save("../results/6d/grasp_poses.npy", poses)
#         print(f"已保存 {len(poses)} 个夹爪指尖6D位姿到 ../results/6d/grasp_poses.npy")
        
#         # 打印第一个位姿作为示例
#         first_pose = poses[0]
#         print("\n第一个夹爪指尖位姿示例:")
#         print(f"中心点: {first_pose['center']}")
#         print(f"x轴(开合方向): {first_pose['rotation'][:, 0]}")
#         print(f"y轴: {first_pose['rotation'][:, 1]}")
#         print(f"z轴(朝上): {first_pose['rotation'][:, 2]}")
#         print(f"旋转矩阵:\n{first_pose['rotation']}")
#     else:
#         print("警告: 没有成功转换任何夹爪指尖位姿")