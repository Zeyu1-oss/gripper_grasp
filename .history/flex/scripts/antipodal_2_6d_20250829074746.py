import numpy as np
import os

def pair_to_grasp_pose(p1, p2, palm_offset=0.1315):
    """
    最稳定可靠的夹爪6D位姿计算方法
    使用世界坐标系z轴作为夹爪z轴方向
    """
    # 确保输入为浮点数
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    
    # 1. 指尖中心点（原点）
    center = (p1 + p2) / 2.0
    
    # 2. x轴：夹爪开合方向（p1->p2）
    x_axis = p2 - p1
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("接触点重合，无法定义夹爪开合方向")
    x_axis = x_axis / x_norm
    
    # 3. z轴：使用世界坐标系的上方向
    world_z = np.array([0.0, 0.0, 1.0])
    
    # 4. 如果x轴接近垂直，调整参考方向
    if abs(np.dot(x_axis, world_z)) > 0.9:
        world_z = np.array([0.0, 1.0, 0.0])  # 使用y轴作为备选
    
    # 5. 计算与x轴正交的y轴 (y = world_z × x)
    y_axis = np.cross(world_z, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        # 如果叉积为0，使用备选方法
        if abs(x_axis[2]) > 0.9:  # x轴接近z轴
            y_axis = np.array([0.0, 1.0, 0.0])
        else:
            y_axis = np.array([0.0, 0.0, 1.0])
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 6. 计算z轴 (z = x × y，确保右手坐标系)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # 7. 构建旋转矩阵 [x, y, z]
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    # 8. palm位置 = 指尖中心 + z轴 × offset
    # (因为z轴现在是正确的抓取方向，指向夹爪)
    palm_center = center + z_axis * palm_offset
    
    return {'center': palm_center, 'rotation': R}

def pairs_to_grasp_poses(pairs, palm_offset=0.1315):
    """
    将多对antipodal点转换为夹爪palm的6D位姿
    """
    poses = []
    success_count = 0
    
    for idx, pair in enumerate(pairs):
        try:
            p1, p2, n1, n2 = pair  # 虽然不需要法向量，但保持接口一致
            pose = pair_to_grasp_pose(p1, p2, palm_offset)
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
        alternative_path = "../results/antipodal_pairs.npy"
        if os.path.exists(alternative_path):
            pairs_path = alternative_path
            print(f"使用备用文件路径: {pairs_path}")
        else:
            raise FileNotFoundError(f"找不到antipodal pairs文件: {pairs_path}")
    
    pairs = np.load(pairs_path, allow_pickle=True)
    print(f"加载了 {len(pairs)} 对antipodal点")
    
    # 创建输出目录
    os.makedirs("../results/6d", exist_ok=True)
    
    # 转换为夹爪palm的6D位姿
    poses = pairs_to_grasp_poses(pairs, palm_offset=0.1315)  # 使用正值！
    
    if len(poses) > 0:
        np.save("../results/6d/grasp_poses.npy", poses)
        print(f"已保存 {len(poses)} 个夹爪palm 6D位姿")
        
        # 打印第一个位姿作为示例
        first_pose = poses[0]
        print("\n第一个夹爪palm位姿示例:")
        print(f"中心点: {first_pose['center']}")
        print(f"x轴(开合方向): {first_pose['rotation'][:, 0]}")
        print(f"y轴: {first_pose['rotation'][:, 1]}")
        print(f"z轴(抓取方向): {first_pose['rotation'][:, 2]}")
        print(f"旋转矩阵行列式: {np.linalg.det(first_pose['rotation']):.6f}")
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