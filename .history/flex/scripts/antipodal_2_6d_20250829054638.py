# import numpy as np
# import os
# def pair_to_grasp_pose(p1, p2, n1, n2, palm_offset=0.000001, mode="random"):
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
#     print(f"已保存 {len(poses)} 个夹爪6D位姿到 ../results/6d/grasp_poses.npy")
import numpy as np
import os

def pair_to_grasp_pose(p1, p2, n1, n2):
    """
    将一对antipodal点转换为夹爪指尖中心的6D位姿
    
    参数:
        p1, p2: 两个接触点的3D位置
        n1, n2: 两个接触点的法向量（指向物体内部）
    
    返回:
        夹爪指尖中心的6D位姿 {center, rotation}
    """
    # 指尖中心点（夹爪坐标系原点）
    center = (p1 + p2) / 2
    
    # x轴：夹爪开合方向（从p1指向p2）
    x_axis = p2 - p1
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("接触点重合，无法定义夹爪开合方向")
    x_axis /= x_norm
    
    # z轴：抓取方向（平均法向的反方向，因为法向量指向物体内部）
    z_axis = -(n1 + n2) / 2  # 取反，指向夹爪方向
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("法向全零，无法定义抓取方向")
    z_axis /= z_norm
    
    # 确保z轴与x轴正交
    z_axis = z_axis - np.dot(z_axis, x_axis) * x_axis
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        # 如果z轴与x轴平行，使用备用方法
        temp_vec = np.array([0.0, 0.0, 1.0])  # 默认使用z轴方向
        if abs(np.dot(temp_vec, x_axis)) > 0.9:
            temp_vec = np.array([1.0, 0.0, 0.0])  # 如果与z轴平行，改用x轴
        z_axis = temp_vec - np.dot(temp_vec, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    
    # y轴：完成右手坐标系 (y = z × x)
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        raise ValueError("无法构建正交坐标系")
    y_axis /= y_norm
    
    # 重新正交化确保完全正交
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    # 构建旋转矩阵 [x, y, z]
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    # 确保是右手坐标系（行列式应为+1）
    if np.linalg.det(R) < 0:
        y_axis = -y_axis
        R = np.column_stack([x_axis, y_axis, z_axis])
    
    return {'center': center, 'rotation': R}

def pairs_to_grasp_poses(pairs):
    """
    将多对antipodal点转换为夹爪指尖中心的6D位姿
    
    参数:
        pairs: 列表，每个元素是 (p1, p2, n1, n2)
    
    返回:
        夹爪指尖中心的6D位姿列表
    """
    poses = []
    success_count = 0
    
    for idx, pair in enumerate(pairs):
        try:
            p1, p2, n1, n2 = pair
            # 确保法向量是单位向量
            n1 = n1 / np.linalg.norm(n1)
            n2 = n2 / np.linalg.norm(n2)
            
            pose = pair_to_grasp_pose(np.array(p1), np.array(p2), 
                                     np.array(n1), np.array(n2))
            poses.append(pose)
            success_count += 1
            
        except Exception as e:
            if idx < 5:  # 只打印前几个错误
                print(f"第{idx}对转换失败: {e}")
            continue
    
    print(f"成功转换 {success_count}/{len(pairs)} 个夹爪指尖位姿")
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
    
    # 创建输出目录
    os.makedirs("../results/6d", exist_ok=True)
    
    # 转换为夹爪指尖中心的6D位姿
    poses = pairs_to_grasp_poses(pairs)
    
    if len(poses) > 0:
        np.save("../results/6d/grasp_poses.npy", poses)
        print(f"已保存 {len(poses)} 个夹爪指尖6D位姿到 ../results/6d/grasp_poses.npy")
        
        # 打印第一个位姿作为示例
        first_pose = poses[0]
        print("\n第一个夹爪指尖位姿示例:")
        print(f"中心点: {first_pose['center']}")
        print(f"旋转矩阵:\n{first_pose['rotation']}")
    else:
        print("警告: 没有成功转换任何夹爪指尖位姿")