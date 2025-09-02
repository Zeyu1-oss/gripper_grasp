import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

def mat2eulerZYX(Rmat):
    # 确保Rmat为正交矩阵
    U, _, Vt = np.linalg.svd(Rmat)
    Rmat = U @ Vt
    if np.linalg.det(Rmat) < 0:
        Rmat *= -1
    if not np.all(np.isfinite(Rmat)):
        raise ValueError("Rmat contains NaN or inf!")
    return R.from_matrix(Rmat).as_euler('zyx', degrees=False)[::-1]  # 返回 roll, pitch, yaw

def set_gripper_6d(data, center, Rmat):
    # 位置
    data.qpos[0:3] = center
    # 姿态（roll, pitch, yaw）
    roll, pitch, yaw = mat2eulerZYX(Rmat)
    data.qpos[3] = roll
    data.qpos[4] = pitch
    data.qpos[5] = yaw

def set_gripper_opening(data, opening=0.02):
    # 左右指对称
    data.qpos[6] = opening / 2
    data.qpos[7] = opening / 2

def control_gripper_motor(data, model, control_value):
    """通过电机控制夹爪开合"""
    gripper_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor")
    if gripper_motor_id < 0:
        raise RuntimeError("未找到gripper_motor执行器，请检查xml文件！")
    
    # 设置电机控制信号：-1为张开，1为闭合
    data.ctrl[gripper_motor_id] = control_value

def get_gripper_opening(data, model):
    """获取夹爪开口大小"""
    left_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_joint")
    right_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_joint")
    
    if left_joint_id >= 0 and right_joint_id >= 0:
        left_pos = data.qpos[6]  # 假设left_joint在qpos的第6位
        right_pos = data.qpos[7]  # 假设right_joint在qpos的第7位
        return abs(left_pos) + abs(right_pos)
    return 0.0

def set_gravity(model, enable=True):
    model.opt.gravity[:] = [0, 0, -9.81] if enable else [0, 0, 0]

def get_lego_pos(data, lego_site_id):
    return data.site_xpos[lego_site_id].copy()

if __name__ == "__main__":
    xml_path = "../g2_eval.xml"
    pose_path = "../results/6d/grasp_poses.npy"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    poses = np.load(pose_path, allow_pickle=True)
    pose = poses[123]  # 验证第一个抓取
    
    print("===== 6D 抓取姿态信息 =====")
    print(f"Grasp Center (xyz): {pose['center']}")
    print("Grasp Rotation Matrix (3x3):")
    print(pose['rotation'])

    rpy = mat2eulerZYX(pose['rotation'])
    print(f"Grasp Orientation (roll, pitch, yaw): {rpy}")

    # 1. 关闭重力
    set_gravity(model, enable=False)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # 2. 设置夹爪6D位姿，张开夹爪
    try:
        set_gripper_6d(data, pose['center'], pose['rotation'])
    except Exception as e:
        print("旋转矩阵异常，跳过该姿态:", e)
        exit(1)
    
    # 初始张开夹爪
    set_gripper_opening(data, opening=0.03)
    mujoco.mj_forward(model, data)

    # 3. 可视化初始状态
    print("初始状态：夹爪到位,lego固定,无重力。关闭窗口继续。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            pass

    # 4. 使用电机控制逐步闭合夹爪（可视化过程）
    print("开始使用电机控制闭合夹爪...")
    
    # 获取电机ID
    gripper_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor")
    if gripper_motor_id < 0:
        raise RuntimeError("未找到gripper_motor执行器！")
    
    # 逐步闭合夹爪（可视化过程）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("逐步增加夹紧力度...")
        
        # 逐步增加闭合力度（从0到1）
        for step in range(100):
            control_value = min(1.0, step * 0.01)  # 从0到1逐步增加
            control_gripper_motor(data, model, control_value)
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 获取当前夹爪开口大小
            opening = get_gripper_opening(data, model)
            print(f"Step {step}: 控制信号 = {control_value:.2f}, 夹爪开口 = {opening:.4f}m")
            
            if opening < 0.001:  # 夹爪基本闭合
                print("夹爪已完全闭合")
                break
        
        # 保持最大夹紧力度一段时间
        print("保持最大夹紧力度...")
        control_gripper_motor(data, model, 1.0)  # 最大夹紧力度
        for _ in range(100):
            mujoco.mj_step(model, data)
            viewer.sync()
            opening = get_gripper_opening(data, model)
            print(f"保持状态: 控制信号 = 1.0, 夹爪开口 = {opening:.4f}m")

    # 5. 打开重力，模拟一段时间
    set_gravity(model, enable=True)
    mujoco.mj_forward(model, data)
    
    lego_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site_id < 0:
        raise RuntimeError("未找到lego_center site，请检查xml文件！")
    
    lego_pos_before = get_lego_pos(data, lego_site_id)
    
    # 6. 模拟重力作用（保持夹紧状态）
    print("开始重力模拟（保持夹紧状态）...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(2000):  # 模拟2秒
            # 保持夹紧状态
            control_gripper_motor(data, model, 1.0)
            
            mujoco.mj_step(model, data)
            
            # 每100步同步一次视图并显示状态
            if step % 100 == 0:
                viewer.sync()
                opening = get_gripper_opening(data, model)
                lego_pos = get_lego_pos(data, lego_site_id)
                print(f"模拟步数 {step}: 夹爪开口 = {opening:.4f}m, LEGO高度 = {lego_pos[2]:.4f}m")
    
    lego_pos_after = get_lego_pos(data, lego_site_id)

    # 7. 判断lego是否被夹住
    displacement = np.linalg.norm(lego_pos_after - lego_pos_before)
    final_opening = get_gripper_opening(data, model)
    
    print(f"LEGO位移: {displacement:.4f} 米")
    print(f"最终夹爪开口: {final_opening:.4f} 米")
    print(f"LEGO最终高度: {lego_pos_after[2]:.4f} 米")
    
    # 判断抓取是否成功
    if displacement < 0.005 and final_opening < 0.005:
        print("✅ 抓取成功: LEGO未掉落且夹爪保持闭合")
    elif displacement < 0.01:
        print("⚠️  部分成功: LEGO有轻微移动")
    else:
        print("❌ 抓取失败: LEGO掉落或移动过大")

    # 8. 可视化最终状态
    print("最终状态：关闭窗口结束。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            pass

    # 9. 可选：松开夹爪观察效果
    print("松开夹爪观察效果...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 逐步松开夹爪
        for step in range(100):
            control_value = max(-1.0, 1.0 - step * 0.02)  # 从1到-1逐步松开
            control_gripper_motor(data, model, control_value)
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            opening = get_gripper_opening(data, model)
            print(f"松开步骤 {step}: 控制信号 = {control_value:.2f}, 夹爪开口 = {opening:.4f}m")
        
        viewer.sync()
        while viewer.is_running():
            pass
# import mujoco
# import mujoco.viewer
# import numpy as np
# from scipy.spatial.transform import Rotation as R

# def mat2eulerZYX(Rmat):
#     # 确保Rmat为正交矩阵
#     U, _, Vt = np.linalg.svd(Rmat)
#     Rmat = U @ Vt
#     if np.linalg.det(Rmat) < 0:
#         Rmat *= -1
#     if not np.all(np.isfinite(Rmat)):
#         raise ValueError("Rmat contains NaN or inf!")
#     return R.from_matrix(Rmat).as_euler('zyx', degrees=False)[::-1]  # 返回 roll, pitch, yaw

# def set_gripper_6d(data, center, Rmat):
#     # 位置
#     data.qpos[0:3] = center
#     # 姿态（roll, pitch, yaw）
#     roll, pitch, yaw = mat2eulerZYX(Rmat)
#     data.qpos[3] = roll
#     data.qpos[4] = pitch
#     data.qpos[5] = yaw

# def set_gripper_opening(data, opening=0.02):
#     # 左右指对称
#     data.qpos[6] = opening / 2
#     data.qpos[7] = opening / 2

# def set_gravity(model, enable=True):
#     model.opt.gravity[:] = [0, 0, -9.81] if enable else [0, 0, 0]

# def get_lego_pos(data, lego_site_id):
#     return data.site_xpos[lego_site_id].copy()

# if __name__ == "__main__":
#     xml_path = "../g2_eval.xml"
#     pose_path = "../results/6d/grasp_poses.npy"
#     model = mujoco.MjModel.from_xml_path(xml_path)
#     data = mujoco.MjData(model)
#     poses = np.load(pose_path, allow_pickle=True)
#     pose = poses[123]  # 验证第一个抓取
#     print("===== 6D 抓取姿态信息 =====")
#     print(f"Grasp Center (xyz): {pose['center']}")
#     print("Grasp Rotation Matrix (3x3):")
#     print(pose['rotation'])

#     rpy = mat2eulerZYX(pose['rotation'])
#     print(f"Grasp Orientation (roll, pitch, yaw): {rpy}")


#     # 1. 固定lego，关闭重力
#     set_gravity(model, enable=False)
#     mujoco.mj_resetData(model, data)
#     mujoco.mj_forward(model, data)

#     # 2. 设置夹爪6D位姿，张开夹爪
#     try:
#         set_gripper_6d(data, pose['center'], pose['rotation'])
#     except Exception as e:
#         print("旋转矩阵异常，跳过该姿态:", e)
#         exit(1)
#     set_gripper_opening(data, opening=0.03)
#     mujoco.mj_forward(model, data)

#     # 3. 可视化初始状态
#     print("初始状态：夹爪到位,lego固定,无重力。关闭窗口继续。")
#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         viewer.sync()
#         while viewer.is_running():
#             pass

#     # 4. 闭合夹爪
#     set_gripper_opening(data, opening=0.0)
#     mujoco.mj_forward(model, data)
#     print("夹爪闭合，准备夹取。关闭窗口继续。")
#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         viewer.sync()
#         while viewer.is_running():
#             pass

#     # 5. 打开重力，模拟一段时间
#     set_gravity(model, enable=True)
#     mujoco.mj_forward(model, data)
#     lego_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
#     if lego_site_id < 0:
#         raise RuntimeError("未找到lego_center site，请检查xml文件！")
#     lego_pos_before = get_lego_pos(data, lego_site_id)
#     for _ in range(2000):  # 模拟2秒
#         mujoco.mj_step(model, data)
#     lego_pos_after = get_lego_pos(data, lego_site_id)

#     # 6. 判断lego是否被夹住
#     displacement = np.linalg.norm(lego_pos_after - lego_pos_before)
#     print(f"LEGO位移: {displacement:.4f} 米")
#     if displacement < 0.005:
#         print("抓取成功,lego未掉落。")
#     else:
#         print("抓取失败,lego掉落或移动。")

#     # 7. 可视化最终状态
#     print("最终状态：关闭窗口结束。")
#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         viewer.sync()
#         while viewer.is_running():
#             pass