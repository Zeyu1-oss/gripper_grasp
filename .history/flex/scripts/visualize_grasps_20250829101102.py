# import os
# import numpy as np
# import mujoco
# import mujoco.viewer
# from scipy.spatial.transform import Rotation as R
# import time

# class GraspTestEnv:
#     def __init__(self, scene_xml_path, pose_path):
#         os.environ["MUJOCO_GL"] = "glfw"

#         self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
#         self.data = mujoco.MjData(self.model)
#         self.poses = np.load(pose_path, allow_pickle=True)

#         # 关闭重力
#         self.set_gravity(False)
#         mujoco.mj_resetData(self.model, self.data)
#         mujoco.mj_forward(self.model, self.data)

#     # --- 工具函数 ---
#     def mat2eulerZYX(self, Rmat):
#         U, _, Vt = np.linalg.svd(Rmat)
#         Rmat = U @ Vt
#         if np.linalg.det(Rmat) < 0:
#             Rmat *= -1
#         return R.from_matrix(Rmat).as_euler('zyx', degrees=False)[::-1]

#     def set_gravity(self, enable=True):
#         self.model.opt.gravity[:] = [0, 0, -9.81] if enable else [0, 0, 0]

#     def set_gripper_pose(self, center, Rmat):
#         roll, pitch, yaw = self.mat2eulerZYX(Rmat)
#         self.data.qpos[0:3] = center
#         self.data.qpos[3:6] = [roll, pitch, yaw]

#     def set_gripper_opening(self, opening=0.02):
#         self.data.qpos[6] = opening / 2  # left finger
#         self.data.qpos[7] = opening / 2  # right finger

#     def get_lego_pos(self):
#         site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
#         return self.data.site_xpos[site_id].copy()

#     # --- 核心抓取测试 ---
#     def run_single_grasp(self, idx=0, sim_time=2.0):
#         pose = self.poses[idx]
#         print(f"\n===== 抓取 {idx} =====")
#         print(f"Center: {pose['center']}")
#         print(f"Rotation:\n{pose['rotation']}")

#         # Step 1: 初始化，关闭重力，张开夹爪
#         self.set_gravity(False)
#         mujoco.mj_resetData(self.model, self.data)
#         self.set_gripper_pose(pose['center'], pose['rotation'])
#         self.set_gripper_opening(0.03)
#         mujoco.mj_forward(self.model, self.data)

#         lego_before = self.get_lego_pos()

#         # 可视化初始状态
#         with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
#             print("Step1: 初始状态，无重力，夹爪张开。关闭窗口继续。")
#             while viewer.is_running():
#                 viewer.sync()
#                 break  # 只显示一次

#         # Step 2: 闭合夹爪（无重力）
#         self.set_gripper_opening(0.0)
#         mujoco.mj_forward(self.model, self.data)
#         with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
#             print("Step2: 夹爪闭合，无重力。关闭窗口继续。")
#             while viewer.is_running():
#                 viewer.sync()
#                 break

#         # Step 3: 开启重力，模拟几秒
#         self.set_gravity(True)
#         mujoco.mj_forward(self.model, self.data)
#         t0 = self.data.time
#         while self.data.time - t0 < sim_time:
#             mujoco.mj_step(self.model, self.data)

#         lego_after = self.get_lego_pos()
#         displacement = np.linalg.norm(lego_after - lego_before)

#         print(f"LEGO 位移: {displacement:.4f} 米")
#         if displacement < 0.005:
#             print("✅ 抓取成功")
#         else:
#             print("❌ 抓取失败")

#         # Step 4: 可视化最终状态
#         with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
#             print("Step3: 最终状态，关闭窗口结束。")
#             while viewer.is_running():
#                 viewer.sync()
#                 break


# # 使用
# if __name__ == "__main__":
#     env = GraspTestEnv(
#         scene_xml_path="../mjcf/combined_scene.xml",   # 你的场景XML (包含gripper + lego)
#         pose_path="../results/6d/grasp_poses.npy" # 你的6D抓取姿态
#     )

#     # 测试第一个抓取
#     env.run_single_grasp(idx=0, sim_time=2.0)

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
    pose = poses[3]  # 验证第一个抓取
    print("===== 6D 抓取姿态信息 =====")
    print(f"Grasp Center (xyz): {pose['center']}")
    print("Grasp Rotation Matrix (3x3):")
    print(pose['rotation'])

    rpy = mat2eulerZYX(pose['rotation'])
    print(f"Grasp Orientation (roll, pitch, yaw): {rpy}")


    # 1. 固定lego，关闭重力
    set_gravity(model, enable=False)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # 2. 设置夹爪6D位姿，张开夹爪
    try:
        set_gripper_6d(data, pose['center'], pose['rotation'])
    except Exception as e:
        print("旋转矩阵异常，跳过该姿态:", e)
        exit(1)
    set_gripper_opening(data, opening=0.03)
    mujoco.mj_forward(model, data)

    # 3. 可视化初始状态
    print("初始状态：夹爪到位,lego固定,无重力。关闭窗口继续。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            pass

    # 4. 闭合夹爪
    set_gripper_opening(data, opening=0.0)
    mujoco.mj_forward(model, data)
    print("夹爪闭合，准备夹取。关闭窗口继续。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            pass

    # 5. 打开重力，模拟一段时间
    set_gravity(model, enable=True)
    mujoco.mj_forward(model, data)
    lego_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site_id < 0:
        raise RuntimeError("未找到lego_center site，请检查xml文件！")
    lego_pos_before = get_lego_pos(data, lego_site_id)
    for _ in range(2000):  # 模拟2秒
        mujoco.mj_step(model, data)
    lego_pos_after = get_lego_pos(data, lego_site_id)

    # 6. 判断lego是否被夹住
    displacement = np.linalg.norm(lego_pos_after - lego_pos_before)
    print(f"LEGO位移: {displacement:.4f} 米")
    if displacement < 0.005:
        print("抓取成功,lego未掉落。")
    else:
        print("抓取失败,lego掉落或移动。")

    # 7. 可视化最终状态
    print("最终状态：关闭窗口结束。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            pass