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
    U, _, Vt = np.linalg.svd(Rmat)
    Rmat = U @ Vt
    if np.linalg.det(Rmat) < 0:
        Rmat *= -1
    if not np.all(np.isfinite(Rmat)):
        raise ValueError("Rmat contains NaN or inf!")
    # 返回 roll, pitch, yaw
    return R.from_matrix(Rmat).as_euler('zyx', degrees=False)[::-1]

def set_gripper_6d_ctrl(model, data, center, Rmat):
    """用6个position执行器的ctrl把底座6D姿态锁到目标"""
    roll, pitch, yaw = mat2eulerZYX(Rmat)
    def aid(name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    ax = aid('x_actuator');     ay = aid('y_actuator');     az = aid('z_actuator')
    aR = aid('roll_actuator');  aP = aid('pitch_actuator'); aY = aid('yaw_actuator')
    assert min(ax,ay,az,aR,aP,aY) >= 0, "找不到6D position执行器，请检查XML里的名字"
    data.ctrl[ax] = float(center[0])
    data.ctrl[ay] = float(center[1])
    data.ctrl[az] = float(center[2])
    data.ctrl[aR] = float(roll)
    data.ctrl[aP] = float(pitch)
    data.ctrl[aY] = float(yaw)

def measure_opening(model, data):
    """用左右指关节位置估计开口大小（越小越闭合）"""
    jidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_joint')
    jidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'right_joint')
    assert jidL >= 0 and jidR >= 0, "未找到 left_joint / right_joint"
    qadrL = model.jnt_qposadr[jidL]
    qadrR = model.jnt_qposadr[jidR]
    # 右指是负向区间，开口粗略用 L - R（右指越靠近0则值越小）
    return float(data.qpos[qadrL] - data.qpos[qadrR])

def auto_pick_close_sign(model, data, aid_grip, test_amp=0.5, steps=80):
    """试探 +amp / -amp 哪个能减小开口，返回推荐符号（+1 或 -1）"""
    qpos0 = data.qpos.copy(); qvel0 = data.qvel.copy()
    act0  = data.act.copy();  ctrl0 = data.ctrl.copy()
    mujoco.mj_forward(model, data)
    base_open = measure_opening(model, data)

    # 试 +amp
    data.ctrl[aid_grip] = +test_amp
    for _ in range(steps): mujoco.mj_step(model, data)
    open_plus = measure_opening(model, data)

    # 回滚
    data.qpos[:] = qpos0; data.qvel[:] = qvel0; data.act[:] = act0; data.ctrl[:] = ctrl0
    mujoco.mj_forward(model, data)

    # 试 -amp
    data.ctrl[aid_grip] = -test_amp
    for _ in range(steps): mujoco.mj_step(model, data)
    open_minus = measure_opening(model, data)

    # 回滚
    data.qpos[:] = qpos0; data.qvel[:] = qvel0; data.act[:] = act0; data.ctrl[:] = ctrl0
    mujoco.mj_forward(model, data)

    # 哪个更小就用哪个方向
    d_plus  = open_plus  - base_open
    d_minus = open_minus - base_open
    return +1 if d_plus < d_minus else -1

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
    pose = poses[3]  # 用第4个抓取
    print("===== 6D 抓取姿态信息 =====")
    print(f"Grasp Center (xyz): {pose['center']}")
    print("Grasp Rotation Matrix (3x3):")
    print(pose['rotation'])

    rpy = mat2eulerZYX(pose['rotation'])
    print(f"Grasp Orientation (roll, pitch, yaw): {rpy}")

    # 1) 无重力，复位
    set_gravity(model, enable=False)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # 2) 用 position 执行器 ctrl 把 6D 位姿锁住；先张开
    set_gripper_6d_ctrl(model, data, pose['center'], pose['rotation'])
    # 通过直接设 qpos 打开手指（只做初始设定，后续靠 motor）
    # 也可以用 motor 反向开合，这里保持简单
    # 左右 slide 关节在 qpos 中的次序与你XML一致时：
    data.qpos[6] = 0.015  # left ~ +15mm
    data.qpos[7] = -0.015 # right ~ -15mm
    mujoco.mj_forward(model, data)

    # 可视化“到位 & 张开”
    print("初始状态：夹爪到位(用ctrl锁住6D), lego固定, 无重力。关闭窗口继续。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            pass

    # 3) 可视化“平滑闭合”：用 gripper_motor 的 ctrl 渐变
    aid_grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor")
    if aid_grip < 0:
        raise RuntimeError("未找到执行器 gripper_motor，请检查XML")

    # 自动判别闭合的控制方向（+ 或 -）
    close_sign = auto_pick_close_sign(model, data, aid_grip, test_amp=0.4, steps=60)
    target_u = 0.9 * close_sign  # 目标 ctrl，留10%余量防颤动
    steps = 300                  # 渐变步数（约 0.3~0.6s）
    print(f"开始抓取：用 ctrl 渐变闭合 (sign={close_sign}, target={target_u:.2f})。关闭窗口继续下一步。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for k in range(steps):
            # 6D 姿态持续锁定（防“飞走”）
            set_gripper_6d_ctrl(model, data, pose['center'], pose['rotation'])
            # 渐变闭合
            alpha = (k + 1) / steps
            data.ctrl[aid_grip] = alpha * target_u
            mujoco.mj_step(model, data)
            viewer.sync()
        # 稳一会
        for _ in range(200):
            set_gripper_6d_ctrl(model, data, pose['center'], pose['rotation'])
            mujoco.mj_step(model, data)
            viewer.sync()

    # 4) 开重力 → 观察位移
    set_gravity(model, enable=True)
    mujoco.mj_forward(model, data)

    lego_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site_id < 0:
        raise RuntimeError("未找到 lego_center site，请检查XML！")
    lego_pos_before = get_lego_pos(data, lego_site_id)

    for _ in range(2000):  # ~2秒
        # 位姿持续锁定
        set_gripper_6d_ctrl(model, data, pose['center'], pose['rotation'])
        mujoco.mj_step(model, data)

    lego_pos_after = get_lego_pos(data, lego_site_id)

    # 5) 判定
    displacement = np.linalg.norm(lego_pos_after - lego_pos_before)
    print(f"LEGO位移: {displacement:.4f} 米")
    if displacement < 0.005:
        print("✅ 抓取成功, LEGO 未掉落。")
    else:
        print("❌ 抓取失败, LEGO 掉落或移动过大。")

    # 6) 最终可视化
    print("最终状态：关闭窗口结束。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            # 6D 姿态仍然锁定，避免掉姿态
            set_gripper_6d_ctrl(model, data, pose['center'], pose['rotation'])
            mujoco.mj_step(model, data)
