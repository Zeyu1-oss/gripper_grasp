import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

# 键盘键码
SPACE_KEY = 32   # 空格：开关重力
G_KEY     = 103  # G键：闭合夹爪
ESC_KEY   = 27   # ESC：退出

# 状态变量
gravity_enabled = False
grab_started = False
simulation_finished = False

def key_callback(key):
    global gravity_enabled, grab_started, simulation_finished
    if key == SPACE_KEY:
        gravity_enabled = not gravity_enabled
        print(f"重力 {'开启' if gravity_enabled else '关闭'}")
        return True
    elif key == G_KEY and not grab_started:
        print("开始抓取：闭合夹爪")
        grab_started = True
        return True
    elif key == ESC_KEY:
        print("用户退出仿真")
        simulation_finished = True
        return True
    return False

def mat2eulerZYX(Rmat):
    """旋转矩阵 -> ZYX欧拉角 (roll,pitch,yaw)"""
    U, _, Vt = np.linalg.svd(Rmat)
    Rmat = U @ Vt
    if np.linalg.det(Rmat) < 0:
        Rmat *= -1
    return R.from_matrix(Rmat).as_euler('zyx', degrees=False)[::-1]

def set_gripper_6d(data, center, Rmat):
    """设置夹爪位姿"""
    data.qpos[0:3] = center
    roll, pitch, yaw = mat2eulerZYX(Rmat)
    data.qpos[3:6] = [roll, pitch, yaw]

def set_gravity(model, enable=True):
    model.opt.gravity[:] = [0, 0, -9.81] if enable else [0, 0, 0]

def get_lego_pos(data, lego_site_id):
    return data.site_xpos[lego_site_id].copy()

if __name__ == "__main__":
    # 路径
    xml_path = "../g2_eval.xml"
    pose_path = "../results/6d/grasp_poses.npy"

    # 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 加载抓取位姿
    poses = np.load(pose_path, allow_pickle=True)
    pose = poses[0]  # 第一个姿态
    print("===== 6D 抓取姿态信息 =====")
    print(f"Grasp Center (xyz): {pose['center']}")
    print("Grasp Rotation Matrix:\n", pose['rotation'])
    print("Euler (r,p,y):", mat2eulerZYX(pose['rotation']))

    # 初始：无重力
    set_gravity(model, enable=False)
    mujoco.mj_resetData(model, data)

    # 设置夹爪位姿 + 张开
    set_gripper_6d(data, pose['center'], pose['rotation'])
    mujoco.mj_forward(model, data)

    # LEGO site
    lego_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site_id < 0:
        raise RuntimeError("lego_center site 未找到")
    lego_pos_before = get_lego_pos(data, lego_site_id)

    print("\n=== 控制说明 ===")
    print("空格: 开启/关闭重力")
    print("G键: 闭合夹爪")
    print("ESC: 退出仿真")
    print("================\n")

    # 获取 gripper motor id
    gripper_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        grab_time = None

        while viewer.is_running() and not simulation_finished:
            # 重力控制
            set_gravity(model, gravity_enabled)

            # 抓取控制：用 motor ctrl 闭合夹爪
            if grab_started:
                data.ctrl[gripper_motor_id] = -1.0  # 给 motor 一个负向指令 -> 闭合
                if grab_time is None:
                    grab_time = data.time

            # 每步保持夹爪6D位姿（固定palm在世界系）
            set_gripper_6d(data, pose['center'], pose['rotation'])

            # step
            mujoco.mj_step(model, data)
            viewer.sync()

            # 判定逻辑（2s后检查）
            if gravity_enabled and grab_started and grab_time is not None:
                if data.time - grab_time > 2.0:
                    lego_pos_after = get_lego_pos(data, lego_site_id)
                    displacement = np.linalg.norm(lego_pos_after - lego_pos_before)
                    print(f"LEGO 位移: {displacement:.4f} m")
                    if displacement < 0.005:
                        print("✅ 成功: LEGO 被抓住")
                    else:
                        print("❌ 失败: LEGO 掉落/移动")
                    simulation_finished = True

            time.sleep(0.001)

    print("\n仿真结束")

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
#     pose = poses[3]  # 验证第一个抓取
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