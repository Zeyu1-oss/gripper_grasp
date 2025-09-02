import os, time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# ---------- 工具 ----------
def mat2eulerZYX(Rmat):
    U, _, Vt = np.linalg.svd(Rmat)
    Rmat = U @ Vt
    if np.linalg.det(Rmat) < 0: Rmat *= -1
    return R.from_matrix(Rmat).as_euler('zyx', degrees=False)[::-1]  # roll,pitch,yaw

def name2id(model, objtype, name):
    try: return mujoco.mj_name2id(model, objtype, name)
    except: return -1

def hold_palm_6d_ctrl(model, data, center, Rmat):
    roll, pitch, yaw = mat2eulerZYX(Rmat)
    # 找6个位置执行器
    aid_x    = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x_actuator")
    aid_y    = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y_actuator")
    aid_z    = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "z_actuator")
    aid_roll = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll_actuator")
    aid_pitch= name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_actuator")
    aid_yaw  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_actuator")
    # 给“目标位姿”
    data.ctrl[aid_x]    = float(center[0])
    data.ctrl[aid_y]    = float(center[1])
    data.ctrl[aid_z]    = float(center[2])
    data.ctrl[aid_roll] = float(roll)
    data.ctrl[aid_pitch]= float(pitch)
    data.ctrl[aid_yaw]  = float(yaw)

def set_gravity(model, enable):
    model.opt.gravity[:] = [0,0,-9.81] if enable else [0,0,0]

def set_opening_by_tendon(model, data, opening):
    """opening = 腱长目标(开口总长度)，配合 XML 里 gripper_motor 是 position@tendon"""
    aid_g = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor")
    lo, hi = model.actuator_ctrlrange[aid_g]
    data.ctrl[aid_g] = float(np.clip(opening, lo, hi))

def get_opening_from_sensor(model, data):
    sid = name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gripper_pos")
    if sid < 0: return None
    adr = model.sensor_adr[sid]
    return float(data.sensordata[adr])

def get_lego_pos(model, data):
    sid = name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    return data.site_xpos[sid].copy()

# ---------- 主流程 ----------
if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "glfw"

    xml_path  = "../g2_eval.xml"                 # 你的（已按上面修改过的）XML
    poses_npy = "../results/6d/grasp_poses.npy"  # 你的6D抓取姿态
    idx       = 123                               # 选一个姿态
    sim_time  = 2.0                               # 开重力后的观察秒数

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    poses = np.load(poses_npy, allow_pickle=True)
    pose  = poses[idx]
    center   = np.array(pose["center"], float)
    Rmat     = np.array(pose["rotation"], float)

    print("===== 6D 抓取姿态 =====")
    print("center:", center)
    print("R:\n", Rmat)

    # 初始：无重力，复位，张开
    set_gravity(model, False)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # 用 ctrl“钉住” palm 的 6D 目标
    hold_palm_6d_ctrl(model, data, center, Rmat)
    set_opening_by_tendon(model, data, opening=0.02)  # 张开 20mm
    for _ in range(200):
        hold_palm_6d_ctrl(model, data, center, Rmat)
        mujoco.mj_step(model, data)

    lego_before = get_lego_pos(model, data)

    # 无重力闭合（把腱长目标从 0.02 线性收小到 0）
    steps_close = 200
    for k in range(steps_close):
        opening = 0.02 * (1.0 - (k+1)/steps_close)
        set_opening_by_tendon(model, data, opening)
        hold_palm_6d_ctrl(model, data, center, Rmat)
        mujoco.mj_step(model, data)

    # 给个缓冲
    for _ in range(100):
        hold_palm_6d_ctrl(model, data, center, Rmat)
        mujoco.mj_step(model, data)

    # 开重力，观察 sim_time 秒，同时持续“钉住” palm、保持闭合
    set_gravity(model, True)
    t0 = data.time
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while data.time - t0 < sim_time and viewer.is_running():
            hold_palm_6d_ctrl(model, data, center, Rmat)  # 每步都喂 ctrl，防飞
            set_opening_by_tendon(model, data, 0.0)       # 保持闭合
            mujoco.mj_step(model, data)
            viewer.sync()

    lego_after = get_lego_pos(model, data)
    disp = np.linalg.norm(lego_after - lego_before)
    opening_now = get_opening_from_sensor(model, data)

    print(f"\nLEGO 位移: {disp*1000:.2f} mm")
    print(f"当前开口(腱长): {opening_now:.4f} m")
    print("结果：", "✅ 成功" if disp < 0.005 else "❌ 失败")

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