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
    # 返回 roll, pitch, yaw
    return R.from_matrix(Rmat).as_euler('zyx', degrees=False)[::-1]

def set_gripper_6d(data, center, Rmat):
    # 直接设置夹爪的6D自由度
    data.qpos[0:3] = center
    roll, pitch, yaw = mat2eulerZYX(Rmat)
    data.qpos[3] = roll
    data.qpos[4] = pitch
    data.qpos[5] = yaw

def set_gripper_opening(data, opening=0.02):
    data.qpos[6] = opening / 2
    data.qpos[7] = opening / 2

def set_gravity(model, enable=True):
    model.opt.gravity[:] = [0, 0, -9.81] if enable else [0, 0, 0]

if __name__ == "__main__":
    xml_path = "../g2_eval.xml"
    pose_path = "../results/6d/grasp_poses.npy"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    poses = np.load(pose_path, allow_pickle=True)
    pose = poses[0]  # 验证第一个抓取

    # ====== 坐标补偿 ======
    # lego mesh缩放
    scale = 0.01
    # lego在mujoco中的位置（来自xml <body name="lego" pos="0 0 0.01">）
    lego_offset = np.array([0, 0, 0.01])
    # 如有body quat，还需补偿旋转（此例无）
    center_mesh = pose['center']
    rotation_mesh = pose['rotation']

    # mesh坐标系 → mujoco世界坐标系
    center_world = center_mesh * scale + lego_offset
    rotation_world = rotation_mesh  # 若body有旋转，这里应 rotation_world = body_rot @ rotation_mesh

    # 1. 固定lego，关闭重力
    set_gravity(model, enable=False)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # 2. 直接设置夹爪6D位姿，张开夹爪
    try:
        set_gripper_6d(data, center_world, rotation_world)
    except Exception as e:
        print("旋转矩阵异常，跳过该姿态:", e)
        exit(1)
    set_gripper_opening(data, opening=0.03)
    mujoco.mj_forward(model, data)

    # 3. 可视化初始状态
    print("初始状态：夹爪到位，lego固定，无重力。关闭窗口继续。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            pass

    # 4. 闭合夹爪
    set_gripper_opening(data, opening=0.0)
    mujoco.mj_forward(model, data)