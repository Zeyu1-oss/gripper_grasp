import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

def mat2eulerZYX(Rmat):
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
    pose_path = "../results/grasp_poses.npy"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    poses = np.load(pose_path, allow_pickle=True)
    pose = poses[0]  # 验证第一个抓取

    # 1. 固定lego，关闭重力
    set_gravity(model, enable=False)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # 2. 设置夹爪6D位姿，张开夹爪
    set_gripper_6d(data, pose['center'], pose['rotation'])
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
    print("夹爪闭合，准备夹取。关闭窗口继续。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            pass

    # 5. 打开重力，模拟一段时间
    set_gravity(model, enable=True)
    mujoco.mj_forward(model, data)
    lego_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    lego_pos_before = get_lego_pos(data, lego_site_id)
    for _ in range(200):  # 模拟0.2秒
        mujoco.mj_step(model, data)
    lego_pos_after = get_lego_pos(data, lego_site_id)

    # 6. 判断lego是否被夹住
    displacement = np.linalg.norm(lego_pos_after - lego_pos_before)
    print(f"LEGO位移: {displacement:.4f} 米")
    if displacement < 0.005:
        print("抓取成功！lego未掉落。")
    else:
        print("抓取失败，lego掉落或移动。")

    # 7. 可视化最终状态
    print("最终状态：关闭窗口结束。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            pass
        