import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

def mat2eulerZYX(Rmat):
    U, _, Vt = np.linalg.svd(Rmat)
    Rmat = U @ Vt
    if np.linalg.det(Rmat) < 0:
        Rmat *= -1
    return R.from_matrix(Rmat).as_euler('zyx', degrees=False)[::-1]  # roll, pitch, yaw

def get_palm_to_tip_offset(model, data, palm_site="gripper_center", tip_site="left_finger_tip"):
    sid_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, palm_site)
    sid_tip  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_site)
    if sid_palm < 0 or sid_tip < 0:
        raise RuntimeError("找不到 palm 或 finger tip site！")
    mujoco.mj_forward(model, data)
    p_palm = data.site_xpos[sid_palm]
    p_tip  = data.site_xpos[sid_tip]
    return np.linalg.norm(p_tip - p_palm)

def set_gripper_6d(data, center, Rmat, palm_to_fingertip):
    """center 是期望指尖中点"""
    # 沿着 -z 偏移回 palm
    offset = -Rmat[:, 2] * palm_to_fingertip
    palm_center = center + offset
    data.qpos[0:3] = palm_center

    roll, pitch, yaw = mat2eulerZYX(Rmat)
    data.qpos[3:6] = [roll, pitch, yaw]

def set_gripper_opening(data, opening=0.02):
    data.qpos[6] = opening / 2
    data.qpos[7] = opening / 2

# === 用法 ===
if __name__ == "__main__":
    xml_path = "../g2_eval.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # 自动算 offset
    offset = get_palm_to_tip_offset(model, data)
    print(f"自动检测 palm->tip 距离: {offset:.4f} m")

    # 假设你有一个 grasp pose
    poses = np.load("../results/6d/grasp_poses.npy", allow_pickle=True)
    pose = poses[0]  # 取第一个

    set_gripper_6d(data, pose['center'], pose['rotation'], offset)
    set_gripper_opening(data, 0.03)
    mujoco.mj_forward(model, data)

    # 可视化看看
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
