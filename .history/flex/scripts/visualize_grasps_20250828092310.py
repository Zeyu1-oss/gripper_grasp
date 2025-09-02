import mujoco
import mujoco.viewer
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def mat2quat(Rmat):
    # scipy输出(x, y, z, w)，mujoco需要(w, x, y, z)
    quat_xyzw = R.from_matrix(Rmat).as_quat()
    return np.roll(quat_xyzw, 1)

def set_gripper_pose(data, center, quat):
    data.qpos[:3] = center
    data.qpos[3:7] = quat

def set_finger_opening(data, opening=0.02, left_idx=7, right_idx=8):
    data.qpos[left_idx] = opening / 2
    data.qpos[right_idx] = -opening / 2

if __name__ == "__main__":
    xml_path = "../g2_eval.xml"
    pose_path = "../results/grasp_poses.npy"
    model = mujoco.MjModel.from_xml_path(xml_path)
    poses = np.load(pose_path, allow_pickle=True)
    print(f"共加载 {len(poses)} 个6D抓取姿态")

    with mujoco.viewer.launch_passive(model) as viewer:
        i = 0
        while True:
            data = mujoco.MjData(model)
            pose = poses[i]
            center = pose['center']
            Rmat = pose['rotation']
            quat = mat2quat(Rmat)
            set_gripper_pose(data, center, quat)
            set_finger_opening(data, opening=0.02)
            mujoco.mj_forward(model, data)
            viewer.sync()
            print(f"当前第 {i+1}/{len(poses)} 个抓取，按空格切换下一个，ESC退出")
            key = viewer.user_synchronize()
            if key == mujoco.viewer.KEY_ESC:
                break
            if key == mujoco.viewer.KEY_SPACE:
                i = (i + 1) % len(poses)