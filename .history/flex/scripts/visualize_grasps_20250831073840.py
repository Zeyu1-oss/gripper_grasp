# viz_grasp_flow.py
import os, time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# ---------- 工具 ----------
def mat2eulerZYX(Rmat):
    U, _, Vt = np.linalg.svd(Rmat)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0: Rm *= -1.0
    return R.from_matrix(Rm).as_euler('zyx', degrees=False)[::-1]  # roll,pitch,yaw

def quat_to_Rmat(quat_xyzw):
    Rm = R.from_quat(np.asarray(quat_xyzw, float)).as_matrix()
    U,_,Vt = np.linalg.svd(Rm); Rm = U @ Vt
    if np.linalg.det(Rm) < 0: Rm *= -1.0
    return Rm

def set_gravity(model, enable=True):
    model.opt.gravity[:] = [0,0,-9.81] if enable else [0,0,0]

def set_gripper_6d(data, center, Rmat):
    data.qpos[0:3] = np.asarray(center, float)
    roll, pitch, yaw = mat2eulerZYX(Rmat)
    data.qpos[3:6] = [roll, pitch, yaw]

def get_lego_pos(data, lego_site_id):
    return data.site_xpos[lego_site_id].copy()

def actuator_opening_id(model, name="gripper_opening"):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0: raise RuntimeError(f"未找到 actuator '{name}'")
    return aid

def set_gripper_opening(model, data, opening, name="gripper_opening"):
    aid = actuator_opening_id(model, name)
    lo, hi = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = float(np.clip(opening, lo, hi))

def approach_offset_world(center, Rmat, dist=0.05, axis="z-"):
    axes = {"x":0, "y":1, "z":2}
    sign = -1 if axis.endswith("-") else +1
    a = axes[axis[0]]
    dir_world = Rmat[:, a] * sign  # R的列即局部轴在世界方向
    return np.asarray(center) - dist * dir_world

def lerp(a, b, t):
    return (1-t)*a + t*b

def load_pose_entry(entry):
    """
    支持三种格式：
      1) dict: {'center': (3,), 'rotation': (3,3)}
      2) (center, quat_xyzw, quality)
      3) (p1, p2, n1, n2)
    返回: (center(3,), Rmat(3x3))
    """
    if isinstance(entry, dict) and 'center' in entry and 'rotation' in entry:
        return np.asarray(entry['center'], float), np.asarray(entry['rotation'], float)

    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        a0, a1 = entry[0], entry[1]
        if np.shape(a0) == (3,) and np.shape(a1) == (4,):
            return np.asarray(a0, float), quat_to_Rmat(a1)

    if isinstance(entry, (list, tuple)) and len(entry) == 4:
        p1, p2, n1, n2 = map(lambda x: np.asarray(x, float), entry)
        c = 0.5*(p1+p2)
        x = p2-p1; x/=np.linalg.norm(x)+1e-12
        up = np.array([0.,0.,1.]) if abs(np.dot(x,[0,0,1]))<0.95 else np.array([0.,1.,0.])
        y0 = np.cross(up, x); y0/=np.linalg.norm(y0)+1e-12
        z0 = np.cross(x, y0)
        Rm = np.column_stack([x, y0, z0])
        return c, Rm

    raise ValueError("未知的姿态条目格式")

# ---------- 主流程 ----------
def run_viz(xml_path, pose_path, pose_index=0,
            opening_open=0.03, opening_close=0.0,
            approach_dist=0.05, approach_axis="z-",
            t_approach=0.6, t_close=0.6, t_hold=0.3, t_freefall=2.0,
            sim_fps=200):
    # 终端提示
    print("阶段 0：加载模型与姿态…")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    poses = np.load(pose_path, allow_pickle=True)
    entry = poses[pose_index]
    center, Rmat = load_pose_entry(entry)
    lego_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site_id < 0: raise RuntimeError("XML 不含 site 'lego_center'")

    # 初始化：无重力
    set_gravity(model, enable=False)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # 计算接近点
    approach_center = approach_offset_world(center, Rmat, dist=approach_dist, axis=approach_axis)

    # 打开 viewer
    print("阶段 1（无重力）：接近中…")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        hz = sim_fps
        n_approach = max(1, int(t_approach * hz))
        n_close    = max(1, int(t_close    * hz))
        n_hold     = max(1, int(t_hold     * hz))
        n_freefall = max(1, int(t_freefall * hz))

        # set initial opening & approach pose
        set_gripper_opening(model, data, opening_open)
        set_gripper_6d(data, approach_center, Rmat)
        mujoco.mj_forward(model, data)

        # Phase 1: approach
        for k in range(n_approach):
            c = lerp(approach_center, center, (k+1)/n_approach)
            set_gripper_6d(data, c, Rmat)
            mujoco.mj_step(model, data)
            viewer.add_overlay(mujoco.viewer.Overlay.Text, "Phase", "APPROACH (no gravity)", 10, 30)
            viewer.sync()

        # Phase 2: close
        print("阶段 2（无重力）：闭合中…")
        for k in range(n_close):
            u = lerp(opening_open, opening_close, (k+1)/n_close)
            set_gripper_opening(model, data, u)
            mujoco.mj_step(model, data)
            viewer.add_overlay(mujoco.viewer.Overlay.Text, "Phase", "CLOSE (no gravity)", 10, 30)
            viewer.add_overlay(mujoco.viewer.Overlay.Text, "Opening (m)", f"{u:.4f}", 10, 50)
            viewer.sync()

        # Phase 3: hold
        print("阶段 3（无重力）：稳定保持…")
        for _ in range(n_hold):
            mujoco.mj_step(model, data)
            viewer.add_overlay(mujoco.viewer.Overlay.Text, "Phase", "HOLD (no gravity)", 10, 30)
            viewer.sync()

        # 记录自由落体前的位置
        lego_pos_before = get_lego_pos(data, lego_site_id)

        # Phase 4: gravity on
        print("阶段 4：开启重力，自由落体模拟…")
        set_gravity(model, enable=True)
        mujoco.mj_forward(model, data)
        for _ in range(n_freefall):
            mujoco.mj_step(model, data)
            viewer.add_overlay(mujoco.viewer.Overlay.Text, "Phase", "FREE-FALL (gravity on)", 10, 30)
            viewer.sync()

        lego_pos_after = get_lego_pos(data, lego_site_id)

    disp = float(np.linalg.norm(lego_pos_after - lego_pos_before))
    print("===== 结果 =====")
    print(f"Grasp center: {center}")
    print(f"LEGO 位移: {disp:.6f} m  （{t_freefall:.2f}s free-fall）")
    print("判据: 位移 < 0.005 m → 认为‘未掉落’")
    print("✅ 抓取成功" if disp < 0.005 else "❌ 抓取失败")
    return disp

if __name__ == "__main__":
    # 路径按你的工程结构改一下
    xml_path  = "../g2_eval.xml"
    pose_path = "../results/6d/grasp_poses.npy"   # 你的姿态文件
    # 选择姿态索引（比如 0 或 99）
    run_viz(xml_path, pose_path, pose_index=0,
            opening_open=0.03, opening_close=0.0,
            approach_dist=0.05, approach_axis="z-",
            t_approach=0.6, t_close=0.6, t_hold=0.3, t_freefall=2.0,
            sim_fps=200)