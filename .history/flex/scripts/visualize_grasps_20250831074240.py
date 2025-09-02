# visualize_grasps.py
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# ---------- 小工具 ----------
def mat2eulerZYX(Rmat):
    U, _, Vt = np.linalg.svd(Rmat)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0: Rm *= -1.0
    return R.from_matrix(Rm).as_euler('zyx', degrees=False)[::-1]  # roll,pitch,yaw

def quat_to_Rmat(q_xyzw):
    Rm = R.from_quat(np.asarray(q_xyzw, float)).as_matrix()
    U,_,Vt = np.linalg.svd(Rm); Rm = U @ Vt
    if np.linalg.det(Rm) < 0: Rm *= -1.0
    return Rm

def set_gravity(model, enable=True):
    model.opt.gravity[:] = [0,0,-9.81] if enable else [0,0,0]

def set_gripper_6d(data, center, Rmat):
    data.qpos[0:3] = np.asarray(center, float)
    roll,pitch,yaw = mat2eulerZYX(Rmat)
    data.qpos[3:6] = [roll,pitch,yaw]

def get_lego_pos(data, lego_site_id):
    return data.site_xpos[lego_site_id].copy()

def get_actuator_id(model, name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise RuntimeError(f"未找到 actuator '{name}'，请检查 XML")
    return aid

def set_gripper_opening(model, data, opening, name="gripper_opening"):
    aid = get_actuator_id(model, name)
    lo, hi = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = float(np.clip(opening, lo, hi))

# 兼容不同版本 viewer：有 overlay 就用；没有就安静跳过
def add_overlay_safe(viewer, title, text, row=0):
    try:
        Overlay = mujoco.viewer.Overlay  # 旧版有这个枚举
        viewer.add_overlay(Overlay.Text, title, text, 10, 30 + 20*row)
    except Exception:
        # 无 overlay 能力的版本，忽略即可（终端会有同样的阶段提示）
        pass

def approach_offset_world(center, Rmat, dist=0.05, axis="z-"):
    axes = {"x":0, "y":1, "z":2}
    sign = -1 if axis.endswith("-") else +1
    a = axes[axis[0]]
    dir_world = Rmat[:, a] * sign  # R 的列是局部轴在世界方向
    return np.asarray(center) - dist * dir_world

def lerp(a, b, t): return (1-t)*a + t*b

def load_pose_entry(entry):
    """
    支持三种格式：
      1) {'center': (3,), 'rotation': (3,3)}
      2) (center, quat_xyzw, quality)
      3) (p1, p2, n1, n2)
    返回: center(3,), Rmat(3x3)
    """
    # dict
    if isinstance(entry, dict) and 'center' in entry and 'rotation' in entry:
        return np.asarray(entry['center'], float), np.asarray(entry['rotation'], float)
    # (center, quat, *)
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        a0, a1 = entry[0], entry[1]
        if np.shape(a0) == (3,) and np.shape(a1) == (4,):
            return np.asarray(a0, float), quat_to_Rmat(a1)
    # (p1,p2,n1,n2) → 6D
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
    print("阶段 0：加载模型与姿态…")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    poses = np.load(pose_path, allow_pickle=True)
    entry = poses[pose_index]
    center, Rmat = load_pose_entry(entry)

    lego_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site_id < 0: raise RuntimeError("XML 不含 site 'lego_center'")

    # 无重力初始化
    set_gravity(model, enable=False)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # 接近点
    approach_center = approach_offset_world(center, Rmat, dist=approach_dist, axis=approach_axis)

    # 打开 viewer
    print("阶段 1（无重力）：接近中…")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        hz = sim_fps
        n_approach = max(1, int(t_approach * hz))
        n_close    = max(1, int(t_close    * hz))
        n_hold     = max(1, int(t_hold     * hz))
        n_freefall = max(1, int(t_freefall * hz))

        # 初始开口 & 接近位姿
        set_gripper_opening(model, data, opening_open)
        set_gripper_6d(data, approach_center, Rmat)
        mujoco.mj_forward(model, data)

        # APPROACH
        for k in range(n_approach):
            if not viewer.is_running(): break
            c = lerp(approach_center, center, (k+1)/n_approach)
            set_gripper_6d(data, c, Rmat)
            mujoco.mj_step(model, data)
            add_overlay_safe(viewer, "Phase", "APPROACH (no gravity)", 0)
            viewer.sync()

        print("阶段 2（无重力）：闭合中…")
        # CLOSE
        for k in range(n_close):
            if not viewer.is_running(): break
            u = lerp(opening_open, opening_close, (k+1)/n_close)
            set_gripper_opening(model, data, u)
            mujoco.mj_step(model, data)
            add_overlay_safe(viewer, "Phase", "CLOSE (no gravity)", 0)
            add_overlay_safe(viewer, "Opening (m)", f"{u:.4f}", 1)
            viewer.sync()

        print("阶段 3（无重力）：稳定保持…")
        # HOLD
        for _ in range(n_hold):
            if not viewer.is_running(): break
            mujoco.mj_step(model, data)
            add_overlay_safe(viewer, "Phase", "HOLD (no gravity)", 0)
            viewer.sync()

        # 记录 free-fall 前位置
        lego_pos_before = get_lego_pos(data, lego_site_id)

        print("阶段 4：开启重力，自由落体模拟…")
        # FREE-FALL
        set_gravity(model, enable=True)
        mujoco.mj_forward(model, data)
        for _ in range(n_freefall):
            if not viewer.is_running(): break
            mujoco.mj_step(model, data)
            add_overlay_safe(viewer, "Phase", "FREE-FALL (gravity on)", 0)
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
    # 路径按你的工程结构改
    xml_path  = "./g2_eval.xml"
    pose_path = "../results/6d/grasp_poses.npy"
    run_viz(xml_path, pose_path, pose_index=0,
            opening_open=0.03, opening_close=0.0,
            approach_dist=0.05, approach_axis="z-",
            t_approach=0.6, t_close=0.6, t_hold=0.3, t_freefall=2.0,
            sim_fps=200)# import mujoco
# import mujoco.viewer
# import numpy as np
# from scipy.spatial.transform import Rotation as R

# def mat2eulerZYX(Rmat):
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
#     pose = poses[3000]  # 验证第一个抓取
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