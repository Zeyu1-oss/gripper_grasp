import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# ---------- 工具 ----------
def safe_overlay(viewer, title, text, row=0):
    try:
        viewer.add_overlay(mujoco.viewer.Overlay.Text, title, text, 10, 30 + 20*row)
    except Exception:
        pass  # 兼容旧版没有 overlay 的情况

def quat_xyzw_to_R(q):
    Rm = R.from_quat(np.asarray(q, float)).as_matrix()
    U,_,Vt = np.linalg.svd(Rm); Rm = U @ Vt
    if np.linalg.det(Rm) < 0: Rm *= -1.0
    return Rm

def R_to_quat_wxyz(Rm):
    q = R.from_matrix(Rm).as_quat()  # [x,y,z,w]
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)  # MuJoCo wants [w,x,y,z]

def set_freejoint_pose(model, data, joint_name, pos, Rm):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"未找到关节 {joint_name}")
    adr = model.jnt_qposadr[jid]
    q = R_to_quat_wxyz(Rm)
    data.qpos[adr:adr+3]   = np.asarray(pos, float)
    data.qpos[adr+3:adr+7] = q

def get_joint_state(model, data, jname):
    jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    qadr = model.jnt_qposadr[jid]
    dadr = model.jnt_dofadr[jid]
    return float(data.qpos[qadr]), float(data.qvel[dadr])

def motor_pd(model, data, actuator_name, target_pos, kp=50.0, kd=2.0):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if aid < 0: raise RuntimeError(f"未找到执行器 {actuator_name}")
    # 该 motor 绑定的关节
    jid = model.actuator_trnid[aid, 0]
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    q, qd = get_joint_state(model, data, jname)
    u = kp*(target_pos - q) - kd*qd
    lo, hi = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = float(np.clip(u, lo, hi))

def approach_offset(center, Rm, dist=0.05, axis="z-"):
    axes = {"x":0, "y":1, "z":2}
    sign = -1 if axis.endswith("-") else +1
    a = axes[axis[0]]
    dirw = Rm[:, a] * sign
    return np.asarray(center, float) - dist * dirw

def lerp(a, b, t): return (1-t)*a + t*b

def load_pose_entry(entry):
    """
    支持三种格式：
      1) {'center': (3,), 'rotation': (3,3)}
      2) (center, quat_xyzw, quality)
      3) (p1, p2, n1, n2)  -> 闭合轴为 p1->p2
    返回: center(3,), R(3x3)
    """
    if isinstance(entry, dict) and 'center' in entry and 'rotation' in entry:
        return np.asarray(entry['center'], float), np.asarray(entry['rotation'], float)

    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        a0, a1 = entry[0], entry[1]
        if np.shape(a0) == (3,) and np.shape(a1) == (4,):
            return np.asarray(a0, float), quat_xyzw_to_R(a1)

    if isinstance(entry, (list, tuple)) and len(entry) == 4:
        p1, p2, n1, n2 = map(lambda x: np.asarray(x, float), entry)
        c = 0.5*(p1+p2)
        x = p2-p1; x/=np.linalg.norm(x)+1e-12
        up = np.array([0.,0.,1.]) if abs(np.dot(x,[0,0,1]))<0.95 else np.array([0.,1.,0.])
        y0 = np.cross(up, x); y0/=np.linalg.norm(y0)+1e-12
        z0 = np.cross(x, y0)
        Rm = np.column_stack([x, y0, z0])
        return c, Rm

    raise ValueError("姿态条目格式不支持")

# ---------- 主流程 ----------
def run(xml_path="./g2.xml",
        pose_path="../results/6d/grasp_poses.npy",
        pose_index=0,
        approach_dist=0.05, approach_axis="z-",
        open_pos=0.03, close_pos=0.0,
        t_approach=0.6, t_close=0.8, t_hold=0.3, t_freefall=2.0,
        sim_fps=200):
    print("阶段 0：加载模型与姿态…")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    poses = np.load(pose_path, allow_pickle=True)
    entry = poses[pose_index]
    center, Rm = load_pose_entry(entry)

    # LEGO site
    lego_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site < 0: raise RuntimeError("XML 未找到 site 'lego_center'")

    # 初始化
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # 接近点
    approach_center = approach_offset(center, Rm, dist=approach_dist, axis=approach_axis)

    # 打开 viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        hz = sim_fps
        n_approach = max(1, int(t_approach * hz))
        n_close    = max(1, int(t_close    * hz))
        n_hold     = max(1, int(t_hold     * hz))
        n_freefall = max(1, int(t_freefall * hz))

        # 初始：放到接近点，手指张开
        print("阶段 1：接近（freejoint驱动），手指张开…")
        for k in range(n_approach):
            if not viewer.is_running(): break
            c = lerp(approach_center, center, (k+1)/n_approach)
            set_freejoint_pose(model, data, "root", c, Rm)
            # 张开（两个 motor，PD 到 open_pos）
            motor_pd(model, data, "left_joint",  open_pos)
            motor_pd(model, data, "right_joint", open_pos)
            mujoco.mj_step(model, data)
            safe_overlay(viewer, "Phase", "APPROACH", 0)
            viewer.sync()

        print("阶段 2：闭合（motor PD），无重力保持…")
        for k in range(n_close):
            if not viewer.is_running(): break
            motor_pd(model, data, "left_joint",  close_pos)
            motor_pd(model, data, "right_joint", close_pos)
            mujoco.mj_step(model, data)
            safe_overlay(viewer, "Phase", "CLOSE (no gravity)", 0)
            viewer.sync()

        print("阶段 3：无重力保持…")
        for _ in range(n_hold):
            if not viewer.is_running(): break
            motor_pd(model, data, "left_joint",  close_pos)
            motor_pd(model, data, "right_joint", close_pos)
            mujoco.mj_step(model, data)
            safe_overlay(viewer, "Phase", "HOLD (no gravity)", 0)
            viewer.sync()

        # 记录自由落体前 LEGO 位置
        lego_before = data.site_xpos[lego_site].copy()

        print("阶段 4：开启重力，自由落体…")
        model.opt.gravity[:] = [0,0,-9.81]
        mujoco.mj_forward(model, data)
        for _ in range(n_freefall):
            if not viewer.is_running(): break
            motor_pd(model, data, "left_joint",  close_pos)
            motor_pd(model, data, "right_joint", close_pos)
            mujoco.mj_step(model, data)
            safe_overlay(viewer, "Phase", "FREE-FALL (gravity on)", 0)
            viewer.sync()

        lego_after = data.site_xpos[lego_site].copy()

    disp = float(np.linalg.norm(lego_after - lego_before))
    print("===== 结果 =====")
    print(f"Grasp center: {center}")
    print(f"LEGO 位移: {disp:.6f} m  （{t_freefall:.2f}s free-fall）")
    print("判据: 位移 < 0.005 m → 认为‘未掉落’")
    print("✅ 抓取成功" if disp < 0.005 else "❌ 抓取失败")
    return disp

if __name__ == "__main__":
    # 路径按需调整；pose 可是 (center, quat, q) 或 (p1,p2,n1,n2) 或 dict
    run(xml_path=".。/g2.xml",
        pose_path="../results/6d/grasp_poses.npy",
        pose_index=0,
        approach_dist=0.05, approach_axis="z-",
        open_pos=0.03, close_pos=0.00,
        t_approach=0.6, t_close=0.8, t_hold=0.3, t_freefall=2.0,
        sim_fps=200)
    # import mujoco
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