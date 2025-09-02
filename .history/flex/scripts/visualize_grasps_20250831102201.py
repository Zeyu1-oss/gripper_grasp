import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

XML_PATH  = "../g2_eval.xml"
POSE_PATH = "../results/6d/grasp_poses.npy"   # 6D 姿态：dict或(center,quat)或(p1,p2,n1,n2)
POSE_IDX  = 0

# 两态扭矩（就是给 motor 的 ctrl），单位≈力（因为 gear=1）
TORQUE_OPEN  = +0.3    # 松开用的扭矩（让两指远离，增大开口）
TORQUE_CLOSE = -0.5    # 加紧用的扭矩（让两指靠拢，减小开口）← 你要的 -0.5

# 时间参数
T_SETTLE   = 0.2   # 到位稳定（无重力）
T_OPEN     = 0.3   # 松开阶段（无重力）
T_TIGHT    = 0.8   # 加紧阶段（无重力）
T_HOLD     = 0.2   # 保持（无重力）
T_FREEFALL = 2.0   # 开重力观察
FPS        = 800

def load_pose(entry):
    # 支持三种格式
    if isinstance(entry, dict) and 'center' in entry and 'rotation' in entry:
        return np.asarray(entry['center'], float), np.asarray(entry['rotation'], float)
    if isinstance(entry, (list,tuple)) and len(entry)>=2:
        a0,a1 = entry[0], entry[1]
        if np.shape(a0)==(3,) and np.shape(a1)==(4,):   # (center, quat(xyzw))
            Rm = R.from_quat(np.asarray(a1,float)).as_matrix()
            return np.asarray(a0,float), Rm
    if isinstance(entry, (list,tuple)) and len(entry)==4:  # (p1,p2,n1,n2)
        p1,p2,n1,n2 = map(lambda x: np.asarray(x,float), entry)
        c = 0.5*(p1+p2)
        x = p2-p1; x/=np.linalg.norm(x)+1e-12
        up = np.array([0,0,1.]) if abs(np.dot(x,[0,0,1]))<0.95 else np.array([0,1,0.])
        y = np.cross(up,x); y/=np.linalg.norm(y)+1e-12
        z = np.cross(x,y)
        return c, np.column_stack([x,y,z])
    raise ValueError("未知姿态格式")

def to_mj_quat_from_R(Rm):
    q_xyzw = R.from_matrix(Rm).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)  # wxyz

def aid(model, name):
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if a < 0: raise RuntimeError(f"未找到 actuator: {name}")
    return a

def jadr(model, jname):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if j < 0: raise RuntimeError(f"未找到 joint: {jname}")
    return model.jnt_qposadr[j]

def clamp_ctrl(model, aid, u):
    lo, hi = model.actuator_ctrlrange[aid]
    return float(np.clip(u, lo, hi))

def main():
    print("阶段 0：加载模型与姿态（motor 两态，SDF）…")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    poses = np.load(POSE_PATH, allow_pickle=True)
    center, Rm = load_pose(poses[POSE_IDX])

    # 关重力并 reset
    model.opt.gravity[:] = [0,0,0]
    mujoco.mj_resetData(model, data)

    # 正确地设置 mocap（用名字→mocapid）
    bid     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_target")
    mocapid = model.body_mocapid[bid]
    data.mocap_pos[mocapid]  = center
    data.mocap_quat[mocapid] = to_mj_quat_from_R(Rm)

    mujoco.mj_forward(model, data)

    # actuator & joints
    aL = aid(model, "left_joint")
    aR = aid(model, "right_joint")
    jL = jadr(model, "left_joint")
    jR = jadr(model, "right_joint")

    nS = max(1, int(T_SETTLE*FPS))
    nO = max(1, int(T_OPEN*FPS))
    nT = max(1, int(T_TIGHT*FPS))
    nH = max(1, int(T_HOLD*FPS))
    nF = max(1, int(T_FREEFALL*FPS))

    # 简单方向自检：把“松开扭矩”打一小会儿，看 qpos 是否增大；若方向反了，反转扭矩。
    mujoco.mj_forward(model, data)
    qL0, qR0 = float(data.qpos[jL]), float(data.qpos[jR])
    tmp_u = clamp_ctrl(model, aL, TORQUE_OPEN)
    data.ctrl[aL] = tmp_u; data.ctrl[aR] = tmp_u
    for _ in range(20): mujoco.mj_step(model, data)
    qL1, qR1 = float(data.qpos[jL]), float(data.qpos[jR])
    if (qL1 - qL0) < 0 or (qR1 - qR0) < 0:
        print("  ⚠️ 发现开/合方向与扭矩符号相反，自动反转 TORQUE_OPEN/TORQUE_CLOSE")
        open_u  = -TORQUE_OPEN
        close_u = -TORQUE_CLOSE
    else:
        open_u  = TORQUE_OPEN
        close_u = TORQUE_CLOSE
    # 清零再 forward
    data.ctrl[aL] = 0.0; data.ctrl[aR] = 0.0
    for _ in range(10): mujoco.mj_step(model, data)

    lego_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site < 0: raise RuntimeError("XML 未找到 site 'lego_center'")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 阶段 1：到位稳定（无重力）
        print("阶段 1：到位稳定（无重力）…")
        for _ in range(nS):
            data.ctrl[aL] = 0.0; data.ctrl[aR] = 0.0
            mujoco.mj_step(model, data); viewer.sync()

        # 阶段 2：松开（无重力）——给“松开扭矩”
        print(f"阶段 2：松开（无重力），u_open = {open_u:.3f}")
        for _ in range(nO):
            data.ctrl[aL] = clamp_ctrl(model, aL, open_u)
            data.ctrl[aR] = clamp_ctrl(model, aR, open_u)
            mujoco.mj_step(model, data); viewer.sync()

        # 阶段 3：加紧（无重力）——给“加紧扭矩”
        print(f"阶段 3：加紧（无重力），u_close = {close_u:.3f}")
        for _ in range(nT):
            data.ctrl[aL] = clamp_ctrl(model, aL, close_u)
            data.ctrl[aR] = clamp_ctrl(model, aR, close_u)
            mujoco.mj_step(model, data); viewer.sync()

        # 阶段 4：保持（无重力）
        print("阶段 4：保持（无重力）…")
        for _ in range(nH):
            data.ctrl[aL] = clamp_ctrl(model, aL, close_u)
            data.ctrl[aR] = clamp_ctrl(model, aR, close_u)
            mujoco.mj_step(model, data); viewer.sync()

        # 记录开重力前 LEGO 位置
        lego_before = data.site_xpos[lego_site].copy()

        # 阶段 5：开重力（手焊死不会动），继续“加紧扭矩”
        print("阶段 5：开启重力，观察 LEGO …")
        model.opt.gravity[:] = [0,0,-9.81]
        mujoco.mj_forward(model, data)
        for _ in range(nF):
            data.ctrl[aL] = clamp_ctrl(model, aL, close_u)
            data.ctrl[aR] = clamp_ctrl(model, aR, close_u)
            mujoco.mj_step(model, data); viewer.sync()

        lego_after = data.site_xpos[lego_site].copy()

    disp = float(np.linalg.norm(lego_after - lego_before))
    print("\n===== 结果（motor 两态 + SDF）=====")
    print(f"6D center: {center}")
    print(f"LEGO 位移: {disp:.6f} m  （{T_FREEFALL:.2f}s）")
    if disp < 0.005:
        print("✅ 抓取成功（未掉落）")
    else:
        print("❌ 抓取失败（掉落或飞走）")

if __name__ == "__main__":
    main()
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