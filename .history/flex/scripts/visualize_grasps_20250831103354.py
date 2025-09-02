# visualize_grasps.py
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# ===== 路径与参数 =====
XML_PATH  = "../g2_eval.xml"                     # 就用上面的 MJCF
POSE_PATH = "../results/6d/grasp_poses.npy"    # 你的6D抓取结果
POSE_IDX  = 0                                   # 用第几个抓取

# 电机“扭矩两态”（gear=1 时近似力）
TORQUE_OPEN  = -0.5     # 松开
TORQUE_CLOSE = 0.5     # 加紧

# 阶段时长（秒）与帧率
T_SETTLE   = 0.20
T_OPEN     = 0.30
T_TIGHT    = 0.80
T_HOLD     = 0.20
T_FREEFALL = 2.00
FPS        = 800

# 接触判据：双指都触到，且法向力总和大于此阈值才开重力
CONTACT_FORCE_TH = 1   # 牛（N），可按需要调大些 1.0~2.0

# ===== 工具函数 =====
def load_pose(entry):
    """支持三种格式：dict、(center, quat_xyzw, *), (p1,p2,n1,n2)"""
    if isinstance(entry, dict) and 'center' in entry and 'rotation' in entry:
        return np.asarray(entry['center'], float), np.asarray(entry['rotation'], float)
    if isinstance(entry, (list,tuple)) and len(entry) >= 2:
        a0,a1 = entry[0], entry[1]
        if np.shape(a0) == (3,) and np.shape(a1) == (4,):
            Rm = R.from_quat(np.asarray(a1,float)).as_matrix()
            U,_,Vt = np.linalg.svd(Rm); Rm = U@Vt
            if np.linalg.det(Rm) < 0: Rm *= -1.0
            return np.asarray(a0,float), Rm
    if isinstance(entry, (list,tuple)) and len(entry) == 4:
        p1,p2,n1,n2 = map(lambda x: np.asarray(x,float), entry)
        c  = 0.5*(p1+p2)
        x  = p2-p1; x/=np.linalg.norm(x)+1e-12
        up = np.array([0,0,1.]) if abs(np.dot(x,[0,0,1]))<0.95 else np.array([0,1,0.])
        y  = np.cross(up,x); y/=np.linalg.norm(y)+1e-12
        z  = np.cross(x,y)
        return c, np.column_stack([x,y,z])
    raise ValueError("未知的抓取姿态格式")

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

def resolve_lego_getter(model):
    """返回 getter(data)->(3,)，优先 site/lego_center，其次 body/lego，最后含 lego 的 geom。"""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if sid >= 0:
        print("LEGO 锚点：site 'lego_center'")
        return lambda data: data.site_xpos[sid].copy()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lego")
    if bid >= 0:
        print("LEGO 锚点：body 'lego'")
        return lambda data: data.xpos[bid].copy()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if "lego" in name.lower():
            print(f"LEGO 锚点：geom '{name}'")
            return lambda data, gid=gid: data.geom_xpos[gid].copy()
    raise RuntimeError("XML 中没有可识别的 LEGO 锚点（site/body/geom）")

def sum_contact_forces_between(model, data, body_a, body_b):
    """求 A 与 B 的法向力总和（N）以及是否接触。"""
    Fa = 0.0
    hit = False
    f6 = np.zeros(6, dtype=float)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]; b2 = model.geom_bodyid[g2]
        if (b1 == body_a and b2 == body_b) or (b2 == body_a and b1 == body_b):
            mujoco.mj_contactForce(model, data, i, f6)
            Fa += abs(float(f6[0]))  # 接触坐标系的法向分量
            hit = True
    return Fa, hit

# ===== 主流程 =====
def main():
    print("加载模型…")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # 读取抓取姿态
    poses = np.load(POSE_PATH, allow_pickle=True)
    center, Rm = load_pose(poses[POSE_IDX])

    # 关重力 → reset → 设置 mocap（palm_target）→ forward
    model.opt.gravity[:] = [0,0,0]
    mujoco.mj_resetData(model, data)

    bid_target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_target")
    if bid_target < 0: raise RuntimeError("XML 未找到 body 'palm_target'")
    mocapid = model.body_mocapid[bid_target]
    if mocapid < 0: raise RuntimeError("'palm_target' 不是 mocap body（缺少 mocap='true'）")

    data.mocap_pos[mocapid]  = center
    data.mocap_quat[mocapid] = to_mj_quat_from_R(Rm)
    mujoco.mj_forward(model, data)

    # 解析 LEGO 锚点与左右指 body
    lego_get = resolve_lego_getter(model)
    bid_left  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    bid_lego  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lego")
    if min(bid_left, bid_right, bid_lego) < 0:
        raise RuntimeError("找不到 left_link / right_link / lego 的 body")

    # 电机与关节
    aL = aid(model, "left_joint")
    aR = aid(model, "right_joint")
    jL = jadr(model, "left_joint")
    jR = jadr(model, "right_joint")

    # 帧数
    nS = max(1, int(T_SETTLE*FPS))
    nO = max(1, int(T_OPEN*FPS))
    nT = max(1, int(T_TIGHT*FPS))
    nH = max(1, int(T_HOLD*FPS))
    nF = max(1, int(T_FREEFALL*FPS))

    # 自动校正“开/合”极性：松开扭矩应让 qpos 增大
    mujoco.mj_forward(model, data)
    qL0, qR0 = float(data.qpos[jL]), float(data.qpos[jR])
    tmp = clamp_ctrl(model, aL, TORQUE_OPEN)
    data.ctrl[aL] = tmp; data.ctrl[aR] = tmp
    for _ in range(20): mujoco.mj_step(model, data)
    qL1, qR1 = float(data.qpos[jL]), float(data.qpos[jR])
    if (qL1 - qL0) < 0 or (qR1 - qR0) < 0:
        print("⚠️  扭矩极性反了，自动反转 TORQUE_OPEN/TORQUE_CLOSE")
        open_u, close_u = -TORQUE_OPEN, -TORQUE_CLOSE
    else:
        open_u, close_u = TORQUE_OPEN, TORQUE_CLOSE
    data.ctrl[aL] = 0.0; data.ctrl[aR] = 0.0
    for _ in range(10): mujoco.mj_step(model, data)

    # ===== 可视化阶段流程 =====
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # 1) 到位稳定（无重力）
        for _ in range(nS):
            data.ctrl[aL] = 0.0; data.ctrl[aR] = 0.0
            mujoco.mj_step(model, data); viewer.sync()

        # 2) 松开（无重力）：确保有开口
        for _ in range(nO):
            u = clamp_ctrl(model, aL, open_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u
            mujoco.mj_step(model, data); viewer.sync()

        # 3) 加紧（无重力）：持续加紧直到“接触达到阈值”
        print(f"开始加紧，阈值={CONTACT_FORCE_TH}N（双指合力且都触到）…")
        reached_contact = False
        for _ in range(nT):
            u = clamp_ctrl(model, aL, close_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u

            mujoco.mj_step(model, data); viewer.sync()

            # 统计左右指与 LEGO 的接触法向力
            FL, hitL = sum_contact_forces_between(model, data, bid_left,  bid_lego)
            FR, hitR = sum_contact_forces_between(model, data, bid_right, bid_lego)
            if hitL and hitR and (FL + FR) > CONTACT_FORCE_TH:
                reached_contact = True
                break

        # 4) 保持（无重力）
        for _ in range(nH):
            u = clamp_ctrl(model, aL, close_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u
            mujoco.mj_step(model, data); viewer.sync()

        # 记录开重力前 LEGO 位置
        lego_before = lego_get(data)

        # 5) 开启重力，观察自由落体位移（仍保持加紧扭矩）
        model.opt.gravity[:] = [0,0,-9.81]
        mujoco.mj_forward(model, data)
        for _ in range(nF):
            u = clamp_ctrl(model, aL, close_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u
            mujoco.mj_step(model, data); viewer.sync()

        lego_after = lego_get(data)

    # ===== 结果 =====
    disp = float(np.linalg.norm(lego_after - lego_before))
    print("\n===== 结果 =====")
    print(f"抓取中心: {center}")
    print(f"自由落体 {T_FREEFALL:.2f}s 的 LEGO 位移: {disp:.6f} m")
    if disp < 0.005:
        print("✅ 抓取成功（未掉落）")
    else:
        print("❌ 抓取失败（掉落或移动过大）")

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