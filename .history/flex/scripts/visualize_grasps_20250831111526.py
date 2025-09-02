# visualize_grasps.py
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import time

# ===== 路径与参数 =====
XML_PATH = "../g2_eval.xml"
POSE_PATH = "../results/6d/grasp_poses.npy"
POSE_IDX = 0

TORQUE_OPEN = -0.5
TORQUE_CLOSE = 0.5

T_SETTLE = 0.20
T_OPEN = 0.30
T_TIGHT = 0.80
T_HOLD = 0.20
T_FREEFALL = 2.00
FPS = 800

CONTACT_FORCE_TH = 1.0

# ===== 工具函数 =====
def load_pose(entry):
    """加载6D姿态"""
    if isinstance(entry, dict) and 'center' in entry and 'rotation' in entry:
        return np.asarray(entry['center'], float), np.asarray(entry['rotation'], float)
    raise ValueError("未知的抓取姿态格式")

def set_gripper_6d_position(model, data, center, Rm):
    """直接设置夹爪的6D位置"""
    # 设置平移
    data.qpos[0] = center[0]  # x
    data.qpos[1] = center[1]  # y  
    data.qpos[2] = center[2]  # z
    
    # 设置旋转 (roll, pitch, yaw)
    euler = R.from_matrix(Rm).as_euler('xyz')
    data.qpos[3] = euler[0]  # roll
    data.qpos[4] = euler[1]  # pitch
    data.qpos[5] = euler[2]  # yaw
    
    mujoco.mj_forward(model, data)

def set_gripper_opening(data, opening=0.03):
    """设置夹爪开口"""
    data.qpos[6] = opening / 2  # 左指
    data.qpos[7] = opening / 2  # 右指

def aid(model, name):
    """获取执行器ID"""
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if a < 0: raise RuntimeError(f"未找到 actuator: {name}")
    return a

def jadr(model, jname):
    """获取关节地址"""
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if j < 0: raise RuntimeError(f"未找到 joint: {jname}")
    return model.jnt_qposadr[j]

def clamp_ctrl(model, aid, u):
    """限制控制信号"""
    lo, hi = model.actuator_ctrlrange[aid]
    return float(np.clip(u, lo, hi))

def sum_contact_forces_between(model, data, body_a, body_b):
    """计算两个body之间的接触力"""
    Fa = 0.0
    hit = False
    f6 = np.zeros(6, dtype=float)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]; b2 = model.geom_bodyid[g2]
        if (b1 == body_a and b2 == body_b) or (b2 == body_a and b1 == body_b):
            mujoco.mj_contactForce(model, data, i, f6)
            Fa += abs(float(f6[0]))
            hit = True
    return Fa, hit

# ===== 主流程 =====
def main():
    print("🔧 加载模型和姿态数据...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 读取抓取姿态
    poses = np.load(POSE_PATH, allow_pickle=True)
    center, Rm = load_pose(poses[POSE_IDX])
    
    print(f"📐 抓取中心: {center}")
    print(f"🎯 抓取姿态矩阵:\n{Rm}")

    # 关重力，重置仿真
    model.opt.gravity[:] = [0, 0, 0]
    mujoco.mj_resetData(model, data)

    # 设置夹爪到目标位姿
    set_gripper_6d_position(model, data, center, Rm)
    set_gripper_opening(data, opening=0.03)  # 初始张开

    # 获取body ID
    bid_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    bid_lego = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lego")
    bid_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")

    # 获取电机ID
    aL = aid(model, "left_joint")
    aR = aid(model, "right_joint")

    # 验证位置
    print(f"\n📍 位置验证:")
    print(f"夹爪位置: {data.qpos[0:3]}")
    print(f"LEGO位置: {data.body(bid_lego).xpos}")
    print(f"位置误差: {np.linalg.norm(data.qpos[0:3] - center)}")

    # 帧数计算
    nS = max(1, int(T_SETTLE * FPS))
    nO = max(1, int(T_OPEN * FPS))
    nT = max(1, int(T_TIGHT * FPS))
    nH = max(1, int(T_HOLD * FPS))
    nF = max(1, int(T_FREEFALL * FPS))

    # 扭矩极性检测
    jL = jadr(model, "left_joint")
    jR = jadr(model, "right_joint")
    
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

    # ===== 可视化抓取流程 =====
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("👀 可视化初始状态...")
        time.sleep(2.0)

        # 1) 到位稳定
        for _ in range(nS):
            data.ctrl[aL] = 0.0; data.ctrl[aR] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()

        # 2) 松开夹爪
        for _ in range(nO):
            u = clamp_ctrl(model, aL, open_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u
            mujoco.mj_step(model, data)
            viewer.sync()

        # 3) 加紧直到接触
        print(f"开始加紧，阈值={CONTACT_FORCE_TH}N...")
        reached_contact = False
        for _ in range(nT):
            u = clamp_ctrl(model, aL, close_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u
            mujoco.mj_step(model, data)
            viewer.sync()

            FL, hitL = sum_contact_forces_between(model, data, bid_left, bid_lego)
            FR, hitR = sum_contact_forces_between(model, data, bid_right, bid_lego)
            if hitL and hitR and (FL + FR) > CONTACT_FORCE_TH:
                reached_contact = True
                break

        # 4) 保持抓取
        for _ in range(nH):
            u = clamp_ctrl(model, aL, close_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u
            mujoco.mj_step(model, data)
            viewer.sync()

        # 记录LEGO初始位置
        lego_before = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")].copy()

        # 5) 开启重力测试
        model.opt.gravity[:] = [0, 0, -9.81]
        mujoco.mj_forward(model, data)
        for _ in range(nF):
            u = clamp_ctrl(model, aL, close_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u
            mujoco.mj_step(model, data)
            viewer.sync()

        lego_after = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")].copy()

    # ===== 结果分析 =====
    disp = float(np.linalg.norm(lego_after - lego_before))
    print("\n===== 抓取结果 =====")
    print(f"抓取中心: {center}")
    print(f"自由落体 {T_FREEFALL:.2f}s 的 LEGO 位移: {disp:.6f} m")
    
    if disp < 0.005:
        print("✅ 抓取成功（未掉落）")
        return True
    else:
        print("❌ 抓取失败（掉落或移动过大）")
        return False

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