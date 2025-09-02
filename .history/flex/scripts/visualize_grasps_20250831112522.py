import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import time

# ===== 路径与参数 =====
XML_PATH = "../g2_eval.xml"
POSE_PATH = "../results/6d/grasp_poses.npy"
POSE_IDX = 3000  # 使用您调好的第3000个抓取

# 电机参数
TORQUE_OPEN = 0.5
TORQUE_CLOSE = -0.5

# 时间参数
T_SETTLE = 0.50  # 稳定时间
T_OPEN = 0.30    # 张开时间
T_CLOSE = 1.00   # 闭合时间
T_HOLD = 0.20    # 保持时间
T_FREEFALL = 2.00  # 自由落体时间
FPS = 800

# 接触力阈值
CONTACT_FORCE_TH = 1.0

def to_mj_quat_from_R(Rm):
    """旋转矩阵转MuJoCo四元数(wxyz)"""
    q_xyzw = R.from_matrix(Rm).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

def aid(model, name):
    """获取执行器ID"""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def jadr(model, jname):
    """获取关节地址"""
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
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

def main():
    print("🔧 加载模型和姿态数据...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 读取抓取姿态
    poses = np.load(POSE_PATH, allow_pickle=True)
    pose = poses[POSE_IDX]
    center, Rm = pose['center'], pose['rotation']
    
    print(f"📐 使用第{POSE_IDX}个抓取姿态")
    print(f"抓取中心: {center}")
    print(f"旋转矩阵行列式: {np.linalg.det(Rm):.6f}")

    # 关重力，重置仿真
    model.opt.gravity[:] = [0, 0, 0]
    mujoco.mj_resetData(model, data)

    # 设置mocap到精确位置
    bid_target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_target")
    mocapid = model.body_mocapid[bid_target]
    
    data.mocap_pos[mocapid] = center
    data.mocap_quat[mocapid] = to_mj_quat_from_R(Rm)
    mujoco.mj_forward(model, data)

    # 获取body和电机ID
    bid_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    bid_lego = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lego")
    bid_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")

    aL = aid(model, "left_joint")
    aR = aid(model, "right_joint")
    jL = jadr(model, "left_joint")
    jR = jadr(model, "right_joint")

    # 验证位置精度
    print(f"\n📍 位置验证:")
    print(f"Mocap位置: {data.mocap_pos[mocapid]}")
    print(f"Palm位置: {data.xpos[bid_palm]}")
    print(f"位置误差: {np.linalg.norm(data.mocap_pos[mocapid] - data.xpos[bid_palm]):.6f}")

    # 帧数计算
    nS = max(1, int(T_SETTLE * FPS))
    nO = max(1, int(T_OPEN * FPS))
    nC = max(1, int(T_CLOSE * FPS))
    nH = max(1, int(T_HOLD * FPS))
    nF = max(1, int(T_FREEFALL * FPS))

    # 扭矩极性检测
    qL0, qR0 = float(data.qpos[jL]), float(data.qpos[jR])
    tmp = clamp_ctrl(model, aL, TORQUE_OPEN)
    data.ctrl[aL] = tmp; data.ctrl[aR] = tmp
    for _ in range(20): mujoco.mj_step(model, data)
    qL1, qR1 = float(data.qpos[jL]), float(data.qpos[jR])
    
    if (qL1 - qL0) < 0 or (qR1 - qR0) < 0:
        print("⚠️  扭矩极性反了，自动反转")
        open_u, close_u = -TORQUE_OPEN, -TORQUE_CLOSE
    else:
        open_u, close_u = TORQUE_OPEN, TORQUE_CLOSE
    
    data.ctrl[aL] = 0.0; data.ctrl[aR] = 0.0
    for _ in range(10): mujoco.mj_step(model, data)

    # ===== 可视化抓取流程 =====
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("👀 初始状态：夹爪到位，准备抓取...")
        time.sleep(2.0)

        # 1) 稳定位置
        print("🔄 稳定位置...")
        for _ in range(nS):
            data.ctrl[aL] = 0.0; data.ctrl[aR] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()

        # 2) 张开夹爪
        print("🟢 张开夹爪...")
        for _ in range(nO):
            u = clamp_ctrl(model, aL, open_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u
            mujoco.mj_step(model, data)
            viewer.sync()

        # 3) 闭合夹爪（使用电机扭矩）
        print("🔴 闭合夹爪...")
        reached_contact = False
        for _ in range(nC):
            u = clamp_ctrl(model, aL, close_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u
            mujoco.mj_step(model, data)
            viewer.sync()

            # 检测接触
            FL, hitL = sum_contact_forces_between(model, data, bid_left, bid_lego)
            FR, hitR = sum_contact_forces_between(model, data, bid_right, bid_lego)
            if hitL and hitR and (FL + FR) > CONTACT_FORCE_TH:
                reached_contact = True
                print(f"✅ 接触检测到！力: {FL + FR:.2f}N")
                break

        # 4) 保持抓取
        print("🤏 保持抓取...")
        for _ in range(nH):
            u = clamp_ctrl(model, aL, close_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u
            mujoco.mj_step(model, data)
            viewer.sync()

        # 记录LEGO初始位置
        lego_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
        lego_before = data.site_xpos[lego_site_id].copy()

        # 5) 开启重力测试
        print("🌍 开启重力测试...")
        model.opt.gravity[:] = [0, 0, -9.81]
        mujoco.mj_forward(model, data)
        
        for _ in range(nF):
            # 保持夹爪闭合扭矩
            u = clamp_ctrl(model, aL, close_u)
            data.ctrl[aL] = u; data.ctrl[aR] = u
            mujoco.mj_step(model, data)
            viewer.sync()

        lego_after = data.site_xpos[lego_site_id].copy()

    # ===== 结果分析 =====
    disp = float(np.linalg.norm(lego_after - lego_before))
    print("\n===== 抓取结果 =====")
    print(f"LEGO位移: {disp:.6f} m")
    print(f"接触检测: {'成功' if reached_contact else '失败'}")
    
    if disp < 0.005 and reached_contact:
        print("✅ 抓取成功！LEGO未掉落且接触良好")
        return True
    else:
        print("❌ 抓取失败")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n测试结果: {'成功' if success else '失败'}")# import mujoco
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