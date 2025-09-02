import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

XML_PATH  = "./g2_eval.xml"
POSE_PATH = "../results/6d/grasp_poses.npy"   # 你的 6D 抓取姿态
POSE_IDX  = 0

# 逐步闭合（不写 0）
OPENING_OPEN   = 0.03     # m，初始开口总和（= left + right）
OPENING_MIN    = 0.0015   # m，最小安全开口
CLOSE_TIME     = 0.8      # s，关指耗时
FPS            = 800      # 提高步率更抗穿透
CLOSE_STEP_MAX = 0.0002   # m/步，单步最大收缩量
BACKOFF_STEP   = 0.0005   # m，渗透回退量

# 接触阈值
FORCE_TH_EACH  = 0.4      # N，每个手指需要的法向力
PEN_THRESH     = 0.0004   # m，允许渗透上限（超过即回退）

# 稳定/落体时间
T_SETTLE   = 0.2
T_HOLD     = 0.3
T_FREEFALL = 2.0

# ---------- 姿态工具 ----------
def quat_to_R(q_xyzw):
    Rm = R.from_quat(np.asarray(q_xyzw, float)).as_matrix()
    U,_,Vt = np.linalg.svd(Rm); Rm = U @ Vt
    if np.linalg.det(Rm) < 0: Rm *= -1.0
    return Rm

def yaw_from_R(Rm):
    # XML 只有 yaw 自由度
    return float(np.arctan2(Rm[1,0], Rm[0,0]))

def load_pose(entry):
    # 支持 dict(center, rotation) / (center, quat, *) / (p1,p2,n1,n2)
    if isinstance(entry, dict) and 'center' in entry and 'rotation' in entry:
        return np.asarray(entry['center'], float), np.asarray(entry['rotation'], float)
    if isinstance(entry, (list,tuple)) and len(entry)>=2:
        a0,a1 = entry[0], entry[1]
        if np.shape(a0)==(3,) and np.shape(a1)==(4,):
            return np.asarray(a0,float), quat_to_R(a1)
    if isinstance(entry, (list,tuple)) and len(entry)==4:
        p1,p2,n1,n2 = map(lambda x: np.asarray(x,float), entry)
        c = 0.5*(p1+p2)
        x = p2-p1; x/=np.linalg.norm(x)+1e-12
        up = np.array([0,0,1.]) if abs(np.dot(x,[0,0,1]))<0.95 else np.array([0,1,0.])
        y = np.cross(up,x); y/=np.linalg.norm(y)+1e-12
        z = np.cross(x,y)
        return c, np.column_stack([x,y,z])
    raise ValueError("未知姿态格式")

# ---------- qpos 写入 ----------
def jaddr(model, jname):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0: raise RuntimeError(f"未找到关节 {jname}")
    return model.jnt_qposadr[jid]

def set_palm_qpos(model, data, center, yaw):
    data.qpos[jaddr(model,"x")]    = float(center[0])
    data.qpos[jaddr(model,"y")]    = float(center[1])
    data.qpos[jaddr(model,"lift")] = float(center[2])
    data.qpos[jaddr(model,"yaw")]  = float(yaw)

def set_opening_qpos(model, data, opening):
    # 腱定义：opening = left + right；限制在各自行程内
    left_max  = 0.015
    right_max = 0.015
    op = float(np.clip(opening, OPENING_MIN, left_max + right_max))
    left  = min(op*0.5, left_max)
    right = min(op*0.5, right_max)
    data.qpos[jaddr(model,"left_joint")]  = left
    data.qpos[jaddr(model,"right_joint")] = right

def measure_finger_contacts(model, data):
    """返回：lF, rF, penL, penR（法向合力与最大渗透量）"""
    bid_left  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    bid_lego  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lego")

    lF = rF = 0.0
    penL = penR = 0.0
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]; b2 = model.geom_bodyid[g2]
        pen = max(0.0, -float(c.dist))  # dist<0 为渗透，取正值
        if (b1==bid_left and b2==bid_lego) or (b2==bid_left and b1==bid_lego):
            mujoco.mj_contactForce(model, data, i, f6)
            lF += abs(f6[0]); penL = max(penL, pen)
        if (b1==bid_right and b2==bid_lego) or (b2==bid_right and b1==bid_lego):
            mujoco.mj_contactForce(model, data, i, f6)
            rF += abs(f6[0]); penR = max(penR, pen)
    return lF, rF, penL, penR

def main():
    print("阶段 0：加载模型与姿态（禁用执行器）…")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # 禁用执行器，确保没有 ctrl 干扰
    try:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    except Exception:
        pass

    poses = np.load(POSE_PATH, allow_pickle=True)
    center, Rm = load_pose(poses[POSE_IDX])
    yaw = yaw_from_R(Rm)  # XML 只有 yaw

    lego_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site < 0: raise RuntimeError("XML 未找到 site 'lego_center'")

    # 先无重力；手有 gravcomp=1，后续开重力也不会受影响
    model.opt.gravity[:] = [0,0,0]
    mujoco.mj_resetData(model, data)

    # 把手直接放到 6D 位姿（用 qpos），夹爪张开
    set_palm_qpos(model, data, center, yaw)
    set_opening_qpos(model, data, OPENING_OPEN)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    nS = max(1, int(T_SETTLE*FPS))
    nH = max(1, int(T_HOLD  *FPS))
    nF = max(1, int(T_FREEFALL*FPS))
    close_steps = max(1, int(CLOSE_TIME * FPS))
    per_step = min((OPENING_OPEN - OPENING_MIN)/close_steps, CLOSE_STEP_MAX)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("阶段 1：固定手到 6D 位姿（无重力），夹爪张开…")
        for _ in range(nS):
            if not viewer.is_running(): return
            mujoco.mj_step(model, data); viewer.sync()

        print("阶段 2：逐步闭合（无重力，检测接触与渗透/回退）…")
        opening = OPENING_OPEN
        for _ in range(close_steps):
            if not viewer.is_running(): return
            opening = max(OPENING_MIN, opening - per_step)
            set_opening_qpos(model, data, opening)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)

            lF, rF, penL, penR = measure_finger_contacts(model, data)
            # 渗透过大 → 回退一点
            if penL > PEN_THRESH or penR > PEN_THRESH:
                opening = min(opening + BACKOFF_STEP, OPENING_OPEN)
                set_opening_qpos(model, data, opening)
                mujoco.mj_forward(model, data)

            # 双侧达到力阈值 → 停止闭合
            if lF >= FORCE_TH_EACH and rF >= FORCE_TH_EACH:
                print(f"  触发停止：L={lF:.2f}N R={rF:.2f}N, opening≈{opening:.4f} m")
                break

            viewer.sync()

        print("阶段 3：稳定保持（无重力）…")
        for _ in range(nH):
            if not viewer.is_running(): return
            mujoco.mj_step(model, data); viewer.sync()

        lego_before = data.site_xpos[lego_site].copy()

        print("阶段 4：开启重力（手 gravcomp，不受重力），观察 LEGO …")
        model.opt.gravity[:] = [0,0,-9.81]
        mujoco.mj_forward(model, data)
        for _ in range(nF):
            if not viewer.is_running(): return
            mujoco.mj_step(model, data); viewer.sync()

        lego_after = data.site_xpos[lego_site].copy()

    disp = float(np.linalg.norm(lego_after - lego_before))
    print("\n===== 结果 =====")
    print(f"输入 center: {center}")
    print(f"LEGO 位移: {disp:.6f} m  （{T_FREEFALL:.2f}s）")
    print("判据: 位移 < 0.005 m → 认为‘未掉落’")
    print("✅ 抓取成功" if disp < 0.005 else "❌ 抓取失败")

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