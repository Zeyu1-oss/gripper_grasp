# grasp_close_then_drop.py
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

XML_PATH  = "../g2_eval_tendon.xml"
POSE_PATH = "../results/6d/grasp_poses.npy"
POSE_IDX  = 0

OPENING_OPEN  = 0.03   # 初始张开
OPENING_CLOSE = 0.00   # 闭合目标

APPROACH_DIST = 0.05   # 从手掌 -Z 方向先退 5cm 再靠近
T_APPROACH    = 0.6    # s
T_CLOSE       = 0.8    # s
T_HOLD        = 0.3    # s
T_FREEFALL    = 2.0    # s
FPS           = 400

# 接触力阈值（牛顿）
CONTACT_TH_ONE  = 0.2     # 单侧指尖-LEGO 触发阈值
CONTACT_TH_BOTH = 0.6     # 双侧合力阈值

def mat2eulerZYX(Rm):
    U,_,Vt = np.linalg.svd(Rm); Rm = U @ Vt
    if np.linalg.det(Rm) < 0: Rm *= -1
    # 返回 roll, pitch, yaw
    return R.from_matrix(Rm).as_euler('zyx', degrees=False)[::-1]

def quat_to_Rmat(q_xyzw):
    Rm = R.from_quat(np.asarray(q_xyzw,float)).as_matrix()
    U,_,Vt = np.linalg.svd(Rm); Rm = U @ Vt
    if np.linalg.det(Rm) < 0: Rm *= -1
    return Rm

def load_pose_entry(entry):
    # 1) {'center':(3,), 'rotation':(3,3)}
    if isinstance(entry, dict) and 'center' in entry and 'rotation' in entry:
        return np.asarray(entry['center'], float), np.asarray(entry['rotation'], float)
    # 2) (center, quat_xyzw, *)
    if isinstance(entry, (list,tuple)) and len(entry) >= 2:
        a0, a1 = entry[0], entry[1]
        if np.shape(a0)==(3,) and np.shape(a1)==(4,):
            return np.asarray(a0,float), quat_to_Rmat(a1)
    # 3) (p1,p2,n1,n2) → 简单构造抓取系
    if isinstance(entry,(list,tuple)) and len(entry)==4:
        p1,p2,n1,n2 = map(lambda x: np.asarray(x,float), entry)
        c = 0.5*(p1+p2)
        x = p2-p1; x/=np.linalg.norm(x)+1e-12
        up = np.array([0,0,1.]) if abs(np.dot(x,[0,0,1]))<0.95 else np.array([0,1,0.])
        y = np.cross(up,x); y/=np.linalg.norm(y)+1e-12
        z = np.cross(x,y)
        return c, np.column_stack([x,y,z])
    raise ValueError("未知姿态格式")

def approach_offset(center, Rm, dist=0.05):
    # 沿手掌局部 -Z 方向后退 dist
    return np.asarray(center) - Rm[:,2]*dist

def set_gravity(model, on):
    model.opt.gravity[:] = [0,0,-9.81] if on else [0,0,0]

def set_palm_ctrl(model, data, center, yaw):
    # 四个位置执行器：x,y,lift,yaw
    ax = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    ay = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    al = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    ar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw")
    data.ctrl[ax] = float(center[0])
    data.ctrl[ay] = float(center[1])
    data.ctrl[al] = float(center[2])
    data.ctrl[ar] = float(yaw)

def set_opening_ctrl(model, data, opening):
    ao = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_opening")
    lo, hi = model.actuator_ctrlrange[ao]
    data.ctrl[ao] = float(np.clip(opening, lo, hi))

def forces_fingers_vs_lego(model, data):
    """返回 (leftF, rightF, hitL, hitR)，法向力之和（N）。"""
    bid_left  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    bid_lego  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lego")
    leftF = rightF = 0.0
    hitL = hitR = False
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]; b2 = model.geom_bodyid[g2]
        # 只统计 finger <-> lego 的接触
        if (b1==bid_left and b2==bid_lego) or (b2==bid_left and b1==bid_lego):
            mujoco.mj_contactForce(model, data, i, f6)
            leftF += abs(f6[0]); hitL = True
        if (b1==bid_right and b2==bid_lego) or (b2==bid_right and b1==bid_lego):
            mujoco.mj_contactForce(model, data, i, f6)
            rightF += abs(f6[0]); hitR = True
    return leftF, rightF, hitL, hitR

def overlay(viewer, text, row=0):
    try:
        viewer.add_overlay(mujoco.viewer.Overlay.Text, "Phase", text, 10, 30 + 20*row)
    except Exception:
        pass

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    poses = np.load(POSE_PATH, allow_pickle=True)
    if POSE_IDX >= len(poses):
        raise IndexError(f"POSE_IDX={POSE_IDX} 超出 grasp_poses 大小 {len(poses)}")
    center, Rm = load_pose_entry(poses[POSE_IDX])
    roll, pitch, yaw = mat2eulerZYX(Rm)

    lego_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site < 0: raise RuntimeError("未找到 site 'lego_center'")

    # 初始化
    set_gravity(model, False)
    mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)

    approach = approach_offset(center, Rm, APPROACH_DIST)

    nA = max(1, int(T_APPROACH*FPS))
    nC = max(1, int(T_CLOSE*FPS))
    nH = max(1, int(T_HOLD*FPS))
    nF = max(1, int(T_FREEFALL*FPS))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 初始：开口 & 到达“接近点”
        set_opening_ctrl(model, data, OPENING_OPEN)
        set_palm_ctrl(model, data, approach, yaw)
        mujoco.mj_forward(model, data)

        # A. 接近
        for k in range(nA):
            if not viewer.is_running(): return
            t = (k+1)/nA
            cur = (1-t)*approach + t*center
            set_palm_ctrl(model, data, cur, yaw)
            mujoco.mj_step(model, data)
            overlay(viewer, "APPROACH (no gravity)")
            viewer.sync()

        # B. 闭合（无重力）
        hitL = hitR = False
        contact_ok = False
        lego_pos_before = None
        for k in range(nC):
            if not viewer.is_running(): return
            t = (k+1)/nC
            opening = (1-t)*OPENING_OPEN + t*OPENING_CLOSE
            set_palm_ctrl(model, data, center, yaw)
            set_opening_ctrl(model, data, opening)
            mujoco.mj_step(model, data)

            lF, rF, hitL, hitR = forces_fingers_vs_lego(model, data)
            contact_ok = (hitL and hitR and (lF+rF) > CONTACT_TH_BOTH) or \
                         ((hitL and lF>CONTACT_TH_ONE) and (hitR and rF>CONTACT_TH_ONE))

            overlay(viewer, f"CLOSE (no gravity) | Opening={opening:.3f} m", 0)
            overlay(viewer, f"Contact L={lF:.2f}N R={rF:.2f}N", 1)
            viewer.sync()

        # C. 稳定保持
        for _ in range(nH):
            if not viewer.is_running(): return
            set_palm_ctrl(model, data, center, yaw)
            set_opening_ctrl(model, data, OPENING_CLOSE)
            mujoco.mj_step(model, data)
            overlay(viewer, "HOLD (no gravity)")
            viewer.sync()

        # 记录自由落体前位置
        lego_pos_before = data.site_xpos[lego_site].copy()

        # 若接触力满足阈值 → 开重力；否则也开，看是否掉落（便于调参）
        set_gravity(model, True)
        mujoco.mj_forward(model, data)

        # D. 自由落体 & 保持夹紧
        for _ in range(nF):
            if not viewer.is_running(): return
            set_palm_ctrl(model, data, center, yaw)
            set_opening_ctrl(model, data, OPENING_CLOSE)
            mujoco.mj_step(model, data)
            overlay(viewer, "FREE-FALL (gravity on)")
            viewer.sync()

        lego_pos_after = data.site_xpos[lego_site].copy()

    disp = float(np.linalg.norm(lego_pos_after - lego_pos_before))
    print("\n===== 结果 =====")
    print(f"位姿 center: {center}")
    print(f"LEGO 位移: {disp:.6f} m  （{T_FREEFALL:.2f}s）")
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