# grasp_fixed_then_drop.py
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

XML_PATH  = "../g2_eval.xml"                # 上面这份 xml 的文件名
POSE_PATH = "../results/6d/grasp_poses.npy"  # 你的 6D 姿态文件
POSE_IDX  = 0

OPENING_OPEN  = 0.03   # 初始张开（m）
OPENING_CLOSE = 0.00   # 闭合目标（m）

T_SETTLE   = 0.2   # 张开后静置（无重力）s
T_CLOSE    = 0.8   # 闭合时间（无重力）s
T_HOLD     = 0.3   # 闭合后静置（无重力）s
T_FREEFALL = 2.0   # 开重力后模拟 s
FPS        = 400

def quat_to_Rmat(q_xyzw):
    Rm = R.from_quat(np.asarray(q_xyzw,float)).as_matrix()
    U,_,Vt = np.linalg.svd(Rm); Rm = U @ Vt
    if np.linalg.det(Rm) < 0: Rm *= -1.0
    return Rm

def yaw_from_R(Rm):
    # 只取绕世界Z的偏航角；相当于让 gripper 的 x 轴在 XY 平面方向对齐
    return float(np.arctan2(Rm[1,0], Rm[0,0]))

def load_pose_entry(entry):
    # 1) {'center':(3,), 'rotation':(3,3)}
    if isinstance(entry, dict) and 'center' in entry and 'rotation' in entry:
        c = np.asarray(entry['center'], float)
        Rm = np.asarray(entry['rotation'], float)
        return c, Rm
    # 2) (center, quat_xyzw, *)
    if isinstance(entry, (list,tuple)) and len(entry)>=2:
        a0,a1 = entry[0], entry[1]
        if np.shape(a0)==(3,) and np.shape(a1)==(4,):
            return np.asarray(a0,float), quat_to_Rmat(a1)
    # 3) (p1,p2,n1,n2)
    if isinstance(entry, (list,tuple)) and len(entry)==4:
        p1,p2,n1,n2 = map(lambda x: np.asarray(x,float), entry)
        c = 0.5*(p1+p2)
        x = p2-p1; x/=np.linalg.norm(x)+1e-12
        up = np.array([0,0,1.]) if abs(np.dot(x,[0,0,1]))<0.95 else np.array([0,1,0.])
        y = np.cross(up,x); y/=np.linalg.norm(y)+1e-12
        z = np.cross(x,y)
        Rm = np.column_stack([x,y,z])
        return c, Rm
    raise ValueError("未知姿态格式")

def set_palm(model, data, center_xyz, yaw):
    # 位置执行器：直接把目标写到 ctrl
    for name, val in zip(("x","y","lift","yaw"), (center_xyz[0], center_xyz[1], center_xyz[2], yaw)):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0: raise RuntimeError(f"未找到 actuator '{name}'")
        lo,hi = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(val, lo, hi))

def set_opening(model, data, opening):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_opening")
    if aid < 0: raise RuntimeError("未找到 actuator 'gripper_opening'")
    lo,hi = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = float(np.clip(opening, lo, hi))

def run():
    print("阶段 0：加载模型与姿态…")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    poses = np.load(POSE_PATH, allow_pickle=True)
    if POSE_IDX >= len(poses):
        raise IndexError(f"POSE_IDX={POSE_IDX} 超出 grasp_poses 大小 {len(poses)}")
    center, Rm = load_pose_entry(poses[POSE_IDX])

    # 重要提示：该 XML 只有 yaw 自由度，输入姿态的 roll/pitch 会被忽略
    yaw = yaw_from_R(Rm)

    lego_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site < 0: raise RuntimeError("XML 未找到 site 'lego_center'")

    # 无重力初始化
    model.opt.gravity[:] = [0,0,0]
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    nS = max(1, int(T_SETTLE*FPS))
    nC = max(1, int(T_CLOSE *FPS))
    nH = max(1, int(T_HOLD  *FPS))
    nF = max(1, int(T_FREEFALL*FPS))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 固定到给定位姿（不做接近轨迹）
        print("阶段 1：固定手掌到给定位姿（无重力），夹爪张开…")
        set_palm(model, data, center, yaw)
        set_opening(model, data, OPENING_OPEN)
        mujoco.mj_forward(model, data)
        for _ in range(nS):
            if not viewer.is_running(): return
            mujoco.mj_step(model, data); viewer.sync()

        # 闭合
        print("阶段 2：闭合夹爪（无重力）…")
        for k in range(nC):
            if not viewer.is_running(): return
            op = OPENING_OPEN + (OPENING_CLOSE-OPENING_OPEN)*((k+1)/nC)
            set_palm(model, data, center, yaw)   # 始终固定
            set_opening(model, data, op)
            mujoco.mj_step(model, data); viewer.sync()

        # 稳定
        print("阶段 3：稳定保持（无重力）…")
        for _ in range(nH):
            if not viewer.is_running(): return
            set_palm(model, data, center, yaw)
            set_opening(model, data, OPENING_CLOSE)
            mujoco.mj_step(model, data); viewer.sync()

        # 记录 free-fall 前的位置
        lego_before = data.site_xpos[lego_site].copy()

        # 开重力
        print("阶段 4：开启重力，自由落体…")
        model.opt.gravity[:] = [0,0,-9.81]
        mujoco.mj_forward(model, data)
        for _ in range(nF):
            if not viewer.is_running(): return
            set_palm(model, data, center, yaw)
            set_opening(model, data, OPENING_CLOSE)
            mujoco.mj_step(model, data); viewer.sync()

        lego_after = data.site_xpos[lego_site].copy()

    disp = float(np.linalg.norm(lego_after - lego_before))
    print("\n===== 结果 =====")
    print(f"输入 center: {center}")
    print(f"LEGO 位移: {disp:.6f} m  （{T_FREEFALL:.2f}s）")
    print("判据: 位移 < 0.005 m → 认为‘未掉落’")
    print("✅ 抓取成功" if disp < 0.005 else "❌ 抓取失败")

if __name__ == "__main__":
    run()
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