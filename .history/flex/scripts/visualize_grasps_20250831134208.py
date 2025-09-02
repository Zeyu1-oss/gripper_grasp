import os, time, argparse
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# ------- 可调参数 -------
TORQUE_OPEN  =  0.5
TORQUE_CLOSE = -0.5            # 如握不住，可改到 -1.5 ~ -2.0，并配合 XML 把 ctrlrange 扩到 -5 5
CONTACT_FORCE_TH = 1.0         # 接触判定阈值（两指合力）
T_SETTLE = 0.50
T_OPEN   = 0.30
T_CLOSE  = 1.00
T_HOLD   = 0.20
T_FREEFALL = 2.00
FPS = 800
# -----------------------

def to_mj_quat_from_R(Rm):
    # scipy as_quat() -> [x,y,z,w]; MuJoCo 用 [w,x,y,z]
    q_xyzw = R.from_matrix(Rm).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

def aid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def jadr(model, jname):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return model.jnt_qposadr[j]

def clamp_ctrl(model, a, u):
    lo, hi = model.actuator_ctrlrange[a]
    return float(np.clip(u, lo, hi))

def sum_contact_forces_between(model, data, body_a, body_b):
    fsum = 0.0; hit = False
    f6 = np.zeros(6, dtype=float)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]; b2 = model.geom_bodyid[g2]
        if (b1 == body_a and b2 == body_b) or (b1 == body_b and b2 == body_a):
            mujoco.mj_contactForce(model, data, i, f6)
            fsum += abs(float(f6[0]))   # 法向力分量
            hit = True
    return fsum, hit

def write_lego_unit(lego_mesh_path, template_path):
    """可选：在运行前自动重写 lego_unit.xml 里的 mesh 文件路径。"""
    content = f'''<mujocoinclude>
  <asset>
    <mesh name="lego" file="{lego_mesh_path}" scale="0.01 0.01 0.01">
      <plugin instance="sdflego"/>
    </mesh>
  </asset>
  <worldbody>
    <body name="lego" pos="0 0 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.02" diaginertia="1.3e-6 1.3e-6 1.3e-6"/>
      <geom type="sdf" name="lego_01" mesh="lego" material="mat_green" friction="2.0 0.01 0.00001">
        <plugin instance="sdflego"/>
      </geom>
      <site name="lego_center" type="sphere" size="0.004" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujocoinclude>
'''
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(content)

def run_once(hand_xml, poses_path, pose_idx, use_viewer=True):
    # 载入模型
    model = mujoco.MjModel.from_xml_path(hand_xml)
    data  = mujoco.MjData(model)

    # 读取抓取姿态
    poses = np.load(poses_path, allow_pickle=True)
    pose  = poses[pose_idx]
    center, Rm = pose['center'], pose['rotation']

    # 名称到 id
    bid_palm  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    bid_left  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    bid_lego  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lego")
    palm_target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_target")
    mocapid = model.body_mocapid[palm_target_bid]

    aL = aid(model, "left_joint")
    aR = aid(model, "right_joint")
    jL = jadr(model, "left_joint")
    jR = jadr(model, "right_joint")

    # 关重力、重置
    model.opt.gravity[:] = [0, 0, 0]
    mujoco.mj_resetData(model, data)

    # 设置 mocap 位姿（精确到 6D）
    data.mocap_pos[mocapid]  = center
    data.mocap_quat[mocapid] = to_mj_quat_from_R(Rm)
    mujoco.mj_forward(model, data)

    # 极性检测（左右独立，避免一边正一边反）
    qL0, qR0 = float(data.qpos[jL]), float(data.qpos[jR])
    tmp = clamp_ctrl(model, aL, TORQUE_OPEN)
    data.ctrl[aL] = tmp; data.ctrl[aR] = tmp
    for _ in range(20): mujoco.mj_step(model, data)
    qL1, qR1 = float(data.qpos[jL]), float(data.qpos[jR])
    invL = (qL1 - qL0) < 0
    invR = (qR1 - qR0) < 0
    open_L  = -TORQUE_OPEN  if invL else TORQUE_OPEN
    open_R  = -TORQUE_OPEN  if invR else TORQUE_OPEN
    close_L = -TORQUE_CLOSE if invL else TORQUE_CLOSE
    close_R = -TORQUE_CLOSE if invR else TORQUE_CLOSE
    data.ctrl[aL] = 0.0; data.ctrl[aR] = 0.0
    for _ in range(10): mujoco.mj_step(model, data)

    # 帧数
    nS = max(1, int(T_SETTLE   * FPS))
    nO = max(1, int(T_OPEN     * FPS))
    nC = max(1, int(T_CLOSE    * FPS))
    nH = max(1, int(T_HOLD     * FPS))
    nF = max(1, int(T_FREEFALL * FPS))

    reached_contact = False
    lego_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")

    # 可视 or 无头
    viewer_ctx = mujoco.viewer.launch_passive(model, data) if use_viewer else None
    def vsync():
        if viewer_ctx: viewer_ctx.sync()

    print("👉 到位稳定...")
    for _ in range(nS):
        data.ctrl[aL] = 0.0; data.ctrl[aR] = 0.0
        mujoco.mj_step(model, data); vsync()

    print("🟢 张开...")
    for _ in range(nO):
        data.ctrl[aL] = clamp_ctrl(model, aL, open_L)
        data.ctrl[aR] = clamp_ctrl(model, aR, open_R)
        mujoco.mj_step(model, data); vsync()

    print("🔴 闭合（零重力）...")
    for _ in range(nC):
        data.ctrl[aL] = clamp_ctrl(model, aL, close_L)
        data.ctrl[aR] = clamp_ctrl(model, aR, close_R)
        mujoco.mj_step(model, data); vsync()

        FL, hitL = sum_contact_forces_between(model, data, bid_left,  bid_lego)
        FR, hitR = sum_contact_forces_between(model, data, bid_right, bid_lego)
        if hitL and hitR and (FL + FR) > CONTACT_FORCE_TH:
            reached_contact = True
            print(f"✅ 接触检测，两指合力 ≈ {FL+FR:.2f} N")
            break

    print("🤏 保持夹紧...")
    for _ in range(nH):
        data.ctrl[aL] = clamp_ctrl(model, aL, close_L)
        data.ctrl[aR] = clamp_ctrl(model, aR, close_R)
        mujoco.mj_step(model, data); vsync()

    lego_before = data.site_xpos[lego_site].copy()

    print("🌍 开启重力，做掉落测试...")
    model.opt.gravity[:] = [0, 0, -9.81]
    mujoco.mj_forward(model, data)
    for _ in range(nF):
        data.ctrl[aL] = clamp_ctrl(model, aL, close_L)
        data.ctrl[aR] = clamp_ctrl(model, aR, close_R)
        mujoco.mj_step(model, data); vsync()

    if viewer_ctx:
        time.sleep(0.5)
        viewer_ctx.close()

    lego_after = data.site_xpos[lego_site].copy()
    disp = float(np.linalg.norm(lego_after - lego_before))

    print("\n===== 结果 =====")
    print(f"位姿 index: {pose_idx}")
    print(f"LEGO 位移: {disp:.6f} m")
    print(f"接触检测: {'成功' if reached_contact else '失败'}")
    ok = (disp < 0.005) and reached_contact
    print(f"判定: {'✅ 抓取成功' if ok else '❌ 抓取失败'}")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand_xml",   type=str, default="../mjcf/g2_hand_base.xml")
    ap.add_argument("--lego_unit",  type=str, default="../mjcf/lego_unit.xml",
                    help="将被本程序重写（或提前手工编辑）以指向 LEGO 网格")
    ap.add_argument("--lego_mesh",  type=str, default="../lego.obj",
                    help="LEGO 网格路径（OBJ/STL）")
    ap.add_argument("--poses",      type=str, default="../results/6d/grasp_poses.npy")
    ap.add_argument("--idx",        type=int, default=3000)
    ap.add_argument("--no_viewer",  action="store_true")
    args = ap.parse_args()

    # 自动把 lego_unit.xml 指向目标网格（也可跳过这步，提前手工写好）
    write_lego_unit(args.lego_mesh, args.lego_unit)

    ok = run_once(args.hand_xml, args.poses, args.idx, use_viewer=not args.no_viewer)
    exit(0 if ok else 1)

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