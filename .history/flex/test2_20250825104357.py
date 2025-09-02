import argparse, time, numpy as np, mujoco, mujoco.viewer

# ---------------- 配置 ----------------
DEFAULT_XML    = "g2_with_30lego.xml"
DEFAULT_TARGET = "lego_09_geom"

XY_ALIGN_TOL   = 0.004
STEP_XY_SETTLE = 10
DOWN_STEP      = 0.003
PRINT_PERIOD   = 0.15

ROT_STEP_MAX   = 0.06   # 单步最大旋转
ROT_TOL        = 0.02   # 收敛阈值 rad (~1.1°)

LIFT_SAFE      = 0.25
LIFT_UP_VALUE  = 0.30

OPEN_CMD       = 0.4
CLOSE_CMD      = -0.4

CONTACT_TH      = 0.4
BOTH_CONTACT_TH = 0.5
DOWN_SAFE_GAP   = 0.0015

VERTICAL_THRESHOLD = 0.7   # 判断竖立的 z 分量阈值

# ---------------- 工具函数 ----------------
def name2id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return None

def contact_normal_force(model, data, i):
    f6 = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, f6)
    return abs(f6[0])

def forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_target):
    leftF = rightF = 0.0
    hitL = hitR = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]; b2 = model.geom_bodyid[g2]
        if (b1 == bid_left and b2 == bid_target) or (b2 == bid_left and b1 == bid_target):
            leftF  += contact_normal_force(model, data, i); hitL = True
        if (b1 == bid_right and b2 == bid_target) or (b2 == bid_right and b1 == bid_target):
            rightF += contact_normal_force(model, data, i); hitR = True
    return leftF, rightF, hitL, hitR

def wait_steps(model, data, n):
    for _ in range(n):
        mujoco.mj_step(model, data)

def body_bottom_z(model, data, bid):
    zmin = +1e9
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid: continue
        if model.geom_conaffinity[gid] == 0: continue
        half_z = float(model.geom_size[gid][2]) if model.geom_size.shape[1] >= 3 else 0.0
        zmin = min(zmin, float(data.geom_xpos[gid][2]) - half_z)
    return zmin

def target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009):
    cz = float(data.geom_xpos[gid_tgt][2])
    half_z = fallback_halfz
    if model.geom_size.shape[1] >= 3:
        hz = float(model.geom_size[gid_tgt][2])
        if hz > 1e-6: half_z = hz
    return cz - half_z

def wrap_to_pi(a): return (a + np.pi) % (2*np.pi) - np.pi
def angle_diff_mod_pi(target, current):
    e = wrap_to_pi(target - current)
    if e >  np.pi/2: e -= np.pi
    if e < -np.pi/2: e += np.pi
    return e

# ---------------- 判断 LEGO 姿态 ----------------
def determine_orientation(model, data, gid_tgt):
    R = data.geom_xmat[gid_tgt].reshape(3,3)
    size = model.geom_size[gid_tgt]
    longest_axis = int(np.argmax(size))
    long_dir = R[:, longest_axis]
    if abs(long_dir[2]) > VERTICAL_THRESHOLD:
        return "vertical", longest_axis
    else:
        return "horizontal", longest_axis

# ---------------- 计算抓取 yaw ----------------
def compute_grasp_yaw(model, data, gid_tgt, state, longest_axis):
    R = data.geom_xmat[gid_tgt].reshape(3,3)   # lego 的旋转矩阵
    x_dir, y_dir, z_dir = R[:,0], R[:,1], R[:,2]

    if state == "vertical":
        # 站着：lego 的 y 轴 ∥ palm 的 y 轴
        dir_xy = np.array([y_dir[0], y_dir[1]])
    else:
        # 躺着：lego 的 z 轴 ∥ palm 的 y 轴
        dir_xy = np.array([z_dir[0], z_dir[1]])

    if np.linalg.norm(dir_xy) < 1e-6:
        return 0.0

    dir_xy /= np.linalg.norm(dir_xy)
    yaw = np.arctan2(dir_xy[1], dir_xy[0])

    # 保证 palm 的 x 轴朝前，而不是反方向
    if state == "vertical":
        # lego 的 x 轴和 palm 的 x 轴对齐
        x_proj = np.dot(x_dir[:2], [np.cos(yaw), np.sin(yaw)])
        if x_proj < 0:   # 如果反了，转 180°
            yaw = wrap_to_pi(yaw + np.pi)
    else:
        # horizontal 时，可以做类似处理，保证抓长边朝外
        y_proj = np.dot(y_dir[:2], [np.cos(yaw), np.sin(yaw)])
        if y_proj < 0:
            yaw = wrap_to_pi(yaw + np.pi)

    return yaw
# ---------------- 主程序 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_XML)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # actuators
    aid_x     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    aid_y     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    aid_lift  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    aid_left  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_joint")
    aid_right = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_joint")
    aid_rot   = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")
    jid_yaw   = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
    qadr_yaw  = model.jnt_qposadr[jid_yaw]

    # bodies
    bid_palm  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    bid_left  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    gid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target)
    bid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.target.replace("_geom",""))

    # init
    loZ, hiZ = model.actuator_ctrlrange[aid_lift]
    loR, hiR = model.actuator_ctrlrange[aid_rot]
    data.ctrl[aid_left] = OPEN_CMD
    data.ctrl[aid_right] = OPEN_CMD
    data.ctrl[aid_lift] = LIFT_SAFE
    data.ctrl[aid_rot]  = 0.0
    wait_steps(model, data, 50)

    palm2tip = data.xpos[bid_palm][2] - body_bottom_z(model,data,bid_left)

    phase = "align_xy"
    state = "unknown"
    desired_yaw = 0.0
    lift_cmd = float(data.ctrl[aid_lift])
    last_print = 0.0

    with mujoco.viewer.launch_passive(model,data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model,data)
            palm = data.xpos[bid_palm]; target = data.geom_xpos[gid_tgt]
            yaw_cur = float(data.qpos[qadr_yaw])

            now = time.time()
            if now-last_print>=PRINT_PERIOD:
                print(f"{phase} state={state} des_yaw={np.degrees(desired_yaw):.1f}° yaw_cur={np.degrees(yaw_cur):.1f}° ctrl_rot={np.degrees(data.ctrl[aid_rot]):.1f}°")
                last_print=now

            if phase=="align_xy":
                data.ctrl[aid_x]=target[0]; data.ctrl[aid_y]=target[1]
                wait_steps(model,data,STEP_XY_SETTLE)
                if abs(palm[0]-target[0])<XY_ALIGN_TOL and abs(palm[1]-target[1])<XY_ALIGN_TOL:
                    state,longest_axis=determine_orientation(model,data,gid_tgt)
                    desired_yaw=compute_grasp_yaw(model,data,gid_tgt,state,longest_axis)
                    phase="rotate"

            elif phase == "rotate":
                err = wrap_to_pi(desired_yaw - yaw_cur)
    
                print(f"[DEBUG] 目标角度: {np.degrees(desired_yaw):.1f}°, "
                      f"当前角度: {np.degrees(yaw_cur):.1f}°, "
                      f"误差: {np.degrees(err):.1f}°, "
                      f"控制命令: {np.degrees(data.ctrl[aid_rot]):.1f}°")
    
                data.ctrl[aid_rot] = float(np.clip(desired_yaw, loR, hiR))
    
    # 观察几帧的变化
                for i in range(5):
                    mujoco.mj_step(model, data)
                    yaw_cur_new = float(data.qpos[qadr_yaw])
                    print(f"  步长 {i}: 角度={np.degrees(yaw_cur_new):.1f}°")
    
                if abs(err) < ROT_TOL:
                    phase = "descend"


            elif phase=="descend":
                tgt_bottom=target_bottom_z(model,data,gid_tgt)
                z_goal=tgt_bottom+DOWN_SAFE_GAP+palm2tip
                err=palm[2]-z_goal
                if err<=0.005: phase="close"
                else:
                    step=min(DOWN_STEP,max(0.0,err)*0.5)
                    lift_cmd=np.clip(lift_cmd-step,loZ,hiZ)
                    data.ctrl[aid_lift]=lift_cmd+0.193
                wait_steps(model,data,2)

            elif phase=="close":
                lF,rF,hitL,hitR=forces_fingers_vs_target_by_body(model,data,bid_left,bid_right,bid_tgt)
                data.ctrl[aid_left]=CLOSE_CMD; data.ctrl[aid_right]=CLOSE_CMD
                wait_steps(model,data,5)
                print(f"[DEBUG] 左手力={lF:.3f}N, 右手力={rF:.3f}N, "
                f"总力={lF+rF:.3f}N, 左接触={hitL}, 右接触={hitR}")            
                if hitL and hitR and (lF+rF)>BOTH_CONTACT_TH: phase="lift"

            elif phase=="lift":
                data.ctrl[aid_left]=CLOSE_CMD; data.ctrl[aid_right]=CLOSE_CMD
                cur=float(data.ctrl[aid_lift])
                step=min(0.003,abs(LIFT_UP_VALUE-cur)*0.5)
                data.ctrl[aid_lift]=np.clip(cur+step,loZ,hiZ)
                wait_steps(model,data,2)

            mujoco.mj_step(model,data); viewer.sync()

if __name__=="__main__":
    main()
