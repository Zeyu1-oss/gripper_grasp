# grasp_lego_v2.py
import argparse, time, numpy as np, mujoco, mujoco.viewer

DEFAULT_XML    = "gripper1.xml"
DEFAULT_TARGET = "lego_02_geom"

PRINT_PERIOD            = 0.15
XY_ALIGN_TOL            = 0.004
Z_TOL                   = 0.003
DESCEND_STEP            = 0.003      # 每步下降 3mm（关节坐标）
BOUNCE_UP_AFTER_FLOOR   = 0.010      # 触底回弹 1cm
APPROACH_GAP_ABOVE_TOP  = 0.004      # 到砖上表面上方 4mm
GRASP_FORCE_THRESHOLD   = 0.6
GRASP_STEP              = 0.008
LIFT_AFTER_GRASP_CTRL   = 0.30       # 成功后抬到这个 lift ctrl

def name2id(model, objtype, name):
    try: return mujoco.mj_name2id(model, objtype, name)
    except: return None

def clamp_ctrl(model, aid, val):
    lo,hi = model.actuator_ctrlrange[aid]; return float(np.clip(val, lo, hi))

def contact_normal_force(model, data, i):
    f6 = np.zeros(6); mujoco.mj_contactForce(model, data, i, f6); return abs(f6[0])

def sum_forces_with_target(model, data, gidL, gidR, gidT):
    lF=rF=0.0; hitL=hitR=False
    for i in range(data.ncon):
        c = data.contact[i]; g1,g2 = c.geom1, c.geom2
        if (g1==gidL and g2==gidT) or (g2==gidL and g1==gidT):
            lF += contact_normal_force(model, data, i); hitL=True
        if (g1==gidR and g2==gidT) or (g2==gidR and g1==gidT):
            rF += contact_normal_force(model, data, i); hitR=True
    return lF, rF, hitL, hitR

def finger_gap(model, data, gidL, gidR):
    mujoco.mj_forward(model, data)
    lx = data.geom_xpos[gidL][0]; rx = data.geom_xpos[gidR][0]
    return abs(rx-lx)

def finger_bottom_z(model, data, gid):
    mujoco.mj_forward(model, data)
    zc = float(data.geom_xpos[gid][2]); h = float(model.geom_size[gid][2])
    return zc - h

def top_z(model, data, gid):
    mujoco.mj_forward(model, data)
    zc = float(data.geom_xpos[gid][2]); h = float(model.geom_size[gid][2])
    return zc + h

def touched(model, data, gidA, gidB):
    for i in range(data.ncon):
        c = data.contact[i]; g1,g2=c.geom1,c.geom2
        if (g1==gidA and g2==gidB) or (g2==gidA and g1==gidB): return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, default=DEFAULT_XML)
    ap.add_argument("--target", type=str, default=DEFAULT_TARGET)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model); mujoco.mj_forward(model, data)

    # ids
    aid_x = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    aid_y = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    aid_lift  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    aid_grasp = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grasp")
    gid_left   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger")
    gid_right  = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger")
    gid_palm   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "palm")
    gid_floor  = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tray_floor")
    gid_target = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target)
    if None in (aid_x,aid_y,aid_lift,aid_grasp,gid_left,gid_right,gid_palm,gid_floor,gid_target):
        print("[ERROR] 名称映射失败。"); return

    # 初始张开、抬高
    data.ctrl[aid_grasp] = 0.5*(model.actuator_ctrlrange[aid_grasp][0] + model.actuator_ctrlrange[aid_grasp][1])
    data.ctrl[aid_lift]  = clamp_ctrl(model, aid_lift, 0.30)
    for _ in range(40): mujoco.mj_step(model, data)

    # 自动判定 grasp 正方向（哪个方向是闭合）
    g0 = finger_gap(model, data, gid_left, gid_right)
    data.ctrl[aid_grasp] = clamp_ctrl(model, aid_grasp, data.ctrl[aid_grasp]+0.05); 
    for _ in range(5): mujoco.mj_step(model, data)
    g1 = finger_gap(model, data, gid_left, gid_right)
    close_dir = -1.0 if g1>g0 else 1.0
    data.ctrl[aid_grasp] = clamp_ctrl(model, aid_grasp, 0.5); 
    for _ in range(5): mujoco.mj_step(model, data)
    print(f"[INFO] grasp ctrl 正方向是{'闭合' if close_dir>0 else '张开'}")

    phase = "calib_down"
    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)

            # 快捷量
            tgt_xyz = data.geom_xpos[gid_target].copy()
            tgt_top = top_z(model, data, gid_target)
            floor_top = top_z(model, data, gid_floor)
            fL_bot = finger_bottom_z(model, data, gid_left)
            fR_bot = finger_bottom_z(model, data, gid_right)
            fbot   = 0.5*(fL_bot + fR_bot)

            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                print(f"[{phase:10s}] xy=({tgt_xyz[0]:+.3f},{tgt_xyz[1]:+.3f}) "
                      f"fbot={fbot:.3f} floor_top={floor_top:.3f} tgt_top={tgt_top:.3f} "
                      f"ctrl(lift)={data.ctrl[aid_lift]:+.3f}")
                last_print = now

            if phase == "calib_down":
                # 直降到触底（指尖或手掌碰地）
                if touched(model, data, gid_floor, gid_left) or \
                   touched(model, data, gid_floor, gid_right) or \
                   touched(model, data, gid_floor, gid_palm):
                    phase = "bounce_up"
                else:
                    lo = model.actuator_ctrlrange[aid_lift][0]
                    if data.ctrl[aid_lift] <= lo + 1e-6:
                        print("[WARN] 到了 lift 下限还没触底：请把 joint.range/ctrlrange 下限再降（如 -0.20）。")
                        break
                    data.ctrl[aid_lift] = clamp_ctrl(model, aid_lift, data.ctrl[aid_lift] - DESCEND_STEP)

            elif phase == "bounce_up":
                # 碰到托盘：回弹 1cm
                target_ctrl = clamp_ctrl(model, aid_lift, data.ctrl[aid_lift] + BOUNCE_UP_AFTER_FLOOR)
                data.ctrl[aid_lift] = target_ctrl
                # 回弹达成（以指尖底-托盘顶的几何差为标准）
                if (fbot - floor_top) >= (BOUNCE_UP_AFTER_FLOOR*0.7):
                    phase = "align_xy"

            elif phase == "align_xy":
                # XY 对齐到目标
                data.ctrl[aid_x] = clamp_ctrl(model, aid_x, float(tgt_xyz[0]))
                data.ctrl[aid_y] = clamp_ctrl(model, aid_y, float(tgt_xyz[1]))
                palm = data.geom_xpos[gid_palm]
                if abs(palm[0]-tgt_xyz[0])<XY_ALIGN_TOL and abs(palm[1]-tgt_xyz[1])<XY_ALIGN_TOL:
                    phase = "descend_to_top"

            elif phase == "descend_to_top":
                # 降到“砖上表面 + 间隙”
                z_goal = tgt_top + APPROACH_GAP_ABOVE_TOP
                if fbot - z_goal > Z_TOL:
                    data.ctrl[aid_lift] = clamp_ctrl(model, aid_lift, data.ctrl[aid_lift] - DESCEND_STEP)
                else:
                    phase = "close"

            elif phase == "close":
                # 逐步闭合，监测与目标的接触力
                data.ctrl[aid_grasp] = clamp_ctrl(model, aid_grasp, data.ctrl[aid_grasp] + close_dir*GRASP_STEP)
                lF, rF, hitL, hitR = sum_forces_with_target(model, data, gid_left, gid_right, gid_target)
                if hitL and hitR and lF>=GRASP_FORCE_THRESHOLD and rF>=GRASP_FORCE_THRESHOLD:
                    phase = "lift"
                # 如果到端点还没夹住，就再往下微降 3mm 重试一次
                g_lo,g_hi = model.actuator_ctrlrange[aid_grasp]
                if (abs(data.ctrl[aid_grasp]-g_lo)<1e-6 or abs(data.ctrl[aid_grasp]-g_hi)<1e-6) and (lF+rF<0.4*GRASP_FORCE_THRESHOLD):
                    data.ctrl[aid_lift] = clamp_ctrl(model, aid_lift, data.ctrl[aid_lift] - 0.003)

            elif phase == "lift":
                data.ctrl[aid_lift] = clamp_ctrl(model, aid_lift, LIFT_AFTER_GRASP_CTRL)

            mujoco.mj_step(model, data); viewer.sync()

if __name__ == "__main__":
    main()
