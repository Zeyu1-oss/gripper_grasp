import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# ============ 配置区（请按你的文件和命名修改） ============

XML_PATH = "gripper1.xml"            # 你的 MJCF 文件路径

LEFT_FINGER_NAME  = "left_finger"   # 左手指 geom 名
RIGHT_FINGER_NAME = "right_finger"  # 右手指 geom 名

TARGET_GEOM_NAMES = {               # 目标物体 geom 名（可多个）
    "target_sdf_geom",
    "target_sdf_geom1",
}

GRASP_FORCE_THRESHOLD = 1.0         # 认为“夹住”的左右法向力阈值（牛顿）
PRINT_PERIOD = 0.10                 # 打印间隔（秒），避免刷屏

# ==========================================================

def name2id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return None

def get_geom_ids(model, names):
    ids = set()
    for nm in names:
        gid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if gid is None:
            print(f"[warn] 找不到 geom: {nm}")
        else:
            ids.add(gid)
    return ids

def contact_normal_force(model, data, contact_index):
    f6 = np.zeros(6, dtype=float)
    mujoco.mj_contactForce(model, data, contact_index, f6)
    return abs(f6[0])  # 接触坐标系 x 轴方向的法向力近似

def main():
    if not os.path.exists(XML_PATH):
        print(f"[error] XML 文件不存在: {XML_PATH}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # 手指/目标 geom id
    gid_left  = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_NAME)
    gid_right = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_NAME)
    if gid_left is None or gid_right is None:
        print("[error] 请确认 XML 中给手指 geom 设置了名字，并与脚本常量一致：")
        print(f"       LEFT_FINGER_NAME={LEFT_FINGER_NAME}, RIGHT_FINGER_NAME={RIGHT_FINGER_NAME}")
        sys.exit(1)

    gid_targets = get_geom_ids(model, TARGET_GEOM_NAMES)
    if not gid_targets:
        print("[error] 没有找到任何目标 geom 名称，请检查 TARGET_GEOM_NAMES。")
        sys.exit(1)

    print("仅监控【手指 ↔ 目标】接触：")
    print(f"  左指: {LEFT_FINGER_NAME} (id={gid_left})")
    print(f"  右指: {RIGHT_FINGER_NAME} (id={gid_right})")
    print(f"  目标: {', '.join(TARGET_GEOM_NAMES)}  (ids={sorted(list(gid_targets))})")

    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            left_force_total  = 0.0
            right_force_total = 0.0

            # 只收集 手指<->目标 的接触索引
            left_contacts_idx  = []
            right_contacts_idx = []

            for i in range(data.ncon):
                c = data.contact[i]
                g1, g2 = c.geom1, c.geom2

                # 左指 <-> 目标
                if (g1 == gid_left and g2 in gid_targets) or (g2 == gid_left and g1 in gid_targets):
                    left_contacts_idx.append(i)
                    left_force_total += contact_normal_force(model, data, i)

                # 右指 <-> 目标
                if (g1 == gid_right and g2 in gid_targets) or (g2 == gid_right and g1 in gid_targets):
                    right_contacts_idx.append(i)
                    right_force_total += contact_normal_force(model, data, i)

            # 是否“夹住”：两边都在接触，且法向力超阈值
            grasped = (len(left_contacts_idx) > 0 and len(right_contacts_idx) > 0 and
                       left_force_total  >= GRASP_FORCE_THRESHOLD and
                       right_force_total >= GRASP_FORCE_THRESHOLD)

            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                # 只打印 手指<->目标 的接触数量与明细
                total_ft_contacts = len(left_contacts_idx) + len(right_contacts_idx)
                print(f"[Finger↔Target ncon={total_ft_contacts:2d}] "
                      f"Left: {left_force_total:6.2f} N ({len(left_contacts_idx)}) | "
                      f"Right: {right_force_total:6.2f} N ({len(right_contacts_idx)}) | "
                      f"GRASPED: {grasped}")

                # 明细：左指<->目标
                for i in left_contacts_idx:
                    con = data.contact[i]
                    print(f"  L 接触{i}: geom{con.geom1} <-> geom{con.geom2}, 位置 {con.pos}")

                # 明细：右指<->目标
                for i in right_contacts_idx:
                    con = data.contact[i]
                    print(f"  R 接触{i}: geom{con.geom1} <-> geom{con.geom2}, 位置 {con.pos}")

                last_print = now

            viewer.sync()

if __name__ == "__main__":
    main()
