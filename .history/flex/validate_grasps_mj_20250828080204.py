#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import os, json, time, math, argparse, traceback
import numpy as np
import multiprocessing as mp

# 设置环境变量
if '--egl' in str(sys.argv):
    os.environ['MUJOCO_GL'] = 'egl'

import mujoco
from mujoco import viewer

# -------------------- 常量定义 --------------------
SETTLE_STEPS        = 200         # 初始沉降步数
XY_ALIGN_TOL        = 0.001       # xy 对齐阈值 (m)
STEP_XY_SETTLE      = 50          # xy 控制后静置步数
DOWN_STEP           = 0.002       # 每次下压增量 (m)
DOWN_SAFE_GAP       = 0.0015      # 手指尖与目标"底面"安全间隙 (m)
ROT_TOL             = 0.02        # 旋转收敛阈值 (rad)
OPEN_CMD            = 0.8
CLOSE_CMD           = -0.6
BOTH_CONTACT_TH     = 0.3         # 合力阈值 (N) 判定"夹住"
APPROACH_DIST       = 0.04        # 沿抓取的接近方向退开这么远作为起始位姿 (m)
LIFT_CLEAR          = 0.08        # 成功后抬到离地高度 (m)
HOLD_TIME           = 1.0         # 抬起后保持时间 (s)
PRINT_EVERY         = 10          # 打印调试信息间隔

# -------------------- 工具函数 --------------------
def name2id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return -1

def wait_steps(model, data, n):
    for _ in range(n):
        mujoco.mj_step(model, data)

def wrap_to_pi(a): 
    return (a + np.pi) % (2*np.pi) - np.pi

def body_bottom_z(model, data, bid):
    zmin = +1e9
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid: 
            continue
        if model.geom_conaffinity[gid] == 0: 
            continue
        half_z = float(model.geom_size[gid][2]) if model.geom_size.shape[1] >= 3 else 0.0
        zmin = min(zmin, float(data.geom_xpos[gid][2]) - half_z)
    return zmin

def target_halfz_bottom_top(model, data, gid_tgt, fallback_halfz=0.009):
    cz = float(data.geom_xpos[gid_tgt][2])
    half_z = fallback_halfz
    if model.geom_size.shape[1] >= 3:
        hz = float(model.geom_size[gid_tgt][2])
        if hz > 1e-6:
            half_z = hz
    bottom = cz - half_z
    top    = cz + half_z
    return half_z, bottom, top

def forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_target):
    leftF = rightF = 0.0
    hitL = hitR = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]; b2 = model.geom_bodyid[g2]
        if (b1 == bid_left and b2 == bid_target) or (b2 == bid_left and b1 == bid_target):
            f6 = np.zeros(6); mujoco.mj_contactForce(model, data, i, f6)
            leftF  += abs(f6[0]); hitL = True
        if (b1 == bid_right and b2 == bid_target) or (b2 == bid_right and b1 == bid_target):
            f6 = np.zeros(6); mujoco.mj_contactForce(model, data, i, f6)
            rightF += abs(f6[0]); hitR = True
    return leftF, rightF, hitL, hitR

def yaw_from_R(R):
    """从旋转矩阵取平面 yaw（假设 gripper-x 是朝前轴）"""
    return math.atan2(R[1,0], R[0,0])

def quat_xyzw_to_R(q):
    x, y, z, w = q
    # 归一化
    n = math.sqrt(x*x+y*y+z*z+w*w) + 1e-12
    x/=n; y/=n; z/=n; w/=n
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=float)
    return R

# -------------------- 可视化器 --------------------
class GraspVisualizer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.viewer = None
        self.is_paused = False
        
    def launch_viewer(self):
        """启动可视化器"""
        try:
            self.viewer = viewer.launch(self.model, self.data, key_callback=self._key_callback)
            print("可视化器已启动 - 按SPACE暂停/继续，按ESC退出")
        except Exception as e:
            print(f"无法启动可视化器: {e}")
            self.viewer = None
            
    def _key_callback(self, key):
        """键盘回调函数"""
        if key == 32:  # SPACE
            self.toggle_pause()
        elif key == 27:  # ESC
            self.close()
            
    def update(self):
        """更新可视化"""
        if self.viewer and not self.is_paused:
            self.viewer.sync()
            
    def toggle_pause(self):
        """切换暂停状态"""
        self.is_paused = not self.is_paused
        status = "暂停" if self.is_paused else "继续"
        print(f"仿真{status}")
        
    def close(self):
        """关闭可视化器"""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# -------------------- 全局变量 --------------------
_g = {}
_visualizer = None

# -------------------- 初始化函数 --------------------
def _worker_init(xml_path, target_geom, headless_env, enable_visualization=False):
    global _visualizer
    
    if headless_env and ("MUJOCO_GL" not in os.environ):
        os.environ["MUJOCO_GL"] = headless_env
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    _g["model"] = model
    _g["data"] = data
    _g["target_geom"] = target_geom

    # 获取各类ID
    _g["aid_x"]     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    _g["aid_y"]     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    _g["aid_lift"]  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    _g["aid_rot"]   = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")
    _g["aid_L"]     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_joint")
    _g["aid_R"]     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_joint")
    _g["jid_yaw"]   = name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    "yaw")
    _g["qadr_yaw"]  = model.jnt_qposadr[_g["jid_yaw"]]
    _g["jid_lift"]  = name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    "lift")
    _g["qadr_lift"] = model.jnt_qposadr[_g["jid_lift"]]
    _g["bid_palm"]  = name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "palm")
    _g["bid_left"]  = name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "left_link")
    _g["bid_right"] = name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "right_link")
    _g["gid_tgt"]   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM,     target_geom)
    _g["bid_tgt"]   = name2id(model, mujoco.mjtObj.mjOBJ_BODY,     target_geom.replace("_geom",""))

    # 初始化可视化器
    if enable_visualization:
        _visualizer = GraspVisualizer(model, data)
        _visualizer.launch_viewer()
        time.sleep(1)  # 给可视化器启动时间

# -------------------- 单次试验函数 --------------------
def _one_trial(args):
    idx, grasp, timing, enable_viz = args
    model = _g["model"]
    
    # 创建新的数据实例
    if enable_viz:
        data = _g["data"]
        mujoco.mj_resetData(model, data)
    else:
        data = mujoco.MjData(model)
    
    # 获取句柄
    aid_x, aid_y, aid_lift, aid_rot = _g["aid_x"], _g["aid_y"], _g["aid_lift"], _g["aid_rot"]
    aid_L, aid_R = _g["aid_L"], _g["aid_R"]
    qadr_yaw = _g["qadr_yaw"]
    bid_palm, bid_left, bid_right = _g["bid_palm"], _g["bid_left"], _g["bid_right"]
    gid_tgt, bid_tgt = _g["gid_tgt"], _g["bid_tgt"]
    loZ, hiZ = model.actuator_ctrlrange[aid_lift]

    print(f"\n{'='*60}")
    print(f"开始测试抓取 #{idx}")
    print(f"{'='*60}")

    # 初始沉降
    mujoco.mj_forward(model, data)
    wait_steps(model, data, SETTLE_STEPS)
    if enable_viz and _visualizer:
        _visualizer.update()

    # 物体世界变换
    mujoco.mj_forward(model, data)
    p_obj = data.geom_xpos[gid_tgt].copy()
    R_obj = data.geom_xmat[gid_tgt].reshape(3,3).copy()

    print(f"\n📦 物体世界位置: {p_obj}")

    # 坐标系转换：相对坐标 -> 世界坐标
    p_g = np.array(grasp["position"], dtype=float)
    R_g = quat_xyzw_to_R(grasp["quaternion_xyzw"])
    p_world = p_obj + R_obj @ p_g
    R_world = R_obj @ R_g

    print(f"🎯 相对抓取位置: {p_g}")
    print(f"🌍 世界抓取位置: {p_world}")
    print("✅ 坐标系转换完成!")

    # 期望yaw
    yaw_des = yaw_from_R(R_world)

    # 计算手指到底盘的距离
    mujoco.mj_forward(model, data)
    palm2tip = data.xpos[bid_palm][2] - body_bottom_z(model, data, bid_left)

    # 起始姿态：沿抓取z轴反向退开
    approach_dir = -R_world[:,2]
    approach_dir = approach_dir / (np.linalg.norm(approach_dir) + 1e-12)
    p_start = p_world - approach_dir * APPROACH_DIST

    print(f"\n🚀 接近阶段开始")
    print(f"📍 起始位置: {p_start}")
    print(f"🧭 接近方向: {approach_dir}")

    # 控制到起始位置
    data.ctrl[aid_L] = OPEN_CMD
    data.ctrl[aid_R] = OPEN_CMD
    data.ctrl[aid_x] = float(p_start[0])
    data.ctrl[aid_y] = float(p_start[1])
    data.ctrl[aid_rot] = float(wrap_to_pi(yaw_des))
    
    target_lift = np.clip(float(p_start[2] + palm2tip), loZ, hiZ)
    data.ctrl[aid_lift] = target_lift

    # 接近阶段
    for step in range(STEP_XY_SETTLE):
        mujoco.mj_step(model, data)
        if enable_viz and _visualizer:
            _visualizer.update()
            time.sleep(0.001)

    print("✅ 接近完成!")

    # 下压阶段
    print(f"\n⬇️ 下压阶段开始")
    _, bot, top = target_halfz_bottom_top(model, data, gid_tgt)
    z_goal = bot + DOWN_SAFE_GAP + palm2tip
    print(f"🎯 目标高度: {z_goal:.4f} (底面: {bot:.4f} + 安全间隙: {DOWN_SAFE_GAP})")

    fail_reason = None
    for step_idx in range(400):
        palm_z = float(data.xpos[bid_palm][2])
        dz = palm_z - z_goal
        
        if dz <= 0.002:
            print(f"✅ 下压完成! 最终高度: {palm_z:.4f}")
            break
            
        step_size = min(DOWN_STEP, max(0.0, dz) * 0.5)
        data.ctrl[aid_lift] = np.clip(float(data.ctrl[aid_lift]) - step_size, loZ, hiZ)
        mujoco.mj_step(model, data)
        
        if enable_viz and _visualizer:
            _visualizer.update()
            
        if step_idx % 50 == 0:
            print(f"📊 下压中... 当前高度: {palm_z:.4f}, 距目标: {dz:.4f}")

    # 闭合夹爪
    print(f"\n🤏 抓取阶段开始")
    data.ctrl[aid_L] = CLOSE_CMD
    data.ctrl[aid_R] = CLOSE_CMD
    
    for _ in range(20):
        mujoco.mj_step(model, data)
        if enable_viz and _visualizer:
            _visualizer.update()
            time.sleep(0.01)

    # 检查接触力
    lF, rF, hitL, hitR = forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_tgt)
    sumF = lF + rF
    
    print(f"📊 接触力 - 左: {lF:.3f}N, 右: {rF:.3f}N, 总和: {sumF:.3f}N")
    print(f"🔗 接触状态 - 左: {hitL}, 右: {hitR}")

    if not (hitL and hitR and sumF > BOTH_CONTACT_TH):
        fail_reason = f"❌ 抓取失败: 接触力不足 (需要 > {BOTH_CONTACT_TH}N)"
        print(fail_reason)

    # 提升验证阶段
    if fail_reason is None:
        print(f"\n⬆️ 提升验证开始")
        lift_target = max(float(data.qpos[_g["qadr_lift"]]) + 0.15, LIFT_CLEAR)
        end_time = time.time() + HOLD_TIME
        ok_hold = True
        max_drop = 0.0

        # 提升
        for step_idx in range(250):
            cur = float(data.ctrl[aid_lift])
            step_size = min(0.004, abs(lift_target - cur) * 0.5)
            data.ctrl[aid_lift] = np.clip(cur + step_size, loZ, hiZ)
            mujoco.mj_step(model, data)
            
            if enable_viz and _visualizer:
                _visualizer.update()
                
            if step_idx % 50 == 0:
                z_now = float(data.geom_xpos[gid_tgt][2])
                print(f"📈 提升中... 物体高度: {z_now:.4f}")

        # 保持监测
        z_min_keep = float(data.geom_xpos[gid_tgt][2])
        print(f"👀 开始保持监测，初始高度: {z_min_keep:.4f}")
        
        start_time = time.time()
        while time.time() - start_time < HOLD_TIME:
            mujoco.mj_step(model, data)
            z_now = float(data.geom_xpos[gid_tgt][2])
            drop = z_min_keep - z_now
            
            if drop < 0.0:
                z_min_keep = z_now
            else:
                max_drop = max(max_drop, drop)
                
            if enable_viz and _visualizer:
                _visualizer.update()
                
            # 检查接触状态
            lF, rF, hitL, hitR = forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_tgt)
            
            if not (hitL or hitR) and (z_now < 0.01):
                ok_hold = False
                fail_reason = f"❌ 物体掉落: 高度={z_now:.3f}, 失去接触"
                print(fail_reason)
                break

    success = (fail_reason is None)
    res = {
        "index": idx,
        "success": success,
        "reason": "✅ 成功" if success else fail_reason,
        "sumF": float(sumF),
        "final_z": float(data.geom_xpos[gid_tgt][2]),
        "max_drop": float(max_drop),
        "yaw_des_deg": float(np.degrees(wrap_to_pi(yaw_des))),
        "grasp_width": float(grasp.get("width", -1.0)),
    }
    
    print(f"\n{'='*60}")
    print(f"测试结果: {'✅ 成功' if success else '❌ 失败'}")
    print(f"原因: {res['reason']}")
    print(f"最终高度: {res['final_z']:.4f}m")
    print(f"最大下落: {res['max_drop']:.4f}m")
    print(f"{'='*60}")
    
    return res

# -------------------- 主函数 --------------------
def main():
    ap = argparse.ArgumentParser(description="6D抓取验证系统 - 支持可视化")
    ap.add_argument("--xml", required=True, help="MuJoCo场景XML文件")
    ap.add_argument("--target", required=True, help="目标几何体名称")
    ap.add_argument("--grasps", required=True, help="抓取姿态JSON文件")
    ap.add_argument("--processes", type=int, default=1, help="进程数（可视化时强制为1）")
    ap.add_argument("--topk", type=int, default=5, help="测试的抓取数量")
    ap.add_argument("--egl", action="store_true", help="使用EGL渲染（无头模式）")
    ap.add_argument("--visualize", action="store_true", help="启用实时可视化")
    ap.add_argument("--out", default="grasp_eval_results.json", help="输出结果文件")
    
    args = ap.parse_args()

    # 可视化模式下强制单进程
    if args.visualize:
        args.processes = 1
        print("🎥 可视化模式已启用，使用单进程")

    headless_env = "egl" if args.egl else os.environ.get("MUJOCO_GL", "")

    # 读取抓取姿态
    try:
        with open(args.grasps, "r") as f:
            G = json.load(f)
        grasps = G.get("grasps", G)
        
        if not isinstance(grasps, list):
            raise ValueError("JSON格式错误：应为数组或含'grasps'的对象")
            
        if len(grasps) == 0:
            raise ValueError("没有找到抓取姿态")
            
    except Exception as e:
        print(f"❌ 读取抓取文件失败: {e}")
        return

    # 按评分排序
    if "score" in grasps[0]:
        grasps = sorted(grasps, key=lambda g: g["score"], reverse=True)
        print(f"📊 按评分排序完成，最高分: {grasps[0]['score']:.3f}")

    if args.topk > 0:
        grasps = grasps[:args.topk]

    print(f"📋 待评测抓取数: {len(grasps)}")
    print(f"👀 可视化模式: {'✅ 启用' if args.visualize else '❌ 禁用'}")

    results = []
    
    # 单进程模式（支持可视化）
    if args.processes == 1 or args.visualize:
        _worker_init(args.xml, args.target, headless_env, args.visualize)
        
        for i in range(len(grasps)):
            try:
                res = _one_trial((i, grasps[i], None, args.visualize))
                results.append(res)
                
                if args.visualize and i < len(grasps) - 1:
                    input("\n⏎ 按Enter继续下一个抓取测试...")
                    
            except Exception as e:
                print(f"❌ 测试#{i}出错: {e}")
                traceback.print_exc()
                results.append({
                    "index": i,
                    "success": False,
                    "reason": f"执行错误: {str(e)}",
                    "sumF": 0.0,
                    "final_z": 0.0,
                    "max_drop": 0.0,
                    "yaw_des_deg": 0.0,
                    "grasp_width": 0.0
                })
    else:
        # 多进程模式
        ctx = mp.get_context("spawn")
        try:
            with ctx.Pool(processes=args.processes, initializer=_worker_init,
                          initargs=(args.xml, args.target, headless_env, False)) as pool:
                tasks = [(i, grasps[i], None, False) for i in range(len(grasps))]
                results = list(pool.imap_unordered(_one_trial, tasks, chunksize=1))
        except Exception as e:
            print(f"❌ 多进程执行失败: {e}")
            return

    # 汇总结果
    ok = [r for r in results if r["success"]]
    bad = [r for r in results if not r["success"]]
    
    summary = {
        "xml": args.xml,
        "target": args.target,
        "grasps_file": args.grasps,
        "processes": args.processes,
        "success_count": len(ok),
        "failed_count": len(bad),
        "total_count": len(results),
        "success_rate": len(ok) / len(results) if len(results) > 0 else 0,
        "results": sorted(results, key=lambda r: r["index"])
    }
    
    # 保存结果
    try:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")

    # 打印最终统计
    print(f"\n{'='*60}")
    print(f"🎯 最终测试结果")
    print(f"{'='*60}")
    print(f"✅ 成功: {len(ok)}")
    print(f"❌ 失败: {len(bad)}")
    print(f"📊 总计: {len(results)}")
    print(f"🏆 成功率: {summary['success_rate']*100:.1f}%")
    print(f"💾 结果已保存: {args.out}")
    print(f"{'='*60}")

    # 关闭可视化器
    if _visualizer:
        _visualizer.close()

if __name__ == "__main__":
    main()