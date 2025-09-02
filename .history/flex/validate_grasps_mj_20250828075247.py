#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, math, argparse, traceback
import numpy as np
import multiprocessing as mp
import os , sys 
if ('--egl' in sys.argv) and ('MUJOCOGL' not in os.environ):
    os.environ['MUJOCO_GL']='egl'
import mujoco
from mujoco import viewer

# ... [之前的常量定义保持不变] ...

# -------------------- 可视化工具 --------------------
class GraspVisualizer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.viewer = None
        self.is_paused = False
        self.step_count = 0
        
    def launch_viewer(self):
        """启动可视化器"""
        try:
            self.viewer = viewer.launch(self.model, self.data)
            print("可视化器已启动 - 按SPACE暂停/继续，按ESC退出")
        except Exception as e:
            print(f"无法启动可视化器: {e}")
            self.viewer = None
            
    def update(self):
        """更新可视化"""
        if self.viewer and not self.is_paused:
            self.viewer.sync()
            self.step_count += 1
            
    def toggle_pause(self):
        """切换暂停状态"""
        self.is_paused = not self.is_paused
        status = "暂停" if self.is_paused else "继续"
        print(f"仿真{status}")
        
    def close(self):
        """关闭可视化器"""
        if self.viewer:
            self.viewer.close()

# -------------------- 修改后的 worker 初始化 --------------------
_g = {}
_visualizer = None

def _worker_init(xml_path, target_geom, headless_env, enable_visualization=False):
    global _visualizer
    
    if headless_env and ("MUJOCO_GL" not in os.environ):
        os.environ["MUJOCO_GL"] = headless_env
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    _g["model"] = model
    _g["data"] = data
    _g["target_geom"] = target_geom

    # 初始化可视化器
    if enable_visualization:
        _visualizer = GraspVisualizer(model, data)
        _visualizer.launch_viewer()

    # ... [之前的ID获取代码保持不变] ...

# -------------------- 修改后的单次试验函数 --------------------
def _one_trial(args):
    idx, grasp, timing, enable_viz = args
    model = _g["model"]
    data = _g["data"].copy() if enable_viz else mujoco.MjData(model)
    
    # 重置数据
    mujoco.mj_resetData(model, data)
    
    # 便捷句柄
    # ... [之前的句柄代码保持不变] ...

    # 初始沉降
    mujoco.mj_forward(model, data)
    wait_steps(model, data, SETTLE_STEPS)
    
    print(f"\n=== 开始测试抓取 #{idx} ===")
    print(f"位置: {grasp['position']}")
    print(f"旋转: {grasp['quaternion_xyzw']}")
    print(f"宽度: {grasp.get('width', 'N/A')}")

    # 物体世界变换
    mujoco.mj_forward(model, data)
    p_obj = data.geom_xpos[gid_tgt].copy()
    R_obj = data.geom_xmat[gid_tgt].reshape(3,3).copy()

    # 坐标系转换演示
    print("\n--- 坐标系转换 ---")
    print(f"物体世界位置: {p_obj}")
    print(f"相对抓取位置: {grasp['position']}")
    
    # 将 grasp 从"网格坐标" -> 世界坐标
    p_g = np.array(grasp["position"], dtype=float)
    R_g = quat_xyzw_to_R(grasp["quaternion_xyzw"])
    p_world = p_obj + R_obj @ p_g
    R_world = R_obj @ R_g
    
    print(f"最终世界位置: {p_world}")
    print("转换完成!")

    # 期望 yaw（绕世界Z）
    yaw_des = yaw_from_R(R_world)

    # 计算手指顶到底盘的差
    mujoco.mj_forward(model, data)
    palm2tip = data.xpos[bid_palm][2] - body_bottom_z(model, data, bid_left)

    # 起始姿态
    approach_dir = -R_world[:,2]
    approach_dir = approach_dir / (np.linalg.norm(approach_dir)+1e-12)
    p_start = p_world - approach_dir * APPROACH_DIST

    print(f"\n--- 接近阶段 ---")
    print(f"起始位置: {p_start}")
    print(f"接近方向: {approach_dir}")

    # 控制到起始位置
    data.ctrl[aid_L] = OPEN_CMD
    data.ctrl[aid_R] = OPEN_CMD
    data.ctrl[aid_x] = float(p_start[0])
    data.ctrl[aid_y] = float(p_start[1])
    data.ctrl[aid_rot] = float(wrap_to_pi(yaw_des))
    
    target_lift = np.clip(float(p_start[2] + palm2tip), loZ, hiZ)
    data.ctrl[aid_lift] = target_lift
    
    # 接近阶段的可视化
    for step in range(STEP_XY_SETTLE):
        mujoco.mj_step(model, data)
        if enable_viz and _visualizer:
            _visualizer.update()
            time.sleep(0.001)

    print("接近完成!")

    # 下压阶段
    print(f"\n--- 下压阶段 ---")
    _, bot, top = target_halfz_bottom_top(model, data, gid_tgt)
    z_goal = bot + DOWN_SAFE_GAP + palm2tip
    print(f"目标高度: {z_goal:.4f} (底面: {bot:.4f} + 安全间隙: {DOWN_SAFE_GAP})")

    fail_reason = None
    for step_idx in range(400):
        palm_z = float(data.xpos[bid_palm][2])
        dz = palm_z - z_goal
        
        if dz <= 0.002:
            print(f"下压完成! 最终高度: {palm_z:.4f}")
            break
            
        step_size = min(DOWN_STEP, max(0.0, dz)*0.5)
        data.ctrl[aid_lift] = np.clip(float(data.ctrl[aid_lift]) - step_size, loZ, hiZ)
        mujoco.mj_step(model, data)
        
        if enable_viz and _visualizer:
            _visualizer.update()
            if step_idx % 50 == 0:
                print(f"下压中... 当前高度: {palm_z:.4f}, 目标差: {dz:.4f}")

    # 闭合夹爪
    print(f"\n--- 抓取阶段 ---")
    data.ctrl[aid_L] = CLOSE_CMD
    data.ctrl[aid_R] = CLOSE_CMD
    
    for _ in range(10):
        mujoco.mj_step(model, data)
        if enable_viz and _visualizer:
            _visualizer.update()
            time.sleep(0.01)

    # 检查接触力
    lF, rF, hitL, hitR = forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_tgt)
    sumF = lF + rF
    print(f"接触力 - 左: {lF:.3f}N, 右: {rF:.3f}N, 总和: {sumF:.3f}N")
    print(f"接触状态 - 左: {hitL}, 右: {hitR}")

    if not (hitL and hitR and sumF > BOTH_CONTACT_TH):
        fail_reason = f"抓取失败: 接触力不足 (需要 > {BOTH_CONTACT_TH}N)"
        print(fail_reason)

    # 提升验证阶段
    if fail_reason is None:
        print(f"\n--- 提升验证 ---")
        lift_target = max(float(data.qpos[_g["qadr_lift"]]) + 0.15, LIFT_CLEAR)
        end_time = time.time() + HOLD_TIME
        ok_hold = True
        max_drop = 0.0

        # 提升
        for step_idx in range(250):
            cur = float(data.ctrl[aid_lift])
            step_size = min(0.004, abs(lift_target-cur)*0.5)
            data.ctrl[aid_lift] = np.clip(cur + step_size, loZ, hiZ)
            mujoco.mj_step(model, data)
            
            if enable_viz and _visualizer:
                _visualizer.update()
                if step_idx % 50 == 0:
                    z_now = float(data.geom_xpos[gid_tgt][2])
                    print(f"提升中... 物体高度: {z_now:.4f}")

        # 保持监测
        z_min_keep = float(data.geom_xpos[gid_tgt][2])
        print(f"开始保持监测，初始高度: {z_min_keep:.4f}")
        
        while time.time() < end_time:
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
                fail_reason = f"物体掉落: 高度={z_now:.3f}, 失去接触"
                print(fail_reason)
                break

    success = (fail_reason is None)
    res = {
        "index": idx,
        "success": success,
        "reason": "成功" if success else fail_reason,
        "sumF": sumF,
        "final_z": float(data.geom_xpos[gid_tgt][2]),
        "max_drop": float(max_drop),
        "yaw_des_deg": float(np.degrees(wrap_to_pi(yaw_des))),
        "grasp_width": float(grasp.get("width", -1.0)),
    }
    
    print(f"\n=== 测试结果: {'成功' if success else '失败'} ===")
    print(f"原因: {res['reason']}")
    print(f"最终高度: {res['final_z']:.4f}")
    print("="*50)
    
    return res

# -------------------- 修改后的主函数 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="MuJoCo 场景 XML含夹爪+LEGO")
    ap.add_argument("--target", required=True, help="目标几何名（如 lego_23_geom")
    ap.add_argument("--grasps", required=True, help="6D 抓取 JSON数组或 {grasps:[]}")
    ap.add_argument("--processes", type=int, default=1, help="进程数，可视化时建议设为1")
    ap.add_argument("--topk", type=int, default=1, help="测试的抓取数量")
    ap.add_argument("--egl", action="store_true", help="设置 MUJOCO_GL=egl")
    ap.add_argument("--visualize", action="store_true", help="启用可视化")
    ap.add_argument("--out", default="grasp_eval_results.json")
    args = ap.parse_args()

    # 可视化模式下强制单进程
    if args.visualize:
        args.processes = 1
        print("可视化模式已启用，使用单进程")

    headless_env = "egl" if args.egl else os.environ.get("MUJOCO_GL", "")

    # 读取 grasps
    with open(args.grasps, "r") as f:
        G = json.load(f)
    grasps = G.get("grasps", G)
    
    if not isinstance(grasps, list):
        raise ValueError("JSON 格式错误：应为数组或含 'grasps' 的对象")

    if len(grasps) > 0 and "score" in grasps[0]:
        grasps = sorted(grasps, key=lambda g: g["score"], reverse=True)

    if args.topk and args.topk > 0:
        grasps = grasps[:args.topk]

    print(f"[INFO] 待评测抓取数: {len(grasps)}")
    print(f"[INFO] 可视化模式: {'启用' if args.visualize else '禁用'}")

    # 单进程模式（支持可视化）
    if args.processes == 1 or args.visualize:
        _worker_init(args.xml, args.target, headless_env, args.visualize)
        results = []
        for i in range(len(grasps)):
            res = _one_trial((i, grasps[i], None, args.visualize))
            results.append(res)
            
            if _visualizer:
                input("按Enter继续下一个抓取测试...")
    else:
        # 多进程模式
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.processes, initializer=_worker_init,
                      initargs=(args.xml, args.target, headless_env, False)) as pool:
            tasks = [(i, grasps[i], None, False) for i in range(len(grasps))]
            results = list(pool.imap_unordered(_one_trial, tasks, chunksize=1))

    # 汇总结果
    ok = [r for r in results if r["success"]]
    bad = [r for r in results if not r["success"]]
    
    summary = {
        "xml": args.xml,
        "target": args.target,
        "grasps_file": args.grasps,
        "processes": args.processes,
        "success": len(ok),
        "failed": len(bad),
        "total": len(results),
        "results": sorted(results, key=lambda r: r["index"])
    }
    
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== 最终结果 ===")
    print(f"成功: {len(ok)}/{len(results)}")
    print(f"成功率: {len(ok)/len(results)*100:.1f}%")
    print(f"结果已保存到: {args.out}")

    # 关闭可视化器
    if _visualizer:
        _visualizer.close()

if __name__ == "__main__":
    main()