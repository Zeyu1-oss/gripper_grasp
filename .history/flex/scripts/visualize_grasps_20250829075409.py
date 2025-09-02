import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

def mat2eulerZYX(Rmat):
    # 确保Rmat为正交矩阵
    U, _, Vt = np.linalg.svd(Rmat)
    Rmat = U @ Vt
    if np.linalg.det(Rmat) < 0:
        Rmat *= -1
    if not np.all(np.isfinite(Rmat)):
        raise ValueError("Rmat contains NaN or inf!")
    return R.from_matrix(Rmat).as_euler('zyx', degrees=False)[::-1]  # 返回 roll, pitch, yaw

def set_gripper_6d(data, center, Rmat):
    # 位置
    data.qpos[0:3] = center
    # 姿态（roll, pitch, yaw）
    roll, pitch, yaw = mat2eulerZYX(Rmat)
    data.qpos[3] = roll
    data.qpos[4] = pitch
    data.qpos[5] = yaw

def set_gripper_opening(data, opening=0.02):
    # 左右指对称
    data.qpos[6] = opening / 2
    data.qpos[7] = opening / 2

def set_gravity(model, enable=True):
    model.opt.gravity[:] = [0, 0, -9.81] if enable else [0, 0, 0]

def get_lego_pos(data, lego_site_id):
    return data.site_xpos[lego_site_id].copy()

if __name__ == "__main__":
    xml_path = "../g2_eval.xml"
    pose_path = "../results/6d/grasp_poses.npy"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    poses = np.load(pose_path, allow_pickle=True)
    pose = poses[3]  # 使用第4个抓取姿态
    
    print("===== 6D 抓取姿态信息 =====")
    print(f"Grasp Center (xyz): {pose['center']}")
    print("Grasp Rotation Matrix (3x3):")
    print(pose['rotation'])

    rpy = mat2eulerZYX(pose['rotation'])
    print(f"Grasp Orientation (roll, pitch, yaw): {rpy}")

    # 初始化
    set_gravity(model, enable=False)
    mujoco.mj_resetData(model, data)
    
    # 设置夹爪初始状态
    try:
        set_gripper_6d(data, pose['center'], pose['rotation'])
    except Exception as e:
        print("旋转矩阵异常，跳过该姿态:", e)
        exit(1)
    set_gripper_opening(data, opening=0.03)
    mujoco.mj_forward(model, data)

    lego_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lego_center")
    if lego_site_id < 0:
        raise RuntimeError("未找到lego_center site，请检查xml文件！")
    
    lego_pos_before = get_lego_pos(data, lego_site_id)
    
    # 状态变量
    gravity_enabled = False
    simulation_finished = False
    grab_started = False
    
    print("\n=== 控制说明 ===")
    print("空格键: 开启/关闭重力")
    print("G键: 开始抓取（闭合夹爪）")
    print("ESC键: 退出仿真")
    print("================\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置初始视角（可选）
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30
        
        while viewer.is_running() and not simulation_finished:
            # 处理键盘输入
            if viewer.is_key_down(mujoco.mjtKey.mjKEY_SPACE):
                gravity_enabled = not gravity_enabled
                set_gravity(model, gravity_enabled)
                print(f"重力 {'开启' if gravity_enabled else '关闭'}")
                time.sleep(0.2)  # 防抖
            
            if viewer.is_key_down(mujoco.mjtKey.mjKEY_G) and not grab_started:
                print("开始抓取：闭合夹爪")
                set_gripper_opening(data, opening=0.0)
                grab_started = True
                time.sleep(0.2)
            
            if viewer.is_key_down(mujoco.mjtKey.mjKEY_ESC):
                print("用户退出仿真")
                break
            
            # 模拟一步
            mujoco.mj_step(model, data)
            
            # 更新可视化
            viewer.sync()
            
            # 检查抓取结果（只有在重力开启且抓取开始后）
            if gravity_enabled and grab_started:
                lego_pos_current = get_lego_pos(data, lego_site_id)
                displacement = np.linalg.norm(lego_pos_current - lego_pos_before)
                
                # 如果位移超过阈值，判定为失败
                if displacement > 0.1:  # 10cm位移阈值
                    print("抓取失败: LEGO掉落！")
                    simulation_finished = True
                
                # 模拟2秒后结束
                if data.time > 2.0 and not simulation_finished:
                    lego_pos_after = get_lego_pos(data, lego_site_id)
                    final_displacement = np.linalg.norm(lego_pos_after - lego_pos_before)
                    print(f"仿真结束，LEGO最终位移: {final_displacement:.4f} 米")
                    
                    if final_displacement < 0.005:
                        print("抓取成功: LEGO未掉落！")
                    else:
                        print("抓取失败: LEGO移动过多")
                    
                    simulation_finished = True
            
            # 控制仿真速度
            time.sleep(0.001)

    # 最终结果统计
    if simulation_finished:
        lego_pos_final = get_lego_pos(data, lego_site_id)
        final_displacement = np.linalg.norm(lego_pos_final - lego_pos_before)
        print(f"\n=== 最终结果 ===")
        print(f"LEGO位移: {final_displacement:.4f} 米")
        print(f"仿真时间: {data.time:.2f} 秒")
        
        if final_displacement < 0.005:
            print("✅ 抓取成功: LEGO稳定被抓取")
        else:
            print("❌ 抓取失败: LEGO掉落或移动过多")