import numpy as np
import mujoco
from mujoco import mjtObj

def compute_grasp_yaw(model, data, gid_target):
    """yaw角度"""
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]   # 半边长 (x,y,z)

    axis = np.argmax(size)  # 0=x, 1=y, 2=z
    long_dir_world = R[:, axis]

    dir_xy = np.array([long_dir_world[0], long_dir_world[1]])
    if np.linalg.norm(dir_xy) < 1e-6:
        return 0.0
    dir_xy /= np.linalg.norm(dir_xy)

    angle = np.arctan2(dir_xy[1], dir_xy[0])
    return angle - np.pi/2

def name2id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return None

def test_lego_rotation():
    # 加载模型
    model = mujoco.MjModel.from_xml_path("g2_with_30lego.xml")
    data = mujoco.MjData(model)
    
    # 获取乐高积木的几何体ID
    gid_lego = name2id(model, mjtObj.mjOBJ_GEOM, "lego_07_geom")
    if gid_lego is None:
        print("错误：找不到乐高积木几何体")
        return
    
    print("=== 乐高积木旋转矩阵测试 ===\n")
    
    # 测试不同旋转角度
    test_angles = [0, 30, 45, 60, 90, 120, 180, 270]  # 度
    
    for angle_deg in test_angles:
        # 设置乐高积木的旋转（绕Z轴）
        angle_rad = np.radians(angle_deg)
        
        # 找到乐高积木的body ID
        bid_lego = name2id(model, mjtObj.mjOBJ_BODY, "lego_07")
        if bid_lego is not None:
            # 设置四元数旋转（绕Z轴）
            data.qpos[model.jnt_qposadr[model.body_jntadr[bid_lego][0]] + 3:7] = [
                np.cos(angle_rad/2), 0, 0, np.sin(angle_rad/2)
            ]
        
        # 前向动力学计算
        mujoco.mj_forward(model, data)
        
        # 获取旋转矩阵
        R = data.geom_xmat[gid_lego].reshape(3, 3)
        
        # 获取尺寸
        size = model.geom_size[gid_lego]
        
        # 计算抓取角度
        grasp_yaw = compute_grasp_yaw(model, data, gid_lego)
        
        # 打印结果
        print(f"设置角度: {angle_deg:3d}°")
        print(f"旋转矩阵 R:")
        print(f"  [{R[0,0]:6.3f} {R[0,1]:6.3f} {R[0,2]:6.3f}]")
        print(f"  [{R[1,0]:6.3f} {R[1,1]:6.3f} {R[1,2]:6.3f}]")
        print(f"  [{R[2,0]:6.3f} {R[2,1]:6.3f} {R[2,2]:6.3f}]")
        print(f"尺寸: [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}]")
        print(f"最长轴: {np.argmax(size)} (0=X, 1=Y, 2=Z)")
        print(f"计算抓取角度: {np.degrees(grasp_yaw):6.1f}°")
        print("-" * 50)
        
        # 验证：计算出的抓取角度应该与设置角度垂直（±90度）
        expected_grasp = (angle_deg - 90) % 360
        if expected_grasp > 180:
            expected_grasp -= 360
        
        calculated_grasp = np.degrees(grasp_yaw) % 360
        if calculated_grasp > 180:
            calculated_grasp -= 360
        
        print(f"预期抓取角度: {expected_grasp:6.1f}°")
        print(f"实际抓取角度: {calculated_grasp:6.1f}°")
        print(f"误差: {abs(expected_grasp - calculated_grasp):6.1f}°")
        
        # 检查最长轴方向
        axis = np.argmax(size)
        long_dir_world = R[:, axis]
        dir_xy = np.array([long_dir_world[0], long_dir_world[1]])
        dir_xy /= np.linalg.norm(dir_xy)
        
        print(f"最长轴世界方向: [{dir_xy[0]:6.3f}, {dir_xy[1]:6.3f}]")
        print("=" * 50 + "\n")

def test_specific_case():
    """测试特定情况"""
    print("=== 特定情况测试 ===")
    
    model = mujoco.MjModel.from_xml_path("g2_with_30lego.xml")
    data = mujoco.MjData(model)
    
    gid_lego = name2id(model, mjtObj.mjOBJ_GEOM, "lego_07_geom")
    
    # 测试垂直情况（最长轴朝上）
    bid_lego = name2id(model, mjtObj.mjOBJ_BODY, "lego_07")
    if bid_lego is not None:
        # 设置乐高竖立（绕X轴旋转90度）
        data.qpos[model.jnt_qposadr[model.body_jntadr[bid_lego][0]] + 3:7] = [
            0.7071, 0.7071, 0, 0  # 绕X轴旋转90度
        ]
    
    mujoco.mj_forward(model, data)
    
    R = data.geom_xmat[gid_lego].reshape(3, 3)
    size = model.geom_size[gid_lego]
    
    print("竖立状态测试:")
    print(f"尺寸: [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}]")
    print(f"最长轴: {np.argmax(size)}")
    
    grasp_yaw = compute_grasp_yaw(model, data, gid_lego)
    print(f"抓取角度: {np.degrees(grasp_yaw):.1f}° (应该接近0°)")

if __name__ == "__main__":
    test_lego_rotation()
    test_specific_case()