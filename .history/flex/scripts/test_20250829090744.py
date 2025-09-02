import mujoco
import mujoco.viewer
import numpy as np
import time

def load_and_visualize_components():
    """
    加载并可视化抓取器和LEGO组件
    """
    
    # 方法1: 分别加载两个组件
    try:
        print("加载抓取器模型...")
        gripper_model = mujoco.MjModel.from_xml_path("../mjcf/g2_gripper.xml")
        gripper_data = mujoco.MjData(gripper_model)
        
        print("加载LEGO模型...")
        lego_model = mujoco.MjModel.from_xml_path("../mjcgf/lego.xml")
        lego_data = mujoco.MjData(lego_model)
        
        # 可视化抓取器
        print("可视化抓取器...")
        with mujoco.viewer.launch_passive(gripper_model, gripper_data) as viewer:
            # 运行仿真
            for _ in range(1000):
                mujoco.mj_step(gripper_model, gripper_data)
                viewer.sync()
                time.sleep(0.001)
        
        # 可视化LEGO
        print("可视化LEGO...")
        with mujoco.viewer.launch_passive(lego_model, lego_data) as viewer:
            # 运行仿真
            for _ in range(1000):
                mujoco.mj_step(lego_model, lego_data)
                viewer.sync()
                time.sleep(0.001)
                
    except Exception as e:
        print(f"加载独立组件失败: {e}")
    
    # 方法2: 加载组合场景
    try:
        print("\n加载组合场景...")
        combined_model = mujoco.MjModel.from_xml_path("../mjcf/combined_scene.xml")
        combined_data = mujoco.MjData(combined_model)
        
        print("可视化组合场景...")
        with mujoco.viewer.launch_passive(combined_model, combined_data) as viewer:
            # 运行仿真
            for step in range(5000):
                # 简单的控制示例
                if step < 1000:
                    # 移动抓取器到LEGO上方
                    combined_data.ctrl[0] = 0.0  # x
                    combined_data.ctrl[1] = 0.0  # y
                    combined_data.ctrl[2] = 0.1  # z
                elif step < 2000:
                    # 下降
                    combined_data.ctrl[2] = 0.05
                elif step < 3000:
                    # 抓取
                    combined_data.ctrl[6] = 0.5  # 夹爪
                else:
                    # 抬起
                    combined_data.ctrl[2] = 0.15
                
                mujoco.mj_step(combined_model, combined_data)
                viewer.sync()
                time.sleep(0.001)
                
    except Exception as e:
        print(f"加载组合场景失败: {e}")

def print_model_info():
    """
    打印模型信息
    """
    try:
        gripper_model = mujoco.MjModel.from_xml_path("gripper.xml")
        print(f"抓取器模型:")
        print(f"  - 关节数量: {gripper_model.njnt}")
        print(f"  - 执行器数量: {gripper_model.nu}")
        print(f"  - 传感器数量: {gripper_model.nsensor}")
        print(f"  - 几何体数量: {gripper_model.ngeom}")
        
        lego_model = mujoco.MjModel.from_xml_path("lego.xml")
        print(f"\nLEGO模型:")
        print(f"  - 关节数量: {lego_model.njnt}")
        print(f"  - 执行器数量: {lego_model.nu}")
        print(f"  - 传感器数量: {lego_model.nsensor}")
        print(f"  - 几何体数量: {lego_model.ngeom}")
        
    except Exception as e:
        print(f"获取模型信息失败: {e}")

def interactive_control():
    """
    交互式控制示例
    """
    try:
        model = mujoco.MjModel.from_xml_path("combined_scene.xml")
        data = mujoco.MjData(model)
        
        print("交互式控制 - 使用键盘控制抓取器:")
        print("W/S: 前后移动")
        print("A/D: 左右移动")
        print("Q/E: 上下移动")
        print("空格: 开合夹爪")
        print("ESC: 退出")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.001)
                
    except Exception as e:
        print(f"交互式控制失败: {e}")

if __name__ == "__main__":
    print("MuJoCo 组件可视化脚本")
    print("=" * 50)
    
    # 打印模型信息
    print_model_info()
    
    print("\n开始可视化...")
    load_and_visualize_components()
    
    # 可选：运行交互式控制
    # interactive_control()