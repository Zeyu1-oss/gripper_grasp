import mujoco
import mujoco.viewer
import numpy as np

def main():
    # 加载模型
    model = mujoco.MjModel.from_xml_path("../g2_eval.xml")  # 替换为你的XML文件路径
    data = mujoco.MjData(model)

    # 获取左右夹爪电机的ID
    left_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_joint")
    right_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_joint")

    # 初始化控制信号（例如，初始状态为张开）
    data.ctrl[left_motor_id] = -1.0  # 张开左夹爪
    data.ctrl[right_motor_id] = -1.0  # 张开右夹爪

    # 打开查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("控制说明:")
        print("  - 按 'c' 键闭合夹爪")
        print("  - 按 'o' 键张开夹爪")
        print("  - 关闭查看器窗口退出")

        # 保持查看器开启
        while viewer.is_running():
            # 这里可以添加更复杂的控制逻辑，例如通过键盘输入
            # 例如，使用 viewer.user_action 来检测键盘事件（具体取决于查看器的实现和版本）
            # 以下是一个简单的示例，实际实现可能需要根据查看器版本调整
            mujoco.mj_step(model, data)
            viewer.sync()

            # 示例：简单的键盘控制（可能需要根据你的查看器版本调整）
            # 更复杂的交互可能需要使用其他库（如glfw）来直接捕获键盘事件
            # 或者考虑使用图形界面滑块控制

if __name__ == "__main__":
    main()