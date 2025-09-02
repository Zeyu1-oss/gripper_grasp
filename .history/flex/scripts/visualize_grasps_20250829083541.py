import os
import numpy as np
import mujoco
import mujoco.viewer
import time

class GraspTestEnv:
    def __init__(self, hand_xml_path, lego_mesh_path, lego_scale=0.01):
        # 设置环境变量
        os.environ["MUJOCO_GL"] = "glfw"
        
        # 创建MJCF规范
        self.spec = mujoco.MjSpec()
        self.spec.option.timestep = 0.004
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        
        # 添加灯光和相机
        self.spec.worldbody.add_light(
            name="light", pos=[0, 0, 3], dir=[0, 0, -1], directional=True
        )
        self.spec.worldbody.add_camera(
            name="main_camera", pos=[1.0, 1.0, 1.0], xyaxes=[-1, 0, 0, 0, -1, 1]
        )
        
        # 添加地板
        self.spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0, 0, 0],
            size=[2, 2, 0.1],
            rgba=[0.8, 0.8, 0.8, 1]
        )
        
        # 添加夹爪
        self._add_hand(hand_xml_path)
        
        # 添加LEGO物体
        self._add_lego(lego_mesh_path, lego_scale)
        
        # 编译模型
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        
        # 初始化
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # 状态变量
        self.gravity_enabled = False
        self.grasp_started = False
        
    def _add_hand(self, xml_path):
        """添加夹爪模型"""
        hand_spec = mujoco.MjSpec.from_file(xml_path)
        
        # 设置夹爪几何体属性
        for geom in hand_spec.geoms:
            geom.rgba = [0.7, 0.7, 0.9, 1.0]  # 设置夹爪颜色
            geom.friction = [1.0, 0.5, 0.1]   # 设置摩擦系数
        
        # 将夹爪添加到主世界
        hand_body = self.spec.worldbody.attach_body(hand_spec.worldbody, "hand_", "")
        
        # 添加固定关节（夹爪位置固定）
        hand_body.add_joint(
            name="hand_fixed",
            type=mujoco.mjtJoint.mjJNT_FREE,
            limited=False
        )
        
        # 设置夹爪初始位置
        hand_body.pos = [0, 0, 0.2]
        
    def _add_lego(self, mesh_path, scale):
        """添加LEGO物体"""
        # 添加LEGO网格
        mesh_name = "lego_mesh"
        self.spec.add_mesh(
            name=mesh_name,
            file=mesh_path,
            scale=[scale, scale, scale]
        )
        
        # 创建LEGO物体
        lego_body = self.spec.worldbody.add_body(name="lego")
        
        # 添加自由关节（LEGO可以自由运动）
        lego_body.add_freejoint(name="lego_freejoint")
        
        # 添加视觉几何体
        lego_body.add_geom(
            name="lego_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            rgba=[0.2, 0.7, 0.2, 1.0],  # LEGO颜色
            contype=0,  # 无碰撞
            conaffinity=0
        )
        
        # 添加碰撞几何体
        lego_body.add_geom(
            name="lego_collision",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            rgba=[0.2, 0.7, 0.2, 0.5],  # 半透明
            density=1000,  # 密度
            friction=[0.9, 0.1, 0.01]   # 摩擦系数
        )
        
        # 设置LEGO初始位置（在夹爪中间）
        lego_body.pos = [0, 0, 0.1]
        
    def set_gravity(self, enabled):
        """设置重力状态"""
        self.gravity_enabled = enabled
        if enabled:
            self.model.opt.gravity[:] = [0, 0, -9.81]
            print("重力已开启")
        else:
            self.model.opt.gravity[:] = [0, 0, 0]
            print("重力已关闭")
            
    def set_gripper_opening(self, opening):
        """设置夹爪开合度"""
        # 假设夹爪有两个对称的滑动关节
        if self.model.nv >= 2:
            self.data.qpos[-2] = opening / 2  # 左指
            self.data.qpos[-1] = opening / 2  # 右指
            
    def get_lego_position(self):
        """获取LEGO位置"""
        lego_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "lego")
        if lego_body_id >= 0:
            return self.data.xpos[lego_body_id].copy()
        return None
        
    def run_grasp_test(self):
        """运行抓取测试"""
        print("=== 抓取测试开始 ===")
        print("控制说明:")
        print("空格键: 切换重力")
        print("G键: 开始抓取（闭合夹爪）")
        print("ESC键: 退出")
        
        # 初始状态：无重力，夹爪张开
        self.set_gravity(False)
        self.set_gripper_opening(0.03)  # 张开3cm
        mujoco.mj_forward(self.model, self.data)
        
        # 记录LEGO初始位置
        lego_pos_initial = self.get_lego_position()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            
            while viewer.is_running():
                # 处理键盘输入（简化版）
                # 在实际应用中，你需要使用key_callback来处理键盘输入
                
                # 模拟步骤
                mujoco.mj_step(self.model, self.data)
                
                # 更新可视化
                viewer.sync()
                
                # 检查抓取状态
                if self.gravity_enabled and self.grasp_started:
                    lego_pos_current = self.get_lego_position()
                    if lego_pos_current is not None and lego_pos_initial is not None:
                        displacement = np.linalg.norm(lego_pos_current - lego_pos_initial)
                        
                        # 如果位移过大，判定为失败
                        if displacement > 0.1:
                            print("❌ 抓取失败: LEGO掉落！")
                            break
                        
                        # 模拟2秒后结束
                        if time.time() - start_time > 2.0:
                            if displacement < 0.005:
                                print("✅ 抓取成功: LEGO稳定被抓取")
                            else:
                                print("❌ 抓取失败: LEGO移动过多")
                            break
                
                time.sleep(0.001)
                
        print("=== 抓取测试结束 ===")

# 使用示例
if __name__ == "__main__":
    # 文件路径 - 需要根据你的实际路径修改
    hand_xml_path = "../../assets/hand/shadow/customized.xml"  # 夹爪XML文件
    lego_mesh_path = "../lego.obj"  # LEGO网格文件
    
    # 创建测试环境
    env = GraspTestEnv(hand_xml_path, lego_mesh_path)
    
    # 运行测试
    try:
        # 第一阶段：无重力，闭合夹爪
        print("第一阶段：闭合夹爪（无重力）")
        env.set_gripper_opening(0.0)  # 完全闭合
        env.grasp_started = True
        time.sleep(1.0)  # 等待夹爪闭合
        
        # 第二阶段：开启重力，测试抓取稳定性
        print("第二阶段：开启重力，测试抓取")
        env.set_gravity(True)
        
        # 运行抓取测试
        env.run_grasp_test()
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    finally:
        print("测试完成")
# import mujoco
# import mujoco.viewer
# import numpy as np
# from scipy.spatial.transform import Rotation as R

# def mat2eulerZYX(Rmat):
#     # 确保Rmat为正交矩阵
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
#     pose = poses[3]  # 验证第一个抓取
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