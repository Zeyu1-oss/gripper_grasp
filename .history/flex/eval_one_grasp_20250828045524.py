<mujoco model="g2_eval">
  <!-- 稳定的物理设置 -->
  <option timestep="0.001" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81"/>
  <compiler angle="radian" eulerseq="zyx" autolimits="true"/>

  <!-- 顶层 default；需要的类用嵌套 default 定义 -->
  <default>
    <joint  limited="true" damping="0.01" armature="0.01" frictionloss="0.001"/>
    <geom   condim="4" contype="1" conaffinity="15" friction="0.7 0.02 0.001" solref="0.004 1"/>
    <motor  ctrllimited="true"/>
    <equality solref="0.002 1"/>

    <!-- 视觉-only 几何（不参与碰撞） -->
    <default class="visualgeom">
      <geom material="visualgeom" condim="1" contype="0" conaffinity="0"/>
    </default>

    <!-- 给 LEGO 略软的接触，减少颤动 -->
    <default class="lego">
      <geom solimp="0.98 0.999 0.0001" solref="0.004 1" friction="0.8 0.02 0.001"/>
    </default>
  </default>

  <asset>
    <mesh name="base_link"  file="meshes/base_link.STL"/>
    <mesh name="left_link"  file="meshes/left_link.STL"/>
    <mesh name="right_link" file="meshes/right_link.STL"/>

    <!-- lego.obj 以 1cm 为单位缩放；若你的 obj 是 1m 尺度，这里 0.01 正好变 1cm -->
    <mesh name="lego" file="meshes/lego.obj" scale="0.01 0.01 0.01"/>

    <material name="visualgeom" rgba="0.5 0.9 0.2 1"/>
  </asset>

  <worldbody>
    <!-- 地面 -->
    <geom name="ground" type="plane" size="2 2 0.02" rgba="0.8 0.8 0.8 1" contype="1" conaffinity="1"/>

    <!-- 驱动 6D 位姿的 mocap 目标（只在 Python 里写 mocap_pos/quat） -->
    <body name="tcp_target" mocap="true" pos="0 0 0.5">
      <geom type="sphere" size="0.005" rgba="0.2 0.6 1 0.15"/>  <!-- 可视化点，非必需 -->
    </body>

    <!-- G2 夹爪（freejoint 允许被 mocap+weld 带着走） -->
    <body name="g2" pos="0 0 0.5">
      <freejoint name="g2_free"/>

      <site name="tcp" size="0.003" pos="0 0 0.0635"/> <!-- 简单 TCP 参考点，可按需要改 -->
      <geom type="mesh" mesh="base_link" rgba="0.8 0.8 0.8 1" class="visualgeom"/>
      <geom type="mesh" mesh="base_link"/>  <!-- 碰撞体 -->

      <body name="left_link" pos="0 0 0.0635" quat="0.707105 0 -0.707108 0">
        <joint name="left_joint" pos="0 0 0" axis="0 0 1" type="slide" range="0 0.0166"/>
        <geom type="mesh" mesh="left_link"  rgba="0.85 0.85 0.85 1"/>
      </body>

      <body name="right_link" pos="0 0 0.0635" quat="-2.59734e-06 0.707105 2.59735e-06 0.707108">
        <joint name="right_joint" pos="0 0 0" axis="0 0 1" type="slide" range="0 0.0166"/>
        <geom type="mesh" mesh="right_link" rgba="0.85 0.85 0.85 1"/>
      </body>
    </body>

    <!-- 物体（示例用 lego.obj；10g 质量） -->
    <body name="obj" pos="0.05 0.00 0.01" childclass="lego">
      <freejoint/>
      <geom type="mesh" mesh="lego" rgba="0.9 0.3 0.1 1" mass="0.01"/>
    </body>
  </worldbody>

  <!-- 用 mocap 焊到夹爪；再让左右手指对称 -->
  <equality>
    <weld body1="g2" body2="tcp_target"/>
    <joint joint1="right_joint" joint2="left_joint"/>
  </equality>

  <!-- 用 position actuator 直接指定开口，稳！ -->
  <actuator>
    <position name="left"  joint="left_joint"  kp="2000" ctrlrange="0 0.0166"/>
    <position name="right" joint="right_joint" kp="2000" ctrlrange="0 0.0166"/>
  </actuator>

  <sensor>
    <actuatorpos name="left_p"  actuator="left"/>
    <actuatorpos name="right_p" actuator="right"/>
    <actuatorfrc name="left_f"  actuator="left"/>
    <actuatorfrc name="right_f" actuator="right"/>
  </sensor>
</mujoco>
