import math

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("/home/nuo/ME5418/franka_emika_panda/scene_withobstacles.xml")
data = mujoco.MjData(model)

box_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "move_box_x")
sphere_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "moving_sphere")

box_amplitude = 0.12
sphere_amplitude = 0.12
motion_frequency = 0.25

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = data.time
        phase = 2.0 * math.pi * motion_frequency * t
        data.ctrl[box_actuator_id] = box_amplitude * math.sin(phase)
        data.ctrl[sphere_actuator_id] = sphere_amplitude * math.sin(phase + math.pi / 2.0)
        mujoco.mj_step(model, data)
        viewer.sync()
