import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("/home/nuo/ME5418/franka_emika_panda/scene_withobstacles.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[model.actuator_name2id("move_box_x")] = 0.2   # 正向
        data.ctrl[model.actuator_name2id("move_sphere_y")] = -0.2  # 反向
        mujoco.mj_step(model, data)
        viewer.sync()
