import mujoco
import numpy as np


model=mujoco.load_model_from_path("franka_emika_panda/scene_withobstacles.xml")
data=mujoco.MjData(model)

left_tip_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_tip")
right_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_tip")

def get_grasp_center(self):
    L = self.data.site_xpos[self.left_tip_id]
    R = self.data.site_xpos[self.right_tip_id]
    return 0.5 * (L + R)

def distance_to_goal(self):
    center = self.get_grasp_center()
    return float(np.linalg.norm(center - self.goal_pos))