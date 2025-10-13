import mujoco
model = mujoco.MjModel.from_xml_path('franka_emika_panda/panda.xml')
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)   # 可选，但推荐：把qpos等设为默认
mujoco.mj_forward(model, data)     # 必需：计算所有派生量

left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'left_tip')
right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right_tip')

print('Left tip:', data.site_xpos[left_id])
print('Right tip:', data.site_xpos[right_id])