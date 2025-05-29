# This scripts launches mujoco viewer

import mujoco
import mujoco.viewer
import numpy as np
import time
import os

model_path = "venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets/humanoid.xml"
walk_qpos = np.load("data/skeleton/walk_qpos.npy")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print(f"Model nq (qpos length): {model.nq}")
print(f"Loaded walk_qpos shape: {walk_qpos.shape}")
print(f"One qpos sample shape: {walk_qpos[0].shape}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for qpos in walk_qpos:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.01)
