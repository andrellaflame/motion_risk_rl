import mujoco
from mujoco import viewer

model = mujoco.MjModel.from_xml_path("models/humanoid.xml")
data = mujoco.MjData(model)

if data is not None:
    print("MuJoCo model loaded successfully!")


with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()

