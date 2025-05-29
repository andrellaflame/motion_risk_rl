import time
import mujoco
import mujoco.viewer
import numpy as np
from bvh import Bvh

from bvh_parser import load_bvh_frames
from bvh_to_qpos import BVH_ROOT_NAME, bvh_frame_to_qpos, JOINT_MAPPING 

MODEL_PATH = "mujoco_animator/models/humanoid.xml" 
BVH_PATH = "mujoco_animator/motion/Walking_FW.bvh"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# --- BVH Loading and Info ---
motion, bvh_names_list, dt = load_bvh_frames(BVH_PATH)
print(f"--- BVH Info ---")
print(f"Loaded '{BVH_PATH}'")
print(f"Number of frames: {len(motion)}")
print(f"Frame time: {dt:.6f} seconds ({1/dt:.2f} FPS)")
print(f"Number of joints found in BVH: {len(bvh_names_list)}")
print(f"BVH Joint names: {bvh_names_list}")
# --- End BVH Info ---

# --- MuJoCo Model Info ---
mj_joint_name_to_id_map = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(model.njnt)}
print(f"\n--- MuJoCo Model Info ---")
print(f"Number of joints in MuJoCo model: {model.njnt}")
# print(f"MuJoCo Joint names: {list(mj_joint_name_to_id_map.keys())}") # Optional: very verbose
# --- End MuJoCo Model Info ---

print(f"\nAttempting to map {len(JOINT_MAPPING)} MuJoCo joints from JOINT_MAPPING dict in bvh_to_qpos.py")


# --- Debug print first frame of BVH data (as loaded by your parser) ---
if len(motion) > 0:
    print("\n[DEBUG] First frame of BVH data (rotations in radians, positions as is):")
    for name_idx, bvh_joint_name_str in enumerate(bvh_names_list):
        channel_values = motion[0][name_idx]
        is_likely_root = (bvh_joint_name_str.lower() == BVH_ROOT_NAME.lower() and len(channel_values) >= 6)
        if is_likely_root:
            pos_channels = channel_values[:3]
            rot_channels_rad = channel_values[3:6] # These are Z,Y,X based on new BVH
            print(f"  {bvh_joint_name_str} (Root): Pos: {pos_channels}, Rot (rad): {rot_channels_rad} (Deg: {np.degrees(rot_channels_rad)})")
        elif len(channel_values) == 3: # Assuming these are Z,Y,X rotations
            print(f"  {bvh_joint_name_str}: Rot (rad): {channel_values} (Deg: {np.degrees(channel_values)})")
        elif len(channel_values) > 0:
            print(f"  {bvh_joint_name_str}: Channels: {channel_values}")
# --- End Debug print ---

all_qpos_trajectory = [] # For logging joint trajectory

with mujoco.viewer.launch_passive(model, data) as viewer:
    frame_count = len(motion)
    if frame_count == 0:
        print("No motion frames loaded. Exiting.")
        exit()
        
    i = 0
    start_time = time.time()
    while viewer.is_running():
        current_frame_bvh_data = motion[i % frame_count] 
        
        qpos = bvh_frame_to_qpos(current_frame_bvh_data, bvh_names_list, model, mj_joint_name_to_id_map)
        
        data.qpos[:] = qpos
        all_qpos_trajectory.append(qpos.copy()) # Log the qpos for this frame

        mujoco.mj_forward(model, data)
        viewer.sync()
        
        # Synchronize with BVH frame rate
        loop_start_time = time.time()
        elapsed_since_start = loop_start_time - start_time
        target_elapsed = i * dt
        sleep_time = target_elapsed - elapsed_since_start
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        i += 1
        if i >= frame_count and frame_count > 1: # Loop the animation
            i = 0
            start_time = time.time() 
