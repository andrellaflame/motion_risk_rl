import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

BVH_ROOT_NAME = 'Hips'
BVH_ROOT_EULER_ORDER = 'zyx'

# --- REFINED JOINT_MAPPING for humanoid.xml AND Walking_FW.bvh ---
JOINT_MAPPING = {
    # MuJoCo Abdomen (target) vs. BVH Spine (source: LowerBack, Spine, Spine1)
    'abdomen_x': {'bvh_joint_name': 'Spine',  'bvh_channel_index': 2, 'sign_multiplier': 1.0}, # BVH Spine Xrot (often flexion)
    'abdomen_y': {'bvh_joint_name': 'Spine',  'bvh_channel_index': 1, 'sign_multiplier': 1.0}, # BVH Spine Yrot (twist)
    'abdomen_z': {'bvh_joint_name': 'Spine',  'bvh_channel_index': 0, 'sign_multiplier': 1.0}, # BVH Spine Zrot (side bend)

    # Left Leg
    'hip_x_left':  {'bvh_joint_name': 'LeftUpLeg', 'bvh_channel_index': 2, 'sign_multiplier': -1.0}, # Swing: BVH Xrot. MJ positive is backward, check BVH Xrot sign.
    'hip_y_left':  {'bvh_joint_name': 'LeftUpLeg', 'bvh_channel_index': 1, 'sign_multiplier': 1.0},  # Twist: BVH Yrot.
    'hip_z_left':  {'bvh_joint_name': 'LeftUpLeg', 'bvh_channel_index': 0, 'sign_multiplier': -1.0}, # Abduction: BVH Zrot. MJ positive is inward, check BVH Zrot sign.
    'knee_left':   {'bvh_joint_name': 'LeftLeg',  'bvh_channel_index': 2, 'sign_multiplier': -1.0}, # Knee flexion (on BVH Xrot of LeftLeg), sign already tuned.
    'ankle_x_left':{'bvh_joint_name': 'LeftFoot',  'bvh_channel_index': 2, 'sign_multiplier': -1.0},# Ankle pitch (on BVH Xrot of LeftFoot)
    'ankle_y_left':{'bvh_joint_name': 'LeftFoot',  'bvh_channel_index': 1, 'sign_multiplier': 1.0},  # Ankle twist (on BVH Yrot of LeftFoot)

    # Right Leg
    'hip_x_right':  {'bvh_joint_name': 'RightUpLeg', 'bvh_channel_index': 2, 'sign_multiplier': 1.0},
    'hip_y_right':  {'bvh_joint_name': 'RightUpLeg', 'bvh_channel_index': 1, 'sign_multiplier': 1.0},
    'hip_z_right':  {'bvh_joint_name': 'RightUpLeg', 'bvh_channel_index': 0, 'sign_multiplier': 1.0},
    'knee_right':   {'bvh_joint_name': 'RightLeg',  'bvh_channel_index': 2, 'sign_multiplier': -1.0},
    'ankle_x_right':{'bvh_joint_name': 'RightFoot',  'bvh_channel_index': 2, 'sign_multiplier': 1.0},
    'ankle_y_right':{'bvh_joint_name': 'RightFoot',  'bvh_channel_index': 1, 'sign_multiplier': 1.0},

    'shoulder1_left': {'bvh_joint_name': 'LeftArm', 'bvh_channel_index': 2, 'sign_multiplier': 1.0}, 
    'shoulder2_left': {'bvh_joint_name': 'LeftArm', 'bvh_channel_index': 0, 'sign_multiplier': 1.0}, 
    'elbow_left':     {'bvh_joint_name': 'LeftForeArm',  'bvh_channel_index': 2, 'sign_multiplier': 1.0},

    'shoulder1_right':{'bvh_joint_name': 'RightArm', 'bvh_channel_index': 2, 'sign_multiplier': 1.0},
    'shoulder2_right':{'bvh_joint_name': 'RightArm', 'bvh_channel_index': 0, 'sign_multiplier': 1.0},
    'elbow_right':    {'bvh_joint_name': 'RightForeArm',  'bvh_channel_index': 2, 'sign_multiplier': 1.0},
}

GLOBAL_Z_OFFSET = 1.0

_first_run_logging_done = False

def bvh_frame_to_qpos(
        bvh_frame_joint_data, 
        bvh_joint_names_list, 
        mujoco_model, 
        mj_joint_name_to_id_map
    ):
    global _first_run_logging_done
    qpos_target = np.zeros(mujoco_model.nq)
    qpos_target[2] = 1.0 
    qpos_target[3] = 1.0 

    try:
        root_bvh_idx = bvh_joint_names_list.index(BVH_ROOT_NAME) 
        root_bvh_channels = bvh_frame_joint_data[root_bvh_idx]

        if len(root_bvh_channels) >= 6:
            bvh_x_pos = root_bvh_channels[0]
            bvh_y_pos = root_bvh_channels[1] 
            bvh_z_pos = root_bvh_channels[2] 
            
            bvh_root_rot_rad = np.array(root_bvh_channels[3:6])
                    
            scale_factor = 0.01
            
            qpos_target[0] = bvh_x_pos * scale_factor
            qpos_target[1] = bvh_z_pos * scale_factor 
            qpos_target[2] = (bvh_y_pos * scale_factor) + GLOBAL_Z_OFFSET

            try:
                r_bvh = R.from_euler(BVH_ROOT_EULER_ORDER.lower(), bvh_root_rot_rad, degrees=False)
                r_transform_to_mujoco_frame = R.from_euler('z', 90, degrees=True)
                final_rotation = r_transform_to_mujoco_frame * r_bvh
                
                quat_xyzw = final_rotation.as_quat() 
                qpos_target[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            except Exception as e:
                if not _first_run_logging_done: print(f"Warning: Error converting root Euler angles: {e}. Using default orientation.")
                qpos_target[3:7] = [1, 0, 0, 0]
        else:
            if not _first_run_logging_done: print(f"Warning: Root joint '{BVH_ROOT_NAME}' has < 6 channels.")
            qpos_target[0:3] = [0, 0, GLOBAL_Z_OFFSET]
            qpos_target[3:7] = [1, 0, 0, 0]

    except ValueError:
        if not _first_run_logging_done: print(f"CRITICAL Error: Root joint '{BVH_ROOT_NAME}' not found in BVH: {bvh_joint_names_list}")
        qpos_target[0:3] = [0, 0, GLOBAL_Z_OFFSET] 
        qpos_target[3:7] = [1, 0, 0, 0]

    mapped_mj_joints_count = 0
    unmapped_log_entries = []

    for mj_joint_name, mapping_info in JOINT_MAPPING.items():
        if mj_joint_name not in mj_joint_name_to_id_map:
            if not _first_run_logging_done:
                unmapped_log_entries.append(f"'{mj_joint_name}' (from JOINT_MAPPING) not in MuJoCo model.")
            continue

        bvh_mapped_joint_name = mapping_info['bvh_joint_name']
        try:
            bvh_joint_idx = bvh_joint_names_list.index(bvh_mapped_joint_name)
            bvh_joint_channel_data = bvh_frame_joint_data[bvh_joint_idx] 
            
            if len(bvh_joint_channel_data) == 3: 
                bvh_rot_channels = np.array(bvh_joint_channel_data)
                channel_idx_to_use = mapping_info['bvh_channel_index']
                sign = mapping_info.get('sign_multiplier', 1.0)

                if 0 <= channel_idx_to_use < len(bvh_rot_channels):
                    angle_rad = bvh_rot_channels[channel_idx_to_use] * sign
                    mujoco_joint_id = mj_joint_name_to_id_map[mj_joint_name]
                    qpos_addr = mujoco_model.jnt_qposadr[mujoco_joint_id]
                    
                    if mujoco_model.jnt_type[mujoco_joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
                        qpos_target[qpos_addr] = angle_rad
                        mapped_mj_joints_count +=1
                else:
                    if not _first_run_logging_done:
                        unmapped_log_entries.append(f"Invalid 'bvh_channel_index': {channel_idx_to_use} for BVH '{bvh_mapped_joint_name}'.")
            else:
                if not _first_run_logging_done:
                     unmapped_log_entries.append(f"BVH joint '{bvh_mapped_joint_name}' for MJ '{mj_joint_name}' does not have 3 rot channels (has {len(bvh_joint_channel_data)}).")
        except ValueError: # BVH joint name not found
            if not _first_run_logging_done:
                unmapped_log_entries.append(f"BVH joint '{bvh_mapped_joint_name}' (for MJ '{mj_joint_name}') not found in bvh_names_list.")
        except IndexError: # Should be caught by the channel_idx_to_use check
             if not _first_run_logging_done:
                unmapped_log_entries.append(f"Index error for BVH joint '{bvh_mapped_joint_name}' (MJ '{mj_joint_name}').")
    
    if not _first_run_logging_done:
        print(f"\n[LOGGING - First Frame Only]")
        print(f"Attempted to map {len(JOINT_MAPPING)} MuJoCo joints based on JOINT_MAPPING dictionary.")
        print(f"Successfully set qpos for {mapped_mj_joints_count} MuJoCo hinge joints this frame.")
        if unmapped_log_entries:
            print("Issues during mapping (MuJoCo joints in JOINT_MAPPING potentially not correctly driven):")
            for item in unmapped_log_entries:
                print(f"  - {item}")
        elif mapped_mj_joints_count == len(JOINT_MAPPING):
             print("All joints in JOINT_MAPPING successfully processed for qpos assignment.")
        print("---------------------------------------\n")
        _first_run_logging_done = True
            
    return qpos_target