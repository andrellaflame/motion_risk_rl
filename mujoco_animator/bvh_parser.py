from bvh import Bvh
import numpy as np

def load_bvh_frames(bvh_path):
    with open(bvh_path) as f:
        mocap = Bvh(f.read())

    joint_names = mocap.get_joints_names()
    n_frames = mocap.nframes
    dt = mocap.frame_time
    frames = []

    for i in range(n_frames):
        frame_data_for_current_bvh_frame = []
        for joint_name in joint_names:            
            joint_channel_names = mocap.joint_channels(joint_name)

            # print("Joint: " + joint_name + ", channels: " + ", ".join(joint_channel_names))

            joint_channel_values = []
            raw_values_for_joint = mocap.frame_joint_channels(i, joint_name, joint_channel_names)

            for value, channel_name in zip(raw_values_for_joint, joint_channel_names):
                if "rotation" in channel_name.lower():
                    joint_channel_values.append(np.radians(value))
                else:
                    joint_channel_values.append(value)
            frame_data_for_current_bvh_frame.append(np.array(joint_channel_values))
        frames.append(frame_data_for_current_bvh_frame)

    return np.array(frames, dtype=object), joint_names, dt