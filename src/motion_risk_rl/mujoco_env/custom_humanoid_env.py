import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
    "azimuth": 120.0,
}

class CustomHumanoidEnv(gym.Env):
    """
    Custom Environment for a Humanoid model defined in a specific XML.
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 0, # Will be set based on model.opt.timestep and frame_skip
    }

    def __init__(self, xml_file_path, frame_skip=5, render_mode=None):
        super().__init__()

        if not os.path.exists(xml_file_path):
            raise FileNotFoundError(f"XML file not found at {xml_file_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_file_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        CustomHumanoidEnv.metadata["render_fps"] = int(np.round(1.0 / (self.model.opt.timestep * self.frame_skip)))

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        obs_dim = (self.model.nq - 7) + 4 + 1 + self.model.nv # (joint_pos) + root_quat + root_z + all_qvel

        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.viewer = None

        self.init_qpos = self.data.qpos.copy()
        self.init_qpos[2] = 1.282 
        self.init_qpos[3:7] = [1, 0, 0, 0]
        self.init_qvel = self.data.qvel.copy()


    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        root_pos = qpos[:3]
        root_quat = qpos[3:7]
        joint_pos = qpos[7:]

        return np.concatenate([
            joint_pos,
            root_quat,
            [root_pos[2]],  # z-position (height)
            qvel,
        ])


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # === Reward: walk forward, stay upright, use less energy ===
        forward_velocity = self.data.qvel[0]  # Velocity in x direction
        height = self.data.qpos[2]
        upright = 1.0 if height > 1.0 else 0.0  # Encourage staying up
        ctrl_cost = 0.1 * np.square(action).sum()

        reward = forward_velocity + upright - ctrl_cost

        # === Done if agent falls ===
        done = bool(height < 0.8 or height > 2.0)

        return obs, reward, done, False, {}

    def render(self):
        # This is the new render API for Gymnasium
        if self.render_mode == "rgb_array":
            if self.viewer is None:
                 self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            # This part needs proper setup for offscreen rendering to an array.
            width, height = 640, 480
            img_data = np.zeros((height, width, 3), dtype=np.uint8)
            return img_data # Placeholder
        elif self.render_mode == "human":
            self._render_frame()
            return True # For compatibility with some wrappers
        elif self.render_mode == "depth_array":
            # Similar to rgb_array, needs setup for depth rendering.
            # self.mujoco_renderer.render(self.render_mode)
            # Fallback:
            width, height = 640, 480 # example
            depth_data = np.zeros((height, width), dtype=np.float32)
            return depth_data # Placeholder
        return None # Or an empty list for some render modes

    def _render_frame(self):
        if self.viewer is None and self.render_mode == "human":
            # For "human" mode, launch_passive is a simple way if you don't manage own GLFW window.
            # Note: launch_passive blocks if not handled carefully in an interactive script.
            # For SB3, it usually expects render() to update and return.
            # A better way for "human" mode with SB3 is to manage own viewer window
            # or use a wrapper that does.
            # However, mujoco.viewer.launch_passive() inits its own loop.
            # A simple sync might be enough if the viewer is already launched by SB3's logic
            # or if you are calling this in a way that an external viewer is expected.
            # For now, if a viewer isn't managed by a wrapper, this will try to launch one.
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception as e:
                # If viewer was closed
                print(f"MuJoCo viewer sync error: {e}. Recreating viewer.")
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.sync()


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        # if hasattr(self, 'mujoco_renderer'):
        #     self.mujoco_renderer.close()

# Optional: To test environment standalone
if __name__ == "__main__":
    xml_path = "../../models/humanoid.xml" # Adjust path as necessary from where you run this
    if not os.path.exists(xml_path):
        print(f"Test XML not found at {xml_path}. Please check path.")
    else:
        env = CustomHumanoidEnv(xml_file_path=xml_path, render_mode="human")
        
        # Test reset
        obs, info = env.reset()
        print("Observation space shape:", env.observation_space.shape)
        print("Initial observation:", obs)
        
        # Test step
        for i in range(1000): # Simulate for 1000 steps
            action = env.action_space.sample() # Take random actions
            obs, reward, terminated, truncated, info = env.step(action)
            env.render() # Should call _render_frame
            if (i % 100 == 0) :
                print(f"Step: {i}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps.")
                obs, info = env.reset()
        env.close()