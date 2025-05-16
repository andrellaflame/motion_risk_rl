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
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        obs = np.concatenate([
            position[7:],      # Joint positions (nq - 7)
            position[3:7],     # Root orientation (quaternion)
            position[2:3],     # Root Z position
            velocity           # All qvel
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility

        # Reset simulation state
        mujoco.mj_resetData(self.model, self.data)

        # Set to initial pose with slight randomization for robustness
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nv
        )
        
        qpos[2] = self.init_qpos[2] # Ensure consistent start height
        qpos[3:7] = self.init_qpos[3:7] # Ensure consistent start orientation

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data) # Recalculate dependent states

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Apply action
        self.data.ctrl[:] = action.copy()

        # Simulate for frame_skip steps
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # --- Refined Reward Calculation ---
        
        # 1. Forward Velocity Reward (Encourage moving forward)
        forward_velocity = self.data.qvel[0] # Root's x-velocity
        forward_reward_weight = 1.25
        forward_reward = forward_reward_weight * forward_velocity

        # 2. Healthy/Alive Bonus (Constant bonus for not being terminated)
        # This is given at each step *if* the agent is not terminated in this step.
        healthy_reward_bonus = 2.0 # Tune this; make it substantial enough to want to stay alive

        # 3. Control Cost (Penalize large motor efforts for efficiency)
        ctrl_cost_weight = 0.01 # Tune this (0.001 to 0.1 is common)
        ctrl_cost = -ctrl_cost_weight * np.sum(np.square(action))

        # 4. Uprightness Reward (CRUCIAL for stable walking)
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        # xmat is (nbody x 9) array of rotation matrices, reshape to (nbody x 3 x 3)
        # The third column (index 2) of a body's rotation matrix is its z-axis in world frame.
        torso_z_axis_world = self.data.xmat[torso_id].reshape(3, 3)[:, 2]
        
        # Dot product of torso's z-axis with world_up_vector [0, 0, 1]
        # This value is 1 if perfectly upright, 0 if horizontal, -1 if upside down.
        upright_value = torso_z_axis_world[2] 
        
        upright_reward_weight = 0.5 # Tune this
        # Reward more strongly as it gets closer to 1. Penalize if leaning too much.
        # Example: Quadratic reward for uprightness, or simply scale upright_value
        # upright_reward = upright_reward_weight * upright_value # Simple linear
        upright_reward = upright_reward_weight * (1.0 + upright_value) / 2.0 # Scales from 0 (upside down) to upright_reward_weight (upright)

        # --- Check Termination Conditions ---
        terminated = False
        current_root_height = self.data.qpos[2] # Z-position of the root/torso base
        
        # Define healthy height range (relative to model's initial height ~1.282)
        min_healthy_height = 0.75 # If root goes below this, it's fallen (adjust based on model proportions)
        max_healthy_height = 1.5  # To prevent excessive jumping if not desired

        # Healthy if within height range AND reasonably upright
        is_healthy = (current_root_height >= min_healthy_height and \
                      current_root_height <= max_healthy_height and \
                      upright_value > 0.5) # Must be leaning forward/upright, not too far back/down

        if not is_healthy:
            terminated = True
            # When terminated, the reward for this step is primarily a large penalty.
            # Other components like forward_reward, ctrl_cost for this step are less relevant
            # or could be included if you want to penalize thrashing while falling.
            reward_on_termination = -100.0 # Significant penalty for falling
            current_step_reward = reward_on_termination
        else:
            # If healthy, sum up all the positive and negative incentives
            current_step_reward = forward_reward + healthy_reward_bonus + ctrl_cost + upright_reward
            # Note: healthy_reward_bonus is only given if not terminated.

        # Truncation (e.g., if episode runs for too long without falling)
        # Not explicitly implemented here, but TimeLimit wrappers in SB3 handle this.
        # If you add it, it would set truncated = True
        truncated = False 

        observation = self._get_obs()
        
        info = {
            "forward_reward": forward_reward,
            "healthy_reward_bonus": healthy_reward_bonus if not terminated else 0,
            "ctrl_cost": ctrl_cost,
            "upright_reward": upright_reward,
            "current_height": current_root_height,
            "upright_value": upright_value,
            "is_healthy": is_healthy,
        }
        if terminated:
            info["termination_reason"] = "unhealthy (fell or tilted too much)"

        if self.render_mode == "human":
            self._render_frame()

        return observation, current_step_reward, terminated, truncated, info
        # Apply action
        self.data.ctrl[:] = action.copy()

        # Simulate for frame_skip steps
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # --- Calculate Reward ---
        # This is the MOST CRITICAL part for learning to walk.
        # You need to design this carefully.
        # Components often include:
        # 1. Forward velocity: Reward moving forward (e.g., x-velocity of torso/root)
        # 2. Healthy reward: Constant bonus for not falling / staying alive.
        # 3. Control cost: Penalty for excessive motor torques (e.g., -sum(square(actions)))
        # 4. Contact cost (optional): Penalty for hard contacts or undesired contacts.
        # 5. Uprightness: Penalty for torso leaning too much.

        # Get root/torso x-velocity (qvel[0])
        forward_velocity = self.data.qvel[0]
        
        # Healthy reward (e.g., +1 as long as not terminated)
        healthy_reward = 1.0 # Simplified, Humanoid-v4 has a more complex one

        # Control cost (penalize large actions)
        ctrl_cost_weight = 0.01 # Tune this
        ctrl_cost = -ctrl_cost_weight * np.sum(np.square(action))

        reward = forward_velocity + healthy_reward + ctrl_cost

        # --- Check Termination & Truncation ---
        terminated = False
        # Terminate if z-position of torso is too low (fallen)
        if self.data.qpos[2] < 0.65: # Adjust this threshold
            terminated = True
            healthy_reward = 0 # No healthy reward if fallen
            reward = -50 # Large penalty for falling (example)

        truncated = False # Placeholder

        observation = self._get_obs()
        info = {
            "forward_velocity": forward_velocity,
            "ctrl_cost": ctrl_cost,
            "healthy_reward": healthy_reward,
        }

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

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