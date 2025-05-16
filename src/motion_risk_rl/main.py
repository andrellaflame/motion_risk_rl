import sys
import time
import os

package_parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if package_parent_directory not in sys.path:
    sys.path.insert(0, package_parent_directory)

from motion_risk_rl.mujoco_env.custom_humanoid_env import CustomHumanoidEnv

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

import mujoco
import mujoco.viewer

# --- Configuration ---
XML_FILE_PATH = "models/humanoid.xml"

LOG_DIR = "rl_logs_custom_humanoid/" 
MODEL_SAVE_PATH = os.path.join(LOG_DIR, "custom_humanoid_ppo_model")
BEST_MODEL_SAVE_PATH = os.path.join(LOG_DIR, "best_custom_humanoid_ppo_model")
TENSORBOARD_LOG_PATH = os.path.join(LOG_DIR, "custom_humanoid_ppo_tensorboard/")

TOTAL_TIMESTEPS = 3_000_000

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

def create_env(render_mode_str=None):
    """Helper function to create the custom environment."""
   
    xml_full_path = "../../" + XML_FILE_PATH 
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    xml_full_path = os.path.join(project_root, XML_FILE_PATH)

    if not os.path.exists(xml_full_path):
        raise FileNotFoundError(f"Humanoid XML not found at {xml_full_path}. Check path construction in main.py")

    env = CustomHumanoidEnv(xml_file_path=xml_full_path, render_mode=render_mode_str)
    return env

def train_humanoid():
    print(f"Training custom humanoid from {XML_FILE_PATH}...")

    env = DummyVecEnv([lambda: create_env(render_mode_str=None)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env = DummyVecEnv([lambda: create_env(render_mode_str=None)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // env.num_envs, 1),
        save_path=LOG_DIR,
        name_prefix="custom_humanoid_checkpoint"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        log_path=LOG_DIR,
        eval_freq=max(25_000 // env.num_envs, 1),
        deterministic=True,     
        render=False,           
        n_eval_episodes=5 
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_PATH,
    )

    print("Starting training for custom humanoid...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback] 
        )
    finally:
        model.save(MODEL_SAVE_PATH)
        env.save(os.path.join(LOG_DIR, "vec_normalize_stats.pkl"))
        print(f"Training finished. Model saved to {MODEL_SAVE_PATH}")

    env.close()
    eval_env.close()


def watch_trained_humanoid(model_path, stats_path):
    print(f"Loading custom humanoid model from {model_path}")
    eval_env_vis = create_env(render_mode_str="human")

    vec_eval_env_vis = DummyVecEnv([lambda: eval_env_vis])
    loaded_vec_env = VecNormalize.load(stats_path, vec_eval_env_vis)
    loaded_vec_env.training = False
    loaded_vec_env.norm_reward = False

    model = PPO.load(model_path, env=loaded_vec_env)
    print("Model loaded. Starting visualization...")
    obs = loaded_vec_env.reset()
    for i in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = loaded_vec_env.step(action)

        time.sleep(0.02)

        if dones.any():
            obs = loaded_vec_env.reset()
    loaded_vec_env.close()


def continue_training_humanoid(model_to_load_path, stats_to_load_path, additional_timesteps):
    print(f"Continuing training from {model_to_load_path} for an additional {additional_timesteps} timesteps...")

    # Create the non-normalized training env first
    train_env_instance = DummyVecEnv([lambda: create_env(render_mode_str=None)])
    # Load the VecNormalize stats and apply them to the environment
    env = VecNormalize.load(stats_to_load_path, train_env_instance)
    env.training = True  # Ensure it's in training mode
    env.norm_reward = True # Ensure reward normalization is active as per original training

    # Load the model
    model = PPO.load(model_to_load_path, env=env, tensorboard_log=TENSORBOARD_LOG_PATH)
    # If you want to adjust learning rate for continued training (fine-tuning):
    # model.learning_rate = 1e-5 # Example: smaller learning rate

    # Setup callbacks for continued training (similar to the initial train_humanoid)
    eval_env_cont = DummyVecEnv([lambda: create_env(render_mode_str=None)])
    # Important: Load the *same* normalization stats for the eval_env, but set training=False
    eval_env_cont = VecNormalize.load(stats_to_load_path, eval_env_cont)
    eval_env_cont.training = False
    eval_env_cont.norm_reward = False

    checkpoint_callback_cont = CheckpointCallback(
        save_freq=max(100_000 // env.num_envs, 1),
        save_path=LOG_DIR,
        name_prefix="custom_humanoid_cont_checkpoint" # New prefix
    )
    eval_callback_cont = EvalCallback(
        eval_env_cont,
        best_model_save_path=BEST_MODEL_SAVE_PATH, # Can continue saving to the same best model path
        log_path=LOG_DIR,
        eval_freq=max(25_000 // env.num_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    print("Starting continued training session...")
    try:
        model.learn(
            total_timesteps=additional_timesteps,
            reset_num_timesteps=False,  # VERY IMPORTANT for continuing
            callback=[checkpoint_callback_cont, eval_callback_cont]
        )
    finally:
        # Save with a new name or overwrite, your choice
        continued_model_save_path = MODEL_SAVE_PATH.replace(".zip", "_continued.zip") 
        continued_stats_save_path = os.path.join(LOG_DIR, "vec_normalize_stats_continued.pkl")
        model.save(continued_model_save_path)
        env.save(continued_stats_save_path)
        print(f"Continued training finished. Model saved to {continued_model_save_path}")

    env.close()
    eval_env_cont.close()

if __name__ == "__main__":
    SHOULD_CONTINUE_TRAINING = False 
    ADDITIONAL_TRAINING_STEPS = 5_000_000

    best_model_file = os.path.join(BEST_MODEL_SAVE_PATH, "best_model.zip")
    stats_file_for_best_model = os.path.join(BEST_MODEL_SAVE_PATH, "vecnormalize.pkl")

    final_model_file = MODEL_SAVE_PATH + ".zip"
    stats_file_for_final_model = os.path.join(LOG_DIR, "vec_normalize_stats.pkl")

    current_model = None
    stats_to_continue = None

    if os.path.exists(best_model_file) and os.path.exists(stats_file_for_best_model):
        print(f"Found best model at {best_model_file}.")
        current_model = best_model_file
        stats_to_continue = stats_file_for_best_model
    elif os.path.exists(final_model_file) and os.path.exists(stats_file_for_final_model):
        print(f"Best model not found, trying final saved model from {final_model_file}.")
        current_model = final_model_file
        stats_to_continue = stats_file_for_final_model

    if SHOULD_CONTINUE_TRAINING:
        if current_model and stats_to_continue:
            print(f"Continuing training saved model from {current_model}.")
            continue_training_humanoid(current_model, stats_to_continue, ADDITIONAL_TRAINING_STEPS)
            print("\nTraining complete. Run script again to watch, modify SHOULD_CONTINUE_TRAINING.")
        else:
            print("No existing model found to continue training. Starting a new training session.")
            train_humanoid()
            print("\nTraining complete. Run script again to watch, modify SHOULD_CONTINUE_TRAINING.")
    else:
        if os.path.exists(current_model) and os.path.exists(stats_to_continue):
            watch_trained_humanoid(model_path=current_model, stats_path=stats_to_continue)
        else:
            print("No trained model found. Cancelling viewer session.")