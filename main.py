import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import os
import datetime


"""# 1. Setup folders
models_dir = "models/PPO_Baseline"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)"""

EXPERIMENT_NAME = "PPO_Version_1"
TOTAL_TIMESTEPS = 10000
SEEDS = [1, 2, 3, 4, 5]
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

for seed in SEEDS:
    # 3. Create a unique path: Experiment/Timestamp/Seed
    run_path = os.path.join(EXPERIMENT_NAME, timestamp, f"Seed_{seed}")
    
    print(f"--- Starting: {run_path} ---")
    
    set_random_seed(seed)
    env = gym.make("BipedalWalker-v3", render_mode=None)
    env.reset(seed=seed)
    
    # Use the run_path as the tb_log_name
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=run_path)
    
    # Save the model with the same naming convention
    save_dir = f"models/{run_path}"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    model.save(save_dir)
    env.close()