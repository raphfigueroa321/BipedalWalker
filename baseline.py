import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import os
import datetime

# --- CONFIGURATION ---
EXPERIMENT_NAME = "PPO_Baseline_V2"
TOTAL_TIMESTEPS = 800000
SEEDS = [1, 2, 3]
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

for seed in SEEDS:
    # Setup file paths
    run_path = os.path.join(EXPERIMENT_NAME, timestamp, f"Seed_{seed}")
    print(f"\n--- Starting: {run_path} ---")
    
    # Setup environment
    set_random_seed(seed)
    env = gym.make("BipedalWalker-v3", render_mode=None)
    
    # Monitor wrapper required for displaying reward stats in TensorBoard
    env = Monitor(env)
    env.reset(seed=seed)
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=run_path)

    # Save model and normalization stats
    model_save_dir = f"models/{run_path}"
    os.makedirs(model_save_dir, exist_ok=True)
    
    model.save(os.path.join(model_save_dir, "walker_model"))
    
    print(f"Finished Seed {seed}. Model saved to: {model_save_dir}")
    env.close()