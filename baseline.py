import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import os
import datetime

# --- CONFIGURATION ---
EXPERIMENT_NAME = "PPO_Baseline_V2" # Named V2 to distinguish from previous messy run
TOTAL_TIMESTEPS = 100000            # Increased to 100k for meaningful baseline data
SEEDS = [1, 2, 3, 4, 5]
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

for seed in SEEDS:
    # 1. Create unique path for both Logs and Models
    run_path = os.path.join(EXPERIMENT_NAME, timestamp, f"Seed_{seed}")
    print(f"\n--- Starting: {run_path} ---")
    
    # 2. Setup Environment
    set_random_seed(seed)
    env = gym.make("BipedalWalker-v3", render_mode=None)
    
    # CRITICAL: Monitor wrapper ensures rollout/ep_rew_mean shows in TensorBoard
    env = Monitor(env)
    env.reset(seed=seed)
    
    # 3. Initialize Model
    # tb_log_name uses the same run_path for consistent TensorBoard grouping
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    
    # 4. Train
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=run_path)
    
    # 5. Save Model in its own specific Seed folder
    model_save_dir = f"models/{run_path}"
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Saving with an explicit filename inside the directory
    model.save(os.path.join(model_save_dir, "walker_model"))
    
    print(f"Finished Seed {seed}. Model saved to: {model_save_dir}")
    env.close()