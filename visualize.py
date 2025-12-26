import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

# 1. Path to your best seed folder
# Change this to match the specific timestamp/seed you want to see
MODEL_PATH = "models/PPO_HullPenaltyV1/20251225-2325/Seed_5"

def visualize():
    # 2. Re-create the environment with Human rendering
    def make_env():
        # render_mode="human" allows you to actually see the window
        return gym.make("BipedalWalker-v3", render_mode="human")

    env = DummyVecEnv([make_env])

    # 3. Load the EXACT normalization stats used during training
    stats_path = os.path.join(MODEL_PATH, "vec_normalize.pkl")
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        
        # CRITICAL: Stop the stats from updating during testing
        env.training = False 
        # We want to see the real score, not the normalized reward
        env.norm_reward = False 
    else:
        print("Warning: No normalization stats found. Robot will likely fail.")

    # 4. Load the trained brain
    model_path = os.path.join(MODEL_PATH, "walker_model.zip")
    model = PPO.load(model_path, env=env)

    # 5. Run the "Enjoy" loop
    obs = env.reset()
    for _ in range(2000): # Run for 2000 frames
        # Use deterministic=True for the 'best' performance
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # VecEnv automatically resets, so we don't need a manual reset check
        env.render()

    env.close()

if __name__ == "__main__":
    visualize()