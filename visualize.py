import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

# File path of the chosen trained model (currently set to best model)
MODEL_PATH = "results/best_model"

def visualize():
    def make_env():
        return gym.make("BipedalWalker-v3", render_mode="human")

    env = DummyVecEnv([make_env])

    # Load normalization stats
    stats_path = os.path.join(MODEL_PATH, "vec_normalize.pkl")
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        
        env.training = False 
        env.norm_reward = False 
    else:
        print("Warning: No normalization stats found")

    model_path = os.path.join(MODEL_PATH, "walker_model.zip")
    model = PPO.load(model_path, env=env)

    # Run a single episode
    obs = env.reset()
    for _ in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        env.render()

    env.close()

if __name__ == "__main__":
    visualize()