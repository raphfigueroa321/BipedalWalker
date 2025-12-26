import datetime
import os
from multiprocessing import freeze_support

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv


class HullAnglePenaltyWrapper(gym.Wrapper):
    """Penalize episodes when the hull tilts beyond a threshold angle."""

    def __init__(self, env, angle_threshold: float = 0.2, penalty_coef: float = 5.0):
        super().__init__(env)
        self.angle_threshold = angle_threshold
        self.penalty_coef = penalty_coef

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        hull = getattr(self.env.unwrapped, "hull", None)
        angle = abs(hull.angle) if hull is not None and hasattr(hull, "angle") else None
        if angle is not None and angle > self.angle_threshold:
            reward -= self.penalty_coef * (angle - self.angle_threshold)
        return obs, reward, terminated, truncated, info

"""# 1. Setup folders
models_dir = "models/PPO_Baseline"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)"""

EXPERIMENT_NAME = "PPO_HullPenaltyV1"
TOTAL_TIMESTEPS = 1000000
SEEDS = [1, 2, 3, 4, 5]
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
NUM_ENVS = 6  # use 6 CPU cores


def make_env(base_seed: int, rank: int):
    def _init():
        env = gym.make("BipedalWalker-v3")
        env = HullAnglePenaltyWrapper(env, angle_threshold=0.2, penalty_coef=5.0)
        env = Monitor(env)
        env.reset(seed=base_seed + rank)
        return env

    return _init


def train_seed(seed: int):
    # 1. Path Management (Keeps TensorBoard exactly as before)
    run_path = os.path.join(EXPERIMENT_NAME, timestamp, f"Seed_{seed}")
    print(f"\n>>> Starting Experiment: {run_path}")

    # 2. Setup Env with Normalization
    set_random_seed(seed)
    env = SubprocVecEnv([make_env(seed, i) for i in range(NUM_ENVS)])
    env.seed(seed)  # Seeding the vectorized env

    # Normalization Wrapper (Crucial for BipedalWalker)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 3. Initialize Model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

    # 4. Train
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=run_path)

    # 5. Save Both Brain and Stats (Maintains folder workflow)
    model_dir = f"models/{run_path}"
    os.makedirs(model_dir, exist_ok=True)

    model.save(f"{model_dir}/walker_model")
    env.save(f"{model_dir}/vec_normalize.pkl")  # The "Filter"

    print(f"Finished Seed {seed}. Files saved to {model_dir}")
    env.close()


def main():
    freeze_support()  # Needed on Windows when using spawn with SubprocVecEnv
    for seed in SEEDS:
        train_seed(seed)


if __name__ == "__main__":
    main()