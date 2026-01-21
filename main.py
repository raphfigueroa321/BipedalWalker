import datetime
import os
from multiprocessing import freeze_support

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

# Wrapper to penalize extreme hull angles
class HullAnglePenaltyWrapper(gym.Wrapper):
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

# Wrapper to penalize dragging legs
class LegContactPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty_coef: float = 0.02, sustained_penalty_coef: float = 0.05, sustain_steps: int = 3):
        super().__init__(env)
        self.penalty_coef = penalty_coef
        self.sustained_penalty_coef = sustained_penalty_coef
        self.sustain_steps = max(1, int(sustain_steps))
        self.contact_counters = None

    def _leg_in_contact(self, leg) -> bool:
        for attr in ("ground_contact", "contact", "foot_contact"):
            if hasattr(leg, attr):
                return bool(getattr(leg, attr))

        if hasattr(leg, "contacts"):
            v = getattr(leg, "contacts")
            try:
                return bool(len(v))
            except Exception:
                return bool(v)

        for k, v in getattr(leg, "__dict__", {}).items():
            if "contact" in k and isinstance(v, (bool, int, float)):
                return bool(v)

        return False

    def reset(self, **kwargs):
        self.contact_counters = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        legs = getattr(self.env.unwrapped, "legs", None)
        if legs:
            contacts = [self._leg_in_contact(l) for l in legs]
            if self.contact_counters is None:
                self.contact_counters = [0] * len(contacts)

            # update counters: increment when in contact, reset when not
            for i, c in enumerate(contacts):
                if c:
                    self.contact_counters[i] += 1
                else:
                    self.contact_counters[i] = 0

            # apply minimal penalty only for legs that have been contacting
            # for at least `sustain_steps` consecutive steps
            sustained_legs = sum(1 for cnt in self.contact_counters if cnt >= self.sustain_steps)
            if sustained_legs:
                reward -= self.sustained_penalty_coef * sustained_legs

        return obs, reward, terminated, truncated, info

EXPERIMENT_NAME = "HullPenaltyV2" # Set to desired experiment name
TOTAL_TIMESTEPS = 2000000
SEEDS = [1, 2, 3]
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
NUM_ENVS = 6 # Number of parallel environments, set to # of CPU cores

def make_env(base_seed: int, rank: int):

    # !! Apply wrappers, adjust parameters as necessary !!
    def _init():
        env = gym.make("BipedalWalker-v3")
        env = HullAnglePenaltyWrapper(env, angle_threshold=0.2, penalty_coef=5.0)
        # env = LegContactPenaltyWrapper(env, penalty_coef=0.02, sustained_penalty_coef=0.05, sustain_steps=3)
        env = Monitor(env)
        env.reset(seed=base_seed + rank)
        return env

    return _init


def train_seed(seed: int):
    # Setup paths
    run_path = os.path.join(EXPERIMENT_NAME, timestamp, f"Seed_{seed}")
    print(f"\n>>> Starting Experiment: {run_path}")

    # Setup Env with Normalization
    set_random_seed(seed)
    env = SubprocVecEnv([make_env(seed, i) for i in range(NUM_ENVS)])
    env.seed(seed)  # Seeding the vectorized env
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Initialize and train model (set tensorboard log file path as well)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/current/")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=run_path)

    # Save model and normalization stats
    model_dir = f"models/{run_path}"
    os.makedirs(model_dir, exist_ok=True)

    model.save(f"{model_dir}/walker_model")
    env.save(f"{model_dir}/vec_normalize.pkl")

    print(f"Finished Seed {seed}. Files saved to {model_dir}")
    env.close()


def main():
    freeze_support()
    for seed in SEEDS:
        train_seed(seed)


if __name__ == "__main__":
    main()