import argparse
import datetime
import json
import os
from typing import Tuple

import gymnasium as gym
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ENV_ID = "BipedalWalker-v3"


def make_env():
    """Create environment configured for RGB array rendering (no UI window)."""

    return gym.make(ENV_ID, render_mode="rgb_array")


def build_output_dir(model_dir: str, output_root: str) -> str:
    """Mirror the models/ layout under a gifs/ root.

    Example:
        model_dir = "models/PPO_Normalized/20251223-2256/Seed_5"
        output_root = "gifs"
    ->  gifs/PPO_Normalized/20251223-2256/Seed_5
    """

    # If the path starts with "models", strip that so we only keep the experiment path
    rel = os.path.relpath(model_dir, "models") if model_dir.startswith("models") else model_dir
    out_dir = os.path.join(output_root, rel)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_env_and_model(model_dir: str):
    """Load VecNormalize stats (if present) and the PPO model."""

    env = DummyVecEnv([make_env])

    stats_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: No normalization stats found. Running without VecNormalize.")

    model_path = os.path.join(model_dir, "walker_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = PPO.load(model_path, env=env)
    return env, model, stats_path if os.path.exists(stats_path) else None


def _get_frame(env) -> "object":
    """Render a single RGB frame from a VecEnv.

    VecEnv.render(mode="rgb_array") may return a list of frames (one per sub-env)
    or a single array. We always return a single HxWxC array.
    """

    frame = env.render(mode="rgb_array")
    if isinstance(frame, (list, tuple)):
        frame = frame[0]
    return frame


def record_gifs(model_dir: str, output_root: str, episodes: int, max_steps: int, fps: int) -> str:
    env, model, stats_path = load_env_and_model(model_dir)
    out_dir = build_output_dir(model_dir, output_root)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    for ep in range(1, episodes + 1):
        obs = env.reset()
        frames = []

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)

            frame = _get_frame(env)
            frames.append(frame)

            # dones is a vector for VecEnv; stop when first env is done
            if bool(dones[0]):
                break

        gif_name = f"episode_{ep:03d}.gif"
        gif_path = os.path.join(out_dir, gif_name)
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Saved GIF: {gif_path}")

    metadata = {
        "model_dir": model_dir,
        "vecnormalize_path": stats_path,
        "episodes": episodes,
        "max_steps": max_steps,
        "fps": fps,
        "env_id": ENV_ID,
        "created_at": timestamp,
    }
    with open(os.path.join(out_dir, "gif_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    env.close()
    return out_dir


def parse_args() -> Tuple[str, str, int, int, int]:
    parser = argparse.ArgumentParser(description="Record GIF(s) of a trained BipedalWalker PPO run.")
    parser.add_argument(
        "model_dir",
        help="Path to a trained model folder, e.g. models/PPO_Normalized/20251223-2256/Seed_5",
    )
    parser.add_argument(
        "--output-root",
        default="gifs",
        help="Root folder where GIFs are stored (mirrors the models/ structure).",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum number of steps per episode before stopping.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second of the output GIF.")

    args = parser.parse_args()
    return args.model_dir, args.output_root, args.episodes, args.max_steps, args.fps


def main() -> None:
    model_dir, output_root, episodes, max_steps, fps = parse_args()
    out_dir = record_gifs(model_dir, output_root, episodes, max_steps, fps)
    print(f"All GIFs and metadata written to: {out_dir}")


if __name__ == "__main__":
    main()
