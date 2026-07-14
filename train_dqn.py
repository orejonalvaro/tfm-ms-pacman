"""
train_dqn.py

Train a baseline DQN agent on ALE/MsPacman-v5 using Stable-Baselines3.

This script:
- reads an optional YAML configuration file;
- creates the vectorized Ms. Pac-Man environment;
- trains a DQN model with CnnPolicy;
- saves the trained model;
- saves TensorBoard logs;
- saves periodic checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from make_env import ENV_ID, make_vec_env, set_global_seed


def ensure_directory(path: str | Path) -> None:
    """
    Create a directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | None) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    If no configuration path is provided, an empty dictionary is returned.
    """
    if config_path is None:
        return {}

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    return config


def get_config_value(
        config: dict[str, Any],
        section: str,
        key: str,
        default: Any,
) -> Any:
    """
    Safely get a value from a nested configuration section.
    """
    return config.get(section, {}).get(key, default)


def train_dqn(
        total_timesteps: int,
        seed: int,
        model_path: str,
        tensorboard_log: str,
        checkpoint_dir: str,
        learning_rate: float,
        buffer_size: int,
        learning_starts: int,
        batch_size: int,
        tau: float,
        gamma: float,
        train_freq: int,
        gradient_steps: int,
        target_update_interval: int,
        exploration_fraction: float,
        exploration_initial_eps: float,
        exploration_final_eps: float,
        device: str,
) -> None:
    """
    Train a baseline DQN model.
    """
    ensure_directory(Path(model_path).parent)
    ensure_directory(tensorboard_log)
    ensure_directory(checkpoint_dir)

    set_global_seed(seed)

    env = make_vec_env(
        env_id=ENV_ID,
        seed=seed,
        n_envs=1,
        reward_shaping=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=checkpoint_dir,
        name_prefix="dqn_mspacman",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=seed,
        device=device,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="dqn_mspacman",
        progress_bar=True,
    )

    model.save(model_path)
    env.close()

    print(f"Model saved to: {model_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a baseline DQN agent on ALE/MsPacman-v5."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/dqn.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total number of training timesteps. Overrides YAML config if provided.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Overrides YAML config if provided.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path where the trained model will be saved. Overrides YAML config if provided.",
    )

    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default=None,
        help="Directory for TensorBoard logs. Overrides YAML config if provided.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for model checkpoints. Overrides YAML config if provided.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_data = load_config(args.config)

    total_timesteps = (
        args.total_timesteps
        if args.total_timesteps is not None
        else get_config_value(config_data, "training", "total_timesteps_test", 10_000)
    )

    seed = (
        args.seed
        if args.seed is not None
        else get_config_value(config_data, "training", "seed", 123)
    )

    model_path = (
        args.model_path
        if args.model_path is not None
        else get_config_value(config_data, "paths", "model_path", "models/dqn_mspacman.zip")
    )

    tensorboard_log = (
        args.tensorboard_log
        if args.tensorboard_log is not None
        else get_config_value(config_data, "paths", "tensorboard_log", "logs/tensorboard")
    )

    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else get_config_value(config_data, "paths", "checkpoint_dir", "logs/checkpoints")
    )

    train_dqn(
        total_timesteps=total_timesteps,
        seed=seed,
        model_path=model_path,
        tensorboard_log=tensorboard_log,
        checkpoint_dir=checkpoint_dir,
        learning_rate=get_config_value(config_data, "model", "learning_rate", 1e-4),
        buffer_size=get_config_value(config_data, "model", "buffer_size", 50_000),
        learning_starts=get_config_value(config_data, "model", "learning_starts", 1_000),
        batch_size=get_config_value(config_data, "model", "batch_size", 32),
        tau=get_config_value(config_data, "model", "tau", 1.0),
        gamma=get_config_value(config_data, "model", "gamma", 0.99),
        train_freq=get_config_value(config_data, "model", "train_freq", 4),
        gradient_steps=get_config_value(config_data, "model", "gradient_steps", 1),
        target_update_interval=get_config_value(
            config_data,
            "model",
            "target_update_interval",
            1_000,
        ),
        exploration_fraction=get_config_value(
            config_data,
            "model",
            "exploration_fraction",
            0.10,
        ),
        exploration_initial_eps=get_config_value(
            config_data,
            "model",
            "exploration_initial_eps",
            1.0,
        ),
        exploration_final_eps=get_config_value(
            config_data,
            "model",
            "exploration_final_eps",
            0.05,
        ),
        device=get_config_value(config_data, "training", "device", "auto"),
    )
