"""
make_env.py

Common environment creation utilities for the Ms. Pas-Man reinforcement learning experiments.

This fiel centralizes:
- ALE environment registration.
- Global seed setup.
- Single Gymnasium environment creation.
- Atari preprocessing.
- Stable-Baselines3 vectorized environment creation.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Optional

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)

from wrappers.reward_wrapper import RewardShapingWrapper

ENV_ID = "ALE/MsPacman-v5"


def register_ale_environment() -> None:
    """
    Register ALE environments in Gymnasium.
    """
    gym.register_envs(ale_py)


def set_global_seed(seed: int) -> None:
    """
    Set basic random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_single_env(
        env_id: str = ENV_ID,
        seed: int = 123,
        render_mode: Optional[str] = None,
        apply_atari_preprocessing: bool = True,
        reward_shaping: bool = False,
        step_penalty_alpha: float = 0.01,
        life_loss_penalty_beta: float = 1.0,
) -> gym.Env:
    """
    Create a single Ms. Pac-Man Gymnasium environment.
    """
    register_ale_environment()
    set_global_seed(seed)

    try:
        env = gym.make(
            env_id,
            render_mode=render_mode,
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=False,
        )
    except:
        # Fallback for versions that may not accept all environment arguments.
        env = gym.make(env_id, render_mode=render_mode)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    if apply_atari_preprocessing:
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=True,
            scale_obs=False,
        )

    if reward_shaping:
        env = RewardShapingWrapper(
            env,
            step_penalty_alpha=step_penalty_alpha,
            life_loss_penalty_beta=life_loss_penalty_beta,
        )

    return env


def make_vec_env(env_id: str = ENV_ID,
                 seed: int = 123,
                 n_envs: int = 1,
                 render_mode: Optional[str] = None,
                 reward_shaping: bool = False,
                 step_penalty_alpha: float = 0.01,
                 life_loss_penalty_beta: float = 1.0):
    """
    Create a vectorized environment compatible with Stable-Baselines3.
    """

    def make_env_fn(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = make_single_env(
                env_id=env_id,
                seed=seed + rank,
                render_mode=render_mode,
                apply_atari_preprocessing=True,
                reward_shaping=reward_shaping,
                step_penalty_alpha=step_penalty_alpha,
                life_loss_penalty_beta=life_loss_penalty_beta,
            )
            return env

        return _init

    env = DummyVecEnv([make_env_fn(i) for i in range(n_envs)])
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    return env
