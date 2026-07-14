"""
reward_wrapper.py

Reward shaping wrapper for the Ms. Pac-Man reinforcement learning experiments.

The wrapper modifies the original environment reward using.

new_reward = original_reward - step_penalty_alpha  - life_loss_penalty_beta  * lives_lost_step
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    """
    Apply reward shaping to the environment.

    Params:
        env: Gymnasium environment to wrap.
        step_penalty_alpha : Small penalty applied at every environment step.
        life_loss_penalty_beta: Penalty applied when a life loss is detected.
    """

    def __init__(self, env: gym.Env, step_penalty_alpha: float = 0.01, life_loss_penalty_beta: float = 1.0) -> None:
        super().__init__(env)

        if step_penalty_alpha < 0:
            raise ValueError("step_penalty_alpha must be greater than or equal to 0.")

        if life_loss_penalty_beta < 0:
            raise ValueError("life_loss_penalty_beta must be greater than or equal to 0.")

        self.step_penalty_alpha = step_penalty_alpha
        self.life_loss_penalty_beta = life_loss_penalty_beta

        self.previous_lives: Optional[int] = None
        self.total_lives_lost = 0
        self.life_detection_available = False

    def _get_lives(self) -> int:
        """
        Try to get the current number of lives from ALE

        Returns:
            Optional[int]: Current number of lives if available, otherwise None.
        """
        try:
            lives = self.env.unwrapped.ale.lives()
            return int(lives)
        except Exception:
            return None

    def reset(self, **kwargs: Any):
        """
        Reset the environment and initialize life tracking.
        """
        observation, info = self.env.reset(**kwargs)
        current_lives = self._get_lives()

        self.previous_lives = current_lives
        self.total_lives_lost = 0
        self.life_detection_available = current_lives is not None

        info = dict(info)
        info["life_detection_available"] = self.life_detection_available
        info["current_lives"] = current_lives
        info["lives_lost"] = self.total_lives_lost

        return observation, info

    def step(self, action):
        """
        Execute one environment step and modify the reward.
        """
        observation, original_reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        current_lives = self._get_lives()
        lives_lost_step = 0

        if self.previous_lives is not None and current_lives is not None:
            if current_lives < self.previous_lives:
                lives_lost_step = self.previous_lives - current_lives
                self.total_lives_lost += lives_lost_step

        if current_lives is not None:
            self.previous_lives = current_lives

        shaped_reward = (
                    float(original_reward) - self.step_penalty_alpha - self.life_loss_penalty_beta * lives_lost_step)

        info["original_reward"] = float(original_reward)
        info["shaped_reward"] = float(shaped_reward)
        info["step_penalty_alpha"] = self.step_penalty_alpha
        info["life_loss_penalty_beta"] = self.life_loss_penalty_beta
        info["life_detection_available"] = current_lives is not None
        info["current_lives"] = current_lives
        info["lives_lost_step"] = lives_lost_step
        info["lives_lost"] = self.total_lives_lost

        return observation, shaped_reward, terminated, truncated, info
