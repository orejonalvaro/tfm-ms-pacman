"""
evaluate_random.py

Evaluate a random agent on ALE/MsPacman-v5

This script runs a fixed number of episodes using 
random actions and saves episode-level metrics to a CSV file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from make_env import ENV_ID, make_single_env

def ensure_directory(path: str | Path) -> None:
    """
    Create a directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def extract_score(info: dict[str, Any]) -> Optional[float]:
    """
    Try to extract the game score from the info dictionary.

    The Atari environment does not guarantee a standard score key.
    If no score is available, this function returns None.
    """

    if "score" in info:
        return float(info["score"])

    if "episode" in info and isinstance(info["episode"], dict):
        episode_info = info["episode"]

        if "score" in episode_info:
            return float(episode_info["score"])
        
    return None

def evaluate_random_agent(episodes: int, seed: int, output_csv: str, render: bool = False) -> None:
    """
    Evaluate a random agent and save the results to a CSV file.
    """
    ensure_directory(Path(output_csv).parent)

    env = make_single_env(
        env_id=ENV_ID,
        seed=seed,
        render_mode="human" if render else None,
        apply_atari_preprocessing=True,
    )

    rows: list[dict[str, Any]] = []

    for episode_id in range(episodes):
        episode_seed = seed + episode_id
        observation, info = env.reset(seed=episode_seed)
        terminated = False
        truncated = False
        reward_total = 0.0
        steps = 0
        final_info: dict[str, Any] = {}

        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            steps += 1
            final_info = dict(info)

        score = extract_score(final_info)

        reward_per_step = reward_total / steps if steps > 0 else None
        score_per_step = score / steps if score is not None and steps > 0 else None

        row = {
            "episode_id": episode_id,
            "agent_name": "random",
            "reward_total": reward_total,
            "score": score,
            "episode_length": steps,
            "steps": steps,
            "score_per_step": score_per_step,
            "reward_per_step": reward_per_step,
            "lives_lost": final_info.get("lives_lost"),
            "seed": episode_seed,
            "model_path": None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        rows.append(row)
        print(
            f"Episode {episode_id} | "
            f"reward_total={reward_total:.2f} | "
            f"steps={steps}"
        )

    env.close()

    results = pd.DataFrame(rows)
    results.to_csv(output_csv, index=False)

    print(f"Results saved to: {output_csv}")

def parse_args() -> argparse.Namespace:
    """"
    Parse comamand-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a random agent on ALE/MsPacman-v5")

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/random_results.csv",
        help="Path where the CSV results file will be saved.",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment while evaluating.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_random_agent(
        episodes=args.episodes,
        seed=args.seed,
        output_csv=args.output_csv,
        render=args.render,
    )