"""
evaluate_model.py

Evaluate a trained Stable-Baselines3 DQN model on ALE/MsPacman-v5.

This script:
- loads a trained model;
- creates the evaluation environment;
- runs a fixed number of evaluation episodes;
- saves episode-level metrics to a CSV file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from stable_baselines3 import DQN

from make_env import ENV_ID, make_vec_env


def ensure_directory(path: str | Path) -> None:
    """
    Create a directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_score(info: dict[str, Any]) -> Optional[float]:
    """
    Try to extract the game score from the info dictionary.

    The Atari environment does not guarantee a standard 'score' key.
    If no score is available, this function returns None.
    """
    if "score" in info:
        return float(info["score"])

    if "episode" in info and isinstance(info["episode"], dict):
        episode_info = info["episode"]

        if "score" in episode_info:
            return float(episode_info["score"])

    return None


def evaluate_trained_model(
        model_path: str,
        agent_name: str,
        episodes: int,
        seed: int,
        output_csv: str,
        deterministic: bool = True,
        render: bool = False,
) -> None:
    """
    Evaluate a trained DQN model and save the results to CSV.
    """
    ensure_directory(Path(output_csv).parent)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    env = make_vec_env(
        env_id=ENV_ID,
        seed=seed,
        n_envs=1,
        render_mode="human" if render else None,
        reward_shaping=False,
    )

    model = DQN.load(
        model_path,
        env=env,
        device="auto",
    )

    rows: list[dict[str, Any]] = []

    for episode_id in range(episodes):
        episode_seed = seed + episode_id
        env.seed(episode_seed)
        observation = env.reset()

        done = False
        reward_total = 0.0
        steps = 0
        final_info: dict[str, Any] = {}

        while not done:
            action, _state = model.predict(
                observation,
                deterministic=deterministic,
            )

            observation, reward, done_array, info_array = env.step(action)

            reward_total += float(reward[0])
            done = bool(done_array[0])
            steps += 1

            if len(info_array) > 0:
                final_info = dict(info_array[0])

        score = extract_score(final_info)

        reward_per_step = reward_total / steps if steps > 0 else None
        score_per_step = score / steps if score is not None and steps > 0 else None

        row = {
            "episode_id": episode_id,
            "agent_name": agent_name,
            "reward_total": reward_total,
            "score": score,
            "episode_length": steps,
            "steps": steps,
            "score_per_step": score_per_step,
            "reward_per_step": reward_per_step,
            "lives_lost": final_info.get("lives_lost"),
            "seed": seed + episode_id,
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        rows.append(row)

        print(
            f"Episode {episode_id} | "
            f"agent={agent_name} | "
            f"reward_total={reward_total:.2f} | "
            f"steps={steps}"
        )

    env.close()

    results = pd.DataFrame(rows)
    results.to_csv(output_csv, index=False)

    print(f"Results saved to: {output_csv}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN model on ALE/MsPacman-v5."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file.",
    )

    parser.add_argument(
        "--agent-name",
        type=str,
        default="dqn",
        help="Name of the evaluated agent.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
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
        default="results/dqn_results.csv",
        help="Path where the CSV results file will be saved.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic actions.",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment while evaluating.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate_trained_model(
        model_path=args.model_path,
        agent_name=args.agent_name,
        episodes=args.episodes,
        seed=args.seed,
        output_csv=args.output_csv,
        deterministic=not args.stochastic,
        render=args.render,
    )
