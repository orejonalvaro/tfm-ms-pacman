"""
analyze_results.py

Analyze evaluation results for the Ms. Pac-Man reinforcement learning agents.

This script:
- loads CSV result files;
- combines them into a single DataFrame;
- computes summary statistics by agent;
- saves the combined results and summary table to CSV;
- generates comparison plots with error bars;
- generates episode-level plots.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_RESULT_FILES = [
    "results/random_seed123_results.csv",
    "results/random_seed456_results.csv",
    "results/random_seed789_results.csv",

    "results/dqn_500k_seed123_results.csv",
    "results/dqn_500k_seed456_results.csv",
    "results/dqn_500k_seed789_results.csv",

    "results/dqn_reward_500k_alpha001_seed123_results.csv",
    "results/dqn_reward_500k_alpha001_seed456_results.csv",
    "results/dqn_reward_500k_alpha001_seed789_results.csv",
]


def ensure_directory(path: str | Path) -> None:
    """
    Create a directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_result_files(result_files: list[str]) -> pd.DataFrame:
    """
    Load and combine several result CSV files.

    Parameters
    ----------
    result_files:
        List of CSV result file paths.

    Returns
    -------
    pd.DataFrame
        Combined results from all available files.
    """
    dataframes = []

    for file_path in result_files:
        path = Path(file_path)

        if not path.exists():
            print(f"Warning: file not found and skipped: {file_path}")
            continue

        df = pd.read_csv(path)
        df["source_file"] = str(path)
        dataframes.append(df)

        print(f"Loaded: {file_path} | rows={len(df)}")

    if not dataframes:
        raise FileNotFoundError("No valid result CSV files were found.")

    combined_results = pd.concat(dataframes, ignore_index=True)

    return combined_results


def validate_required_columns(results: pd.DataFrame) -> None:
    """
    Validate that the required columns exist in the results DataFrame.
    """
    required_columns = [
        "episode_id",
        "agent_name",
        "reward_total",
        "steps",
        "reward_per_step",
    ]

    missing_columns = [
        column for column in required_columns if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def prepare_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns used for fairer comparisons.

    This groups different training seeds under a common name:
    - dqn_500k_seed123 -> dqn
    - dqn_reward_500k_alpha001_seed123 -> dqn_reward
    - random_seed123 -> random

    The primary comparison metric is performance_total. It uses game score when
    available and falls back to reward_total when the Atari score is not exposed.
    """
    validate_required_columns(results)

    results = results.copy()

    def map_agent_name(agent_name: str) -> str:
        if agent_name.startswith("dqn_reward"):
            return "dqn_reward"

        if agent_name.startswith("dqn"):
            return "dqn"

        if agent_name.startswith("random"):
            return "random"

        return agent_name

    def extract_training_seed(agent_name: str) -> int | None:
        match = re.search(r"seed(\d+)", agent_name)
        return int(match.group(1)) if match else None

    def extract_timesteps(agent_name: str) -> int | None:
        match = re.search(r"_(\d+)k_", agent_name)
        return int(match.group(1)) * 1_000 if match else None

    def extract_reward_alpha(agent_name: str) -> float | None:
        match = re.search(r"alpha(\d+)", agent_name)
        if not match:
            return None

        digits = match.group(1)
        return int(digits) / (10 ** len(digits))

    results["agent_group"] = results["agent_name"].apply(map_agent_name)
    results["training_seed"] = results["agent_name"].apply(extract_training_seed)
    results["training_timesteps"] = results["agent_name"].apply(extract_timesteps)
    results["reward_alpha"] = results["agent_name"].apply(extract_reward_alpha)

    if "score" not in results.columns:
        results["score"] = pd.NA

    if "score_per_step" not in results.columns:
        results["score_per_step"] = pd.NA

    results["score"] = pd.to_numeric(results["score"], errors="coerce")
    results["score_per_step"] = pd.to_numeric(results["score_per_step"], errors="coerce")
    results["performance_total"] = results["score"].fillna(results["reward_total"])
    results["performance_per_step"] = results["score_per_step"].fillna(results["reward_per_step"])
    results["evaluation_seed"] = results["seed"] if "seed" in results.columns else pd.NA

    return results


def add_summary_statistics(grouped_results) -> pd.DataFrame:
    """
    Compute robust summary statistics for grouped episode results.
    """
    summary = (
        grouped_results
        .agg(
            episodes=("episode_id", "count"),
            performance_mean=("performance_total", "mean"),
            performance_median=("performance_total", "median"),
            performance_std=("performance_total", "std"),
            performance_min=("performance_total", "min"),
            performance_max=("performance_total", "max"),
            performance_per_step_mean=("performance_per_step", "mean"),
            performance_per_step_median=("performance_per_step", "median"),
            performance_per_step_std=("performance_per_step", "std"),
            score_available_episodes=("score", "count"),
            score_mean=("score", "mean"),
            score_median=("score", "median"),
            score_std=("score", "std"),
            reward_mean=("reward_total", "mean"),
            reward_median=("reward_total", "median"),
            reward_std=("reward_total", "std"),
            reward_min=("reward_total", "min"),
            reward_max=("reward_total", "max"),
            steps_mean=("steps", "mean"),
            steps_median=("steps", "median"),
            steps_std=("steps", "std"),
            reward_per_step_mean=("reward_per_step", "mean"),
            reward_per_step_median=("reward_per_step", "median"),
            reward_per_step_std=("reward_per_step", "std"),
        )
        .reset_index()
    )

    summary["performance_sem"] = summary["performance_std"] / summary["episodes"].pow(0.5)
    summary["performance_ci95"] = 1.96 * summary["performance_sem"]
    summary["performance_per_step_sem"] = (
        summary["performance_per_step_std"] / summary["episodes"].pow(0.5)
    )
    summary["performance_per_step_ci95"] = 1.96 * summary["performance_per_step_sem"]
    summary = summary.fillna(0.0)

    return summary


def compute_group_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics grouped by agent family.
    """
    return add_summary_statistics(results.groupby("agent_group"))


def compute_run_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics grouped by individual trained model.
    """
    return add_summary_statistics(
        results.groupby(["agent_group", "agent_name", "training_seed"], dropna=False)
    )


def compute_paired_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compare DQN reward shaping against baseline DQN on matching evaluation seeds.
    """
    comparable = results[results["agent_group"].isin(["dqn", "dqn_reward"])].copy()

    if comparable.empty or "evaluation_seed" not in comparable.columns:
        return pd.DataFrame()

    per_seed = (
        comparable
        .groupby(["agent_group", "evaluation_seed"], dropna=False)["performance_total"]
        .mean()
        .reset_index()
    )

    paired = per_seed.pivot(
        index="evaluation_seed",
        columns="agent_group",
        values="performance_total",
    ).reset_index()

    if "dqn" not in paired.columns or "dqn_reward" not in paired.columns:
        return pd.DataFrame()

    paired = paired.dropna(subset=["dqn", "dqn_reward"])
    paired["dqn_reward_minus_dqn"] = paired["dqn_reward"] - paired["dqn"]
    paired["dqn_reward_pct_change"] = (
        paired["dqn_reward_minus_dqn"] / paired["dqn"].replace(0, pd.NA) * 100
    )

    return paired


def save_dataframe(dataframe: pd.DataFrame, output_csv: str) -> None:
    """
    Save a DataFrame to CSV.
    """
    ensure_directory(Path(output_csv).parent)
    dataframe.to_csv(output_csv, index=False)
    print(f"CSV saved to: {output_csv}")


def plot_bar_chart_with_error(
        summary: pd.DataFrame,
        x_column: str,
        y_column: str,
        error_column: str,
        title: str,
        y_label: str,
        output_path: str,
) -> None:
    """
    Generate and save a bar chart with standard deviation error bars.
    """
    ensure_directory(Path(output_path).parent)

    plot_data = summary.sort_values(y_column, ascending=False)

    plt.figure(figsize=(8, 5))

    plt.bar(
        plot_data[x_column],
        plot_data[y_column],
        yerr=plot_data[error_column],
        capsize=6,
    )

    plt.title(title)
    plt.xlabel("Agent")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figure saved to: {output_path}")


def plot_boxplot(
        results: pd.DataFrame,
        metric_column: str,
        title: str,
        y_label: str,
        output_path: str,
) -> None:
    """
    Generate and save a distribution plot for each agent group.
    """
    ensure_directory(Path(output_path).parent)

    plot_data = [
        agent_data[metric_column].dropna()
        for _, agent_data in results.groupby("agent_group")
    ]
    labels = [
        agent_name
        for agent_name, _ in results.groupby("agent_group")
    ]

    if not plot_data:
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(plot_data, tick_labels=labels, showmeans=True)
    plt.title(title)
    plt.xlabel("Agent")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figure saved to: {output_path}")


def plot_run_metric(
        run_summary: pd.DataFrame,
        metric_column: str,
        title: str,
        y_label: str,
        output_path: str,
) -> None:
    """
    Generate and save a per-model plot to inspect seed-to-seed variability.
    """
    ensure_directory(Path(output_path).parent)

    plot_data = run_summary.sort_values(["agent_group", "training_seed"])

    plt.figure(figsize=(10, 5))
    plt.bar(plot_data["agent_name"], plot_data[metric_column])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(y_label)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figure saved to: {output_path}")


def plot_episode_metric(
        results: pd.DataFrame,
        metric_column: str,
        title: str,
        y_label: str,
        output_path: str,
) -> None:
    """
    Generate an episode-level plot using the mean per agent group and episode.
    """
    ensure_directory(Path(output_path).parent)

    plt.figure(figsize=(9, 5))

    grouped = (
        results
        .groupby(["agent_group", "episode_id"])[metric_column]
        .agg(["mean", "std"])
        .reset_index()
    )

    for agent_name, agent_data in grouped.groupby("agent_group"):
        agent_data = agent_data.sort_values("episode_id")

        plt.plot(
            agent_data["episode_id"],
            agent_data["mean"],
            marker="o",
            label=agent_name,
        )
        lower = agent_data["mean"] - agent_data["std"].fillna(0.0)
        upper = agent_data["mean"] + agent_data["std"].fillna(0.0)
        plt.fill_between(agent_data["episode_id"], lower, upper, alpha=0.15)

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figure saved to: {output_path}")


def generate_figures(
        results: pd.DataFrame,
        group_summary: pd.DataFrame,
        run_summary: pd.DataFrame,
        figures_dir: str,
) -> None:
    """
    Generate comparison and episode-level figures.
    """
    ensure_directory(figures_dir)

    plot_bar_chart_with_error(
        summary=group_summary,
        x_column="agent_group",
        y_column="performance_mean",
        error_column="performance_ci95",
        title="Mean performance by agent",
        y_label="Mean score or reward",
        output_path=str(Path(figures_dir) / "comparison_performance.png"),
    )

    plot_bar_chart_with_error(
        summary=group_summary,
        x_column="agent_group",
        y_column="performance_per_step_mean",
        error_column="performance_per_step_ci95",
        title="Mean performance per step by agent",
        y_label="Mean score/reward per step",
        output_path=str(Path(figures_dir) / "comparison_efficiency.png"),
    )

    plot_bar_chart_with_error(
        summary=group_summary,
        x_column="agent_group",
        y_column="steps_mean",
        error_column="steps_std",
        title="Mean episode length by agent",
        y_label="Mean steps",
        output_path=str(Path(figures_dir) / "comparison_episode_length.png"),
    )

    plot_boxplot(
        results=results,
        metric_column="performance_total",
        title="Performance distribution by agent",
        y_label="Score or reward",
        output_path=str(Path(figures_dir) / "performance_distribution.png"),
    )

    plot_run_metric(
        run_summary=run_summary,
        metric_column="performance_mean",
        title="Mean performance by trained model",
        y_label="Mean score or reward",
        output_path=str(Path(figures_dir) / "performance_by_model.png"),
    )

    plot_episode_metric(
        results=results,
        metric_column="performance_total",
        title="Performance by episode",
        y_label="Score or reward",
        output_path=str(Path(figures_dir) / "performance_by_episode.png"),
    )

    plot_episode_metric(
        results=results,
        metric_column="reward_per_step",
        title="Reward per step by episode",
        y_label="Reward per step",
        output_path=str(Path(figures_dir) / "reward_per_step_by_episode.png"),
    )


def print_summary(summary: pd.DataFrame) -> None:
    """
    Print the summary table in the console.
    """
    display_columns = [
        "agent_group",
        "episodes",
        "performance_mean",
        "performance_median",
        "performance_std",
        "performance_ci95",
        "reward_mean",
        "steps_mean",
        "performance_per_step_mean",
    ]
    existing_columns = [column for column in display_columns if column in summary.columns]

    print("\nSummary by agent group:")
    print(summary[existing_columns].to_string(index=False))


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyze Ms. Pac-Man agent evaluation results."
    )

    parser.add_argument(
        "--result-files",
        nargs="+",
        default=DEFAULT_RESULT_FILES,
        help="CSV result files to analyze.",
    )

    parser.add_argument(
        "--combined-csv",
        type=str,
        default="results/combined_results.csv",
        help="Path where the combined episode-level CSV will be saved.",
    )

    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/summary_results.csv",
        help="Path where the group summary CSV will be saved.",
    )

    parser.add_argument(
        "--run-summary-csv",
        type=str,
        default="results/run_summary_results.csv",
        help="Path where the per-model summary CSV will be saved.",
    )

    parser.add_argument(
        "--paired-comparison-csv",
        type=str,
        default="results/paired_comparison_results.csv",
        help="Path where the paired DQN vs DQN reward comparison CSV will be saved.",
    )

    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures",
        help="Directory where figures will be saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results_df = load_result_files(args.result_files)
    results_df = prepare_results(results_df)

    group_summary_df = compute_group_summary(results_df)
    run_summary_df = compute_run_summary(results_df)
    paired_comparison_df = compute_paired_comparison(results_df)

    print_summary(group_summary_df)

    save_dataframe(results_df, args.combined_csv)
    save_dataframe(group_summary_df, args.summary_csv)
    save_dataframe(run_summary_df, args.run_summary_csv)

    if not paired_comparison_df.empty:
        save_dataframe(paired_comparison_df, args.paired_comparison_csv)

    generate_figures(
        results=results_df,
        group_summary=group_summary_df,
        run_summary=run_summary_df,
        figures_dir=args.figures_dir,
    )
