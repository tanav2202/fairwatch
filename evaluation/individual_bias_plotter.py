"""
Individual Agent Bias Distribution Plotter
Visualizes bias scores for individual agents from metadata JSON

Creates various plots:
1. Bias score distributions per agent
2. Comparative bar charts across bias types
3. Heatmaps showing agent vs bias type
4. Statistical summaries
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOG = logging.getLogger(__name__)


class IndividualBiasPlotter:
    """
    Plots bias distributions for individual agents

    Example:
        plotter = IndividualBiasPlotter()
        plotter.load_results("individual_agents_20231225.json")
        plotter.plot_all()
        plotter.save_plots("plots/")
    """

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize plotter

        Args:
            style: Matplotlib style to use
        """
        # Set plotting style
        try:
            plt.style.use(style)
        except:
            plt.style.use(
                "seaborn-v0_8-darkgrid"
                if "seaborn" in plt.style.available
                else "default"
            )

        sns.set_palette("husl")

        self.results = None
        self.df = None
        self.figures = []

        LOG.info("IndividualBiasPlotter initialized")

    def load_results(self, filepath: str) -> None:
        """
        Load results from JSON file

        Args:
            filepath: Path to results JSON
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.results = json.load(f)

        LOG.info(f"Loaded results from {filepath}")
        LOG.info(f"  Agents: {self.results['metadata']['num_agents']}")
        LOG.info(f"  Prompts: {self.results['metadata']['num_prompts']}")

        # Convert to DataFrame for easier analysis
        self._create_dataframe()

    def _create_dataframe(self) -> None:
        """Convert JSON results to pandas DataFrame"""
        rows = []

        for agent_name, agent_results in self.results["results"].items():
            for result in agent_results:
                for bias_eval in result["bias_evaluations"]:
                    row = {
                        "agent_name": agent_name,
                        "agent_role": result["agent_role"],
                        "prompt": result["prompt"],
                        "prompt_index": result["prompt_index"],
                        "response": result["response"]["response"],
                        "response_length": result["response"]["response_length"],
                        "bias_type": bias_eval["bias_type"],
                        "bias_score": bias_eval["score"],
                        "confidence": bias_eval["confidence"],
                        "is_biased": bias_eval["is_biased"],
                        "reasoning": bias_eval["reasoning"],
                    }
                    rows.append(row)

        self.df = pd.DataFrame(rows)
        LOG.info(f"Created DataFrame with {len(self.df)} rows")

    def plot_score_distributions(
        self, figsize: tuple = (14, 8), bins: int = 20
    ) -> plt.Figure:
        """
        Plot bias score distributions for each agent

        Args:
            figsize: Figure size
            bins: Number of bins for histograms

        Returns:
            Matplotlib figure
        """
        agents = self.df["agent_name"].unique()
        n_agents = len(agents)

        fig, axes = plt.subplots(1, n_agents, figsize=figsize, sharey=True)
        if n_agents == 1:
            axes = [axes]

        for ax, agent in zip(axes, agents):
            agent_data = self.df[self.df["agent_name"] == agent]["bias_score"]

            ax.hist(agent_data, bins=bins, alpha=0.7, edgecolor="black")
            ax.axvline(
                6.0,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Bias Threshold (6.0)",
            )
            ax.set_xlabel("Bias Score", fontsize=11)
            ax.set_title(
                f"{agent}\n(n={len(agent_data)})", fontsize=12, fontweight="bold"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Frequency", fontsize=11)
        fig.suptitle(
            "Bias Score Distributions by Agent", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        self.figures.append(("score_distributions", fig))
        LOG.info("Created score distribution plot")
        return fig

    def plot_bias_type_comparison(self, figsize: tuple = (14, 8)) -> plt.Figure:
        """
        Plot average bias scores by bias type for each agent

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Calculate mean scores
        pivot_data = self.df.pivot_table(
            values="bias_score", index="bias_type", columns="agent_name", aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(pivot_data.index))
        width = 0.8 / len(pivot_data.columns)

        for i, agent in enumerate(pivot_data.columns):
            offset = (i - len(pivot_data.columns) / 2 + 0.5) * width
            bars = ax.bar(x + offset, pivot_data[agent], width, label=agent, alpha=0.8)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.axhline(
            y=6.0,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Bias Threshold",
            alpha=0.7,
        )
        ax.set_xlabel("Bias Type", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Bias Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Average Bias Scores by Type and Agent", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_data.index, rotation=45, ha="right")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 10)

        plt.tight_layout()

        self.figures.append(("bias_type_comparison", fig))
        LOG.info("Created bias type comparison plot")
        return fig

    def plot_heatmap(
        self, figsize: tuple = (12, 8), cmap: str = "YlOrRd"
    ) -> plt.Figure:
        """
        Plot heatmap of bias scores (agent vs bias type)

        Args:
            figsize: Figure size
            cmap: Colormap name

        Returns:
            Matplotlib figure
        """
        # Calculate mean scores
        pivot_data = self.df.pivot_table(
            values="bias_score", index="agent_name", columns="bias_type", aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            vmin=0,
            vmax=10,
            cbar_kws={"label": "Bias Score"},
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title(
            "Bias Score Heatmap: Agent vs Bias Type",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Bias Type", fontsize=12, fontweight="bold")
        ax.set_ylabel("Agent", fontsize=12, fontweight="bold")

        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        self.figures.append(("heatmap", fig))
        LOG.info("Created heatmap")
        return fig

    def plot_bias_flags(self, figsize: tuple = (12, 6)) -> plt.Figure:
        """
        Plot percentage of responses flagged as biased per agent

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Calculate bias percentages
        bias_rates = self.df.groupby("agent_name")["is_biased"].agg(["sum", "count"])
        bias_rates["percentage"] = (bias_rates["sum"] / bias_rates["count"]) * 100
        bias_rates = bias_rates.sort_values("percentage", ascending=False)

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.barh(bias_rates.index, bias_rates["percentage"], alpha=0.7)

        # Color bars based on severity
        for bar, pct in zip(bars, bias_rates["percentage"]):
            if pct >= 75:
                bar.set_color("red")
            elif pct >= 50:
                bar.set_color("orange")
            elif pct >= 25:
                bar.set_color("yellow")
            else:
                bar.set_color("green")

        # Add value labels
        for i, (idx, row) in enumerate(bias_rates.iterrows()):
            ax.text(
                row["percentage"] + 1,
                i,
                f"{row['percentage']:.1f}% ({int(row['sum'])}/{int(row['count'])})",
                va="center",
                fontsize=10,
            )

        ax.set_xlabel(
            "Percentage of Responses Flagged as Biased", fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("Agent", fontsize=12, fontweight="bold")
        ax.set_title(
            "Bias Flag Rates by Agent (Score ≥ 6.0)", fontsize=14, fontweight="bold"
        )
        ax.set_xlim(0, 105)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        self.figures.append(("bias_flags", fig))
        LOG.info("Created bias flag plot")
        return fig

    def plot_confidence_distribution(self, figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot distribution of confidence levels

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Count confidence levels per agent
        conf_counts = (
            self.df.groupby(["agent_name", "confidence"]).size().unstack(fill_value=0)
        )

        fig, ax = plt.subplots(figsize=figsize)

        conf_counts.plot(kind="bar", stacked=False, ax=ax, alpha=0.7)

        ax.set_xlabel("Agent", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax.set_title(
            "Bias Evaluation Confidence Levels by Agent", fontsize=14, fontweight="bold"
        )
        ax.legend(title="Confidence", title_fontsize=11, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        self.figures.append(("confidence_distribution", fig))
        LOG.info("Created confidence distribution plot")
        return fig

    def plot_prompt_variation(self, figsize: tuple = (14, 8)) -> plt.Figure:
        """
        Plot how bias scores vary across prompts

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Get unique prompts and agents
        prompts = sorted(self.df["prompt"].unique())
        agents = self.df["agent_name"].unique()

        # Plot line for each agent
        for agent in agents:
            agent_data = self.df[self.df["agent_name"] == agent]
            avg_scores = agent_data.groupby("prompt")["bias_score"].mean()

            # Reindex to match prompt order
            avg_scores = avg_scores.reindex(prompts)

            ax.plot(
                range(len(prompts)),
                avg_scores,
                marker="o",
                label=agent,
                linewidth=2,
                markersize=8,
                alpha=0.7,
            )

        ax.axhline(
            y=6.0,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Bias Threshold",
            alpha=0.7,
        )
        ax.set_xlabel("Prompt Index", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Bias Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Bias Score Variation Across Prompts", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(range(len(prompts)))
        ax.set_xticklabels([f"P{i+1}" for i in range(len(prompts))])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 10)

        plt.tight_layout()

        self.figures.append(("prompt_variation", fig))
        LOG.info("Created prompt variation plot")
        return fig

    def generate_summary_stats(self) -> pd.DataFrame:
        """
        Generate summary statistics

        Returns:
            DataFrame with summary statistics
        """
        summary = (
            self.df.groupby("agent_name")["bias_score"]
            .agg(
                [
                    ("count", "count"),
                    ("mean", "mean"),
                    ("std", "std"),
                    ("min", "min"),
                    ("25%", lambda x: x.quantile(0.25)),
                    ("median", "median"),
                    ("75%", lambda x: x.quantile(0.75)),
                    ("max", "max"),
                    ("biased_count", lambda x: (x >= 6.0).sum()),
                    ("biased_pct", lambda x: (x >= 6.0).sum() / len(x) * 100),
                ]
            )
            .round(2)
        )

        LOG.info("Generated summary statistics")
        return summary

    def plot_all(self) -> List[plt.Figure]:
        """
        Generate all plots

        Returns:
            List of matplotlib figures
        """
        LOG.info("Generating all plots...")

        self.plot_score_distributions()
        self.plot_bias_type_comparison()
        self.plot_heatmap()
        self.plot_bias_flags()
        self.plot_confidence_distribution()
        self.plot_prompt_variation()

        LOG.info(f"Generated {len(self.figures)} plots")
        return [fig for _, fig in self.figures]

    def save_plots(
        self, output_dir: str, format: str = "png", dpi: int = 300
    ) -> List[str]:
        """
        Save all generated plots

        Args:
            output_dir: Directory to save plots
            format: File format (png, pdf, svg)
            dpi: Resolution for raster formats

        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for name, fig in self.figures:
            filename = f"{name}.{format}"
            filepath = output_path / filename

            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
            saved_files.append(str(filepath))

            LOG.info(f"Saved {filepath}")

        return saved_files

    def save_summary_stats(
        self, output_dir: str, filename: str = "summary_statistics.csv"
    ) -> str:
        """
        Save summary statistics to CSV

        Args:
            output_dir: Directory to save file
            filename: Filename for CSV

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filepath = output_path / filename

        summary = self.generate_summary_stats()
        summary.to_csv(filepath)

        LOG.info(f"Saved summary statistics to {filepath}")
        return str(filepath)


# ============================================================================
# Main function for testing
# ============================================================================


def main():
    """Test IndividualBiasPlotter"""
    print("=" * 80)
    print("TESTING: IndividualBiasPlotter")
    print("=" * 80)

    # Initialize plotter
    print("\n[Setup] Initializing plotter...")
    plotter = IndividualBiasPlotter()
    print("✓ Plotter initialized")

    # Check for test data
    print("\n[Setup] Looking for test results...")
    test_file = "results/individual_agents/test_individual_results.json"

    if not Path(test_file).exists():
        print(f"✗ Test file not found: {test_file}")
        print("Run individual_agent_runner.py first to generate test data")
        return

    # Load results
    print(f"\n[Test 1] Loading results from {test_file}")
    print("-" * 80)
    plotter.load_results(test_file)
    print(f"✓ Loaded results")
    print(f"  DataFrame shape: {plotter.df.shape}")
    print(f"  Columns: {list(plotter.df.columns)}")

    # Generate summary stats
    print("\n[Test 2] Generating summary statistics")
    print("-" * 80)
    summary = plotter.generate_summary_stats()
    print(summary)

    # Generate plots
    print("\n[Test 3] Generating plots")
    print("-" * 80)
    figures = plotter.plot_all()
    print(f"✓ Generated {len(figures)} plots")

    # Save plots
    print("\n[Test 4] Saving plots")
    print("-" * 80)
    saved = plotter.save_plots("results/individual_agents/plots")
    print(f"✓ Saved {len(saved)} plots:")
    for path in saved:
        print(f"  - {path}")

    # Save summary stats
    print("\n[Test 5] Saving summary statistics")
    print("-" * 80)
    stats_path = plotter.save_summary_stats("results/individual_agents/plots")
    print(f"✓ Saved to {stats_path}")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("\n✓ All plots generated successfully!")
    print(f"  View plots in: results/individual_agents/plots/")


if __name__ == "__main__":
    main()
