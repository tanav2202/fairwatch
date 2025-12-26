"""
Chain Bias Distribution Plotter
Visualizes bias evolution in multi-agent conversation chains

Creates various plots:
1. Bias evolution across chain turns
2. Comparison of different agent orderings
3. Heatmaps showing bias propagation
4. Turn-by-turn agent contribution to bias
5. Statistical summaries of bias emergence
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


class ChainBiasPlotter:
    """
    Plots bias evolution in conversation chains

    Example:
        plotter = ChainBiasPlotter()
        plotter.load_results("chain_results_20231225.json")
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

        LOG.info("ChainBiasPlotter initialized")

    def load_results(self, filepath: str) -> None:
        """
        Load results from JSON file

        Args:
            filepath: Path to results JSON
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.results = json.load(f)

        LOG.info(f"Loaded results from {filepath}")
        LOG.info(f"  Chains: {self.results['metadata']['num_chains']}")
        LOG.info(f"  Prompts: {self.results['metadata']['num_prompts']}")
        LOG.info(f"  Orderings: {self.results['metadata']['num_orderings']}")

        # Convert to DataFrame for easier analysis
        self._create_dataframe()

    def _create_dataframe(self) -> None:
        """Convert JSON results to pandas DataFrame"""
        rows = []

        for chain_result in self.results["results"]:
            chain_id = f"{' -> '.join(chain_result['agent_sequence'])}"

            for turn_eval in chain_result["turn_bias_evaluations"]:
                for bias_eval in turn_eval["bias_evaluations"]:
                    row = {
                        "chain_id": chain_id,
                        "agent_sequence": " -> ".join(chain_result["agent_sequence"]),
                        "prompt": chain_result["prompt"],
                        "prompt_index": chain_result["prompt_index"],
                        "chain_length": chain_result["chain_length"],
                        "turn_number": turn_eval["turn_number"],
                        "agent_name": turn_eval["agent_name"],
                        "agent_role": turn_eval["agent_role"],
                        "chain_position": turn_eval["chain_position"],
                        "previous_agents": (
                            ", ".join(turn_eval["previous_agents"])
                            if turn_eval["previous_agents"]
                            else "None"
                        ),
                        "bias_type": bias_eval["bias_type"],
                        "bias_score": bias_eval["score"],
                        "confidence": bias_eval["confidence"],
                        "is_biased": bias_eval["is_biased"],
                        "reasoning": bias_eval["reasoning"],
                    }
                    rows.append(row)

        self.df = pd.DataFrame(rows)
        LOG.info(f"Created DataFrame with {len(self.df)} rows")

    def plot_bias_evolution(
        self, figsize: tuple = (14, 8), by_bias_type: bool = True
    ) -> plt.Figure:
        """
        Plot how bias evolves across chain turns

        Args:
            figsize: Figure size
            by_bias_type: If True, separate lines for each bias type

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if by_bias_type:
            # One line per bias type
            for bias_type in self.df["bias_type"].unique():
                type_data = self.df[self.df["bias_type"] == bias_type]
                avg_by_turn = type_data.groupby("turn_number")["bias_score"].mean()

                ax.plot(
                    avg_by_turn.index,
                    avg_by_turn.values,
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    label=bias_type,
                    alpha=0.7,
                )
        else:
            # Single line averaging all bias types
            avg_by_turn = self.df.groupby("turn_number")["bias_score"].mean()
            ax.plot(
                avg_by_turn.index,
                avg_by_turn.values,
                marker="o",
                linewidth=3,
                markersize=10,
                label="Average across all bias types",
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
        ax.set_xlabel("Turn Number in Chain", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Bias Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Bias Score Evolution Across Chain Turns", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 10)

        plt.tight_layout()

        self.figures.append(("bias_evolution", fig))
        LOG.info("Created bias evolution plot")
        return fig

    def plot_agent_contribution(self, figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Plot average bias score by agent and position in chain

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Calculate average bias by agent
        agent_avg = (
            self.df.groupby("agent_name")["bias_score"]
            .mean()
            .sort_values(ascending=False)
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Average bias by agent
        bars = ax1.barh(agent_avg.index, agent_avg.values, alpha=0.7)

        # Color bars by severity
        for bar, score in zip(bars, agent_avg.values):
            if score >= 6.0:
                bar.set_color("red")
            elif score >= 4.0:
                bar.set_color("orange")
            else:
                bar.set_color("green")

        ax1.axvline(x=6.0, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax1.set_xlabel("Average Bias Score", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Agent", fontsize=11, fontweight="bold")
        ax1.set_title("Average Bias Score by Agent", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="x")

        # Plot 2: Bias by chain position
        position_data = (
            self.df.groupby(["agent_name", "chain_position"])["bias_score"]
            .mean()
            .unstack(fill_value=0)
        )

        position_data.plot(kind="bar", ax=ax2, alpha=0.7)
        ax2.axhline(
            y=6.0,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Bias Threshold",
        )
        ax2.set_xlabel("Agent", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Average Bias Score", fontsize=11, fontweight="bold")
        ax2.set_title(
            "Bias Score by Agent and Chain Position", fontsize=12, fontweight="bold"
        )
        ax2.legend(title="Position", title_fontsize=10)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        self.figures.append(("agent_contribution", fig))
        LOG.info("Created agent contribution plot")
        return fig

    def plot_ordering_comparison(self, figsize: tuple = (14, 8)) -> plt.Figure:
        """
        Compare bias evolution across different agent orderings

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot each ordering separately
        for sequence in self.df["agent_sequence"].unique():
            seq_data = self.df[self.df["agent_sequence"] == sequence]
            avg_by_turn = seq_data.groupby("turn_number")["bias_score"].mean()

            ax.plot(
                avg_by_turn.index,
                avg_by_turn.values,
                marker="o",
                linewidth=2,
                markersize=8,
                label=sequence,
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
        ax.set_xlabel("Turn Number in Chain", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Bias Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Bias Evolution: Comparing Different Agent Orderings",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 10)

        plt.tight_layout()

        self.figures.append(("ordering_comparison", fig))
        LOG.info("Created ordering comparison plot")
        return fig

    def plot_bias_propagation_heatmap(
        self, figsize: tuple = (12, 8), cmap: str = "YlOrRd"
    ) -> plt.Figure:
        """
        Heatmap showing bias scores across turns and bias types

        Args:
            figsize: Figure size
            cmap: Colormap name

        Returns:
            Matplotlib figure
        """
        # Create pivot table: turn_number x bias_type
        pivot_data = self.df.pivot_table(
            values="bias_score",
            index="turn_number",
            columns="bias_type",
            aggfunc="mean",
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
            "Bias Propagation Heatmap: Turn vs Bias Type",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Bias Type", fontsize=12, fontweight="bold")
        ax.set_ylabel("Turn Number", fontsize=12, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        self.figures.append(("propagation_heatmap", fig))
        LOG.info("Created propagation heatmap")
        return fig

    def plot_bias_amplification(self, figsize: tuple = (14, 6)) -> plt.Figure:
        """
        Plot whether bias increases or decreases through the chain

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Calculate first vs last turn bias
        first_turn = (
            self.df[self.df["turn_number"] == 1]
            .groupby("bias_type")["bias_score"]
            .mean()
        )
        last_turn_num = self.df["turn_number"].max()
        last_turn = (
            self.df[self.df["turn_number"] == last_turn_num]
            .groupby("bias_type")["bias_score"]
            .mean()
        )

        # Plot 1: First vs Last comparison
        x = np.arange(len(first_turn))
        width = 0.35

        ax1.bar(x - width / 2, first_turn.values, width, label="First Turn", alpha=0.7)
        ax1.bar(x + width / 2, last_turn.values, width, label="Last Turn", alpha=0.7)
        ax1.axhline(y=6.0, color="red", linestyle="--", linewidth=2, alpha=0.7)

        ax1.set_xlabel("Bias Type", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Average Bias Score", fontsize=11, fontweight="bold")
        ax1.set_title(
            "Bias Amplification: First vs Last Turn", fontsize=12, fontweight="bold"
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(first_turn.index, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.set_ylim(0, 10)

        # Plot 2: Change in bias (delta)
        delta = last_turn - first_turn
        colors = ["red" if d > 0 else "green" for d in delta.values]

        ax2.barh(delta.index, delta.values, alpha=0.7, color=colors)
        ax2.axvline(x=0, color="black", linestyle="-", linewidth=1)

        ax2.set_xlabel(
            "Change in Bias Score (Last - First)", fontsize=11, fontweight="bold"
        )
        ax2.set_ylabel("Bias Type", fontsize=11, fontweight="bold")
        ax2.set_title("Bias Change Through Chain", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="x")

        # Add annotations
        for i, (bias_type, change) in enumerate(delta.items()):
            ax2.text(
                change + 0.1 if change > 0 else change - 0.1,
                i,
                f"{change:+.1f}",
                va="center",
                ha="left" if change > 0 else "right",
            )

        plt.tight_layout()

        self.figures.append(("bias_amplification", fig))
        LOG.info("Created bias amplification plot")
        return fig

    def plot_confidence_by_turn(self, figsize: tuple = (12, 6)) -> plt.Figure:
        """
        Plot how evaluation confidence changes across turns

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Convert confidence to numeric
        confidence_map = {"low": 1, "medium": 2, "high": 3}
        self.df["confidence_numeric"] = self.df["confidence"].map(confidence_map)

        fig, ax = plt.subplots(figsize=figsize)

        # Count confidence levels by turn
        conf_by_turn = (
            self.df.groupby(["turn_number", "confidence"]).size().unstack(fill_value=0)
        )

        conf_by_turn.plot(kind="bar", stacked=True, ax=ax, alpha=0.7)

        ax.set_xlabel("Turn Number", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax.set_title(
            "Evaluation Confidence Distribution Across Chain Turns",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(title="Confidence", title_fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        self.figures.append(("confidence_by_turn", fig))
        LOG.info("Created confidence distribution plot")
        return fig

    def plot_individual_chains(
        self, figsize: tuple = (14, 10), max_chains: int = 6
    ) -> plt.Figure:
        """
        Plot individual chain traces (one subplot per chain)

        Args:
            figsize: Figure size
            max_chains: Maximum number of chains to plot

        Returns:
            Matplotlib figure
        """
        # Get unique chain combinations
        unique_chains = []
        for result in self.results["results"][:max_chains]:
            chain_key = (tuple(result["agent_sequence"]), result["prompt_index"])
            if chain_key not in unique_chains:
                unique_chains.append((result, chain_key))

        n_chains = min(len(unique_chains), max_chains)
        n_cols = 2
        n_rows = (n_chains + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        axes = axes.flatten()

        for idx, (chain_result, _) in enumerate(unique_chains[:max_chains]):
            ax = axes[idx]

            # Plot bias evolution for this specific chain
            for bias_type, scores in chain_result["bias_evolution"].items():
                turns = list(range(1, len(scores) + 1))
                ax.plot(turns, scores, marker="o", label=bias_type, linewidth=2)

            ax.axhline(y=6.0, color="red", linestyle="--", linewidth=1.5, alpha=0.5)
            ax.set_xlabel("Turn", fontsize=10)
            ax.set_ylabel("Bias Score", fontsize=10)
            ax.set_title(
                f"Chain: {' → '.join(chain_result['agent_sequence'])}\n"
                f"Prompt {chain_result['prompt_index']}",
                fontsize=10,
                fontweight="bold",
            )
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 10)

        # Hide unused subplots
        for idx in range(n_chains, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Individual Chain Bias Evolution", fontsize=14, fontweight="bold")
        plt.tight_layout()

        self.figures.append(("individual_chains", fig))
        LOG.info("Created individual chains plot")
        return fig

    def generate_summary_stats(self) -> pd.DataFrame:
        """
        Generate summary statistics for chains

        Returns:
            DataFrame with summary statistics
        """
        # Calculate stats by agent sequence
        summary = (
            self.df.groupby("agent_sequence")
            .agg(
                {
                    "bias_score": ["count", "mean", "std", "min", "max"],
                    "is_biased": lambda x: (x.sum() / len(x) * 100),  # percentage
                }
            )
            .round(2)
        )

        summary.columns = ["count", "mean", "std", "min", "max", "biased_pct"]

        LOG.info("Generated summary statistics")
        return summary

    def plot_all(self) -> List[plt.Figure]:
        """
        Generate all plots

        Returns:
            List of matplotlib figures
        """
        LOG.info("Generating all plots...")

        self.plot_bias_evolution()
        self.plot_agent_contribution()
        self.plot_ordering_comparison()
        self.plot_bias_propagation_heatmap()
        self.plot_bias_amplification()
        self.plot_confidence_by_turn()
        self.plot_individual_chains()

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
        self, output_dir: str, filename: str = "chain_summary_statistics.csv"
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
    """Test ChainBiasPlotter"""
    print("=" * 80)
    print("TESTING: ChainBiasPlotter")
    print("=" * 80)

    # Initialize plotter
    print("\n[Setup] Initializing plotter...")
    plotter = ChainBiasPlotter()
    print("✓ Plotter initialized")

    # Check for test data
    print("\n[Setup] Looking for test results...")
    test_file = "results/chain_evaluations/test_chain_results.json"

    if not Path(test_file).exists():
        print(f"✗ Test file not found: {test_file}")
        print("Run chain_runner.py first to generate test data")
        return

    # Load results
    print(f"\n[Test 1] Loading results from {test_file}")
    print("-" * 80)
    plotter.load_results(test_file)
    print(f"✓ Loaded results")
    print(f"  DataFrame shape: {plotter.df.shape}")

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
    saved = plotter.save_plots("results/chain_evaluations/plots")
    print(f"✓ Saved {len(saved)} plots:")
    for path in saved:
        print(f"  - {path}")

    # Save summary stats
    print("\n[Test 5] Saving summary statistics")
    print("-" * 80)
    stats_path = plotter.save_summary_stats("results/chain_evaluations/plots")
    print(f"✓ Saved to {stats_path}")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("\n✓ All plots generated successfully!")
    print(f"  View plots in: results/chain_evaluations/plots/")


if __name__ == "__main__":
    main()
