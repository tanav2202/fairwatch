"""
Agent Comparison and Analysis Plots
Compares performance across different agents in baseline, sequential, and parallel modes.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from data_loader import load_data, FairWatchDataLoader

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_agent_approval_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare approval rates across different agents."""
    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    # Exclude business decision for fair comparison
    df_agents = df_valid[df_valid["agent"] != "Business Decision"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, exp_type in zip(axes, ["baseline", "sequential", "parallel"]):
        df_exp = df_agents[df_agents["experiment_type"] == exp_type]
        if df_exp.empty:
            ax.set_title(f"{exp_type.title()} - No Data")
            continue

        stats = (
            df_exp.groupby("agent")["approved"]
            .agg(["mean", "count", "std"])
            .reset_index()
        )
        stats = stats.sort_values("mean", ascending=False)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(stats)))
        bars = ax.bar(
            range(len(stats)),
            stats["mean"],
            color=colors,
            yerr=stats["std"] / np.sqrt(stats["count"]),
            capsize=3,
        )

        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(stats["agent"], rotation=45, ha="right")
        ax.set_ylabel("Approval Rate")
        ax.set_title(f"{exp_type.title()} - Agent Approval Rates")
        ax.set_ylim(0, 1)

        # Add sample size labels
        for i, (bar, count) in enumerate(zip(bars, stats["count"])):
            ax.annotate(
                f"n={count}",
                xy=(bar.get_x() + bar.get_width() / 2, 0.02),
                ha="center",
                va="bottom",
                fontsize=8,
                color="gray",
            )

    plt.tight_layout()
    plt.savefig(
        output_dir / "agent_approval_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "agent_approval_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: agent_approval_comparison.png")


def plot_agent_interest_rate_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare interest rate distributions across agents."""
    df_approved = df[
        (df["approval_decision"] == "approve") & (df["agent"] != "Business Decision")
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, exp_type in zip(axes, ["baseline", "sequential", "parallel"]):
        df_exp = df_approved[df_approved["experiment_type"] == exp_type]
        if df_exp.empty:
            ax.set_title(f"{exp_type.title()} - No Data")
            continue

        # Violin plot for interest rate distribution
        df_plot = df_exp[["agent", "interest_rate"]].dropna()

        if len(df_plot) > 0:
            sns.violinplot(
                data=df_plot,
                x="agent",
                y="interest_rate",
                ax=ax,
                palette="Set2",
                inner="box",
            )
            ax.tick_params(axis="x", rotation=45)

        ax.set_xlabel("Agent")
        ax.set_ylabel("Interest Rate (%)")
        ax.set_title(f"{exp_type.title()} - Interest Rate by Agent")

    plt.tight_layout()
    plt.savefig(
        output_dir / "agent_interest_rate_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "agent_interest_rate_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: agent_interest_rate_comparison.png")


def plot_agent_confidence_levels(df: pd.DataFrame, output_dir: Path):
    """Plot confidence level distributions by agent."""
    df_agents = df[df["agent"] != "Business Decision"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, exp_type in zip(axes, ["baseline", "sequential", "parallel"]):
        df_exp = df_agents[df_agents["experiment_type"] == exp_type]
        if df_exp.empty:
            continue

        # Create pivot table for confidence levels
        pivot = pd.crosstab(
            df_exp["agent"], df_exp["confidence_level"], normalize="index"
        )

        # Reorder columns
        cols = ["low", "medium", "high"]
        pivot = pivot.reindex(columns=[c for c in cols if c in pivot.columns])

        pivot.plot(
            kind="bar", stacked=True, ax=ax, color=["#e74c3c", "#f39c12", "#27ae60"]
        )
        ax.set_ylabel("Proportion")
        ax.set_xlabel("Agent")
        ax.set_title(f"{exp_type.title()} - Confidence Levels by Agent")
        ax.legend(title="Confidence", loc="upper right")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        output_dir / "agent_confidence_levels.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "agent_confidence_levels.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: agent_confidence_levels.png")


def plot_approval_type_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot approval type distribution by agent."""
    df_agents = df[df["agent"] != "Business Decision"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    approval_types = ["STANDARD_TERMS", "SUBOPTIMAL_TERMS", "MANUAL_REVIEW", "DENIAL"]
    colors = {
        "STANDARD_TERMS": "#27ae60",
        "SUBOPTIMAL_TERMS": "#f39c12",
        "MANUAL_REVIEW": "#3498db",
        "DENIAL": "#e74c3c",
    }

    for ax, exp_type in zip(axes, ["baseline", "sequential", "parallel"]):
        df_exp = df_agents[df_agents["experiment_type"] == exp_type]
        if df_exp.empty:
            continue

        pivot = pd.crosstab(df_exp["agent"], df_exp["approval_type"], normalize="index")
        pivot = pivot.reindex(columns=[c for c in approval_types if c in pivot.columns])

        pivot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[colors.get(c, "#95a5a6") for c in pivot.columns],
        )
        ax.set_ylabel("Proportion")
        ax.set_xlabel("Agent")
        ax.set_title(f"{exp_type.title()} - Approval Types by Agent")
        ax.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        output_dir / "approval_type_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "approval_type_distribution.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: approval_type_distribution.png")


def plot_agent_agreement_matrix(df: pd.DataFrame, output_dir: Path):
    """Plot agent agreement matrix for parallel evaluations."""
    # Filter for parallel mode where multiple agents evaluate same prompt
    df_parallel = df[
        (df["experiment_type"] == "parallel") & (df["agent"] != "Business Decision")
    ]

    if df_parallel.empty:
        print("No parallel data for agreement matrix")
        return

    # Create pivot: prompt_id x agent -> approval_decision
    pivot = df_parallel.pivot_table(
        index="prompt_id", columns="agent", values="approval_decision", aggfunc="first"
    )

    agents = pivot.columns.tolist()
    n_agents = len(agents)

    # Calculate agreement matrix
    agreement_matrix = np.zeros((n_agents, n_agents))

    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                mask = pivot[[agent1, agent2]].notna().all(axis=1)
                if mask.sum() > 0:
                    agree = (pivot.loc[mask, agent1] == pivot.loc[mask, agent2]).mean()
                    agreement_matrix[i, j] = agree

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        agreement_matrix,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn",
        xticklabels=agents,
        yticklabels=agents,
        ax=ax,
        vmin=0.5,
        vmax=1.0,
        cbar_kws={"label": "Agreement Rate"},
    )

    ax.set_title("Agent Agreement Matrix (Parallel Mode)\nApproval Decision Agreement")
    plt.tight_layout()
    plt.savefig(output_dir / "agent_agreement_matrix.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "agent_agreement_matrix.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: agent_agreement_matrix.png")


def plot_agent_bias_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare how different agents exhibit bias across ethnicities."""
    df_valid = df[
        (df["approval_decision"].isin(["approve", "deny"]))
        & (df["ethnicity_signal"] != "Unknown")
        & (df["agent"] != "Business Decision")
    ].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    agents = df_valid["agent"].unique()

    for ax, agent in zip(axes.flat, agents[:4]):
        df_agent = df_valid[df_valid["agent"] == agent]

        if df_agent.empty:
            ax.set_title(f"{agent} - No Data")
            continue

        # Calculate approval rate by ethnicity for each experiment type
        pivot = df_agent.pivot_table(
            values="approved",
            index="ethnicity_signal",
            columns="experiment_type",
            aggfunc="mean",
        )
        pivot.index = [e.replace("_Signal", "") for e in pivot.index]

        pivot.plot(
            kind="bar", ax=ax, width=0.8, color=["#3498db", "#e74c3c", "#2ecc71"]
        )
        ax.set_ylabel("Approval Rate")
        ax.set_xlabel("Ethnicity")
        ax.set_title(f"{agent} - Approval by Ethnicity")
        ax.legend(title="Mode")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "agent_bias_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "agent_bias_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: agent_bias_comparison.png")


def plot_overall_system_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare overall system performance across experiment types."""
    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall approval rate by experiment type
    ax = axes[0, 0]
    stats = (
        df_valid.groupby("experiment_type")
        .agg({"approved": ["mean", "std", "count"]})
        .round(4)
    )
    stats.columns = ["approval_rate", "std", "count"]
    stats = stats.reset_index()

    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(
        stats["experiment_type"],
        stats["approval_rate"],
        color=colors,
        yerr=stats["std"] / np.sqrt(stats["count"]),
        capsize=5,
    )
    ax.set_ylabel("Approval Rate")
    ax.set_title("Overall Approval Rate by Experiment Type")
    ax.set_ylim(0, 1)

    for bar, rate in zip(bars, stats["approval_rate"]):
        ax.annotate(
            f"{rate:.2%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )

    # 2. Average interest rate by experiment type
    ax = axes[0, 1]
    df_approved = df[df["approval_decision"] == "approve"]

    stats_ir = df_approved.groupby("experiment_type")["interest_rate"].agg(
        ["mean", "std", "count"]
    )
    stats_ir = stats_ir.reset_index()

    bars = ax.bar(
        stats_ir["experiment_type"],
        stats_ir["mean"],
        color=colors,
        yerr=stats_ir["std"] / np.sqrt(stats_ir["count"]),
        capsize=5,
    )
    ax.set_ylabel("Average Interest Rate (%)")
    ax.set_title("Average Interest Rate by Experiment Type")

    for bar, rate in zip(bars, stats_ir["mean"]):
        ax.annotate(
            f"{rate:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )

    # 3. Confidence distribution
    ax = axes[1, 0]
    pivot = pd.crosstab(
        df_valid["experiment_type"], df_valid["confidence_level"], normalize="index"
    )
    cols = ["low", "medium", "high"]
    pivot = pivot.reindex(columns=[c for c in cols if c in pivot.columns])

    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#e74c3c", "#f39c12", "#27ae60"])
    ax.set_ylabel("Proportion")
    ax.set_title("Confidence Level Distribution")
    ax.legend(title="Confidence")
    ax.tick_params(axis="x", rotation=0)

    # 4. Approval type distribution
    ax = axes[1, 1]
    pivot_at = pd.crosstab(
        df_valid["experiment_type"], df_valid["approval_type"], normalize="index"
    )
    approval_types = ["STANDARD_TERMS", "SUBOPTIMAL_TERMS", "MANUAL_REVIEW", "DENIAL"]
    pivot_at = pivot_at.reindex(
        columns=[c for c in approval_types if c in pivot_at.columns]
    )

    pivot_at.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#27ae60", "#f39c12", "#3498db", "#e74c3c"],
    )
    ax.set_ylabel("Proportion")
    ax.set_title("Approval Type Distribution")
    ax.legend(title="Type", fontsize=8)
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "system_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "system_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: system_comparison.png")


def main():
    """Run all agent comparison plots."""
    output_dir = Path("plotting/outputs/agent_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, _ = load_data("70bresults")
    print(f"Loaded {len(df)} records")

    print("\nGenerating agent comparison plots...")
    plot_agent_approval_comparison(df, output_dir)
    plot_agent_interest_rate_comparison(df, output_dir)
    plot_agent_confidence_levels(df, output_dir)
    plot_approval_type_distribution(df, output_dir)
    plot_agent_agreement_matrix(df, output_dir)
    plot_agent_bias_comparison(df, output_dir)
    plot_overall_system_comparison(df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
