"""
Sequential Chain Analysis Plots
Analyzes the impact of agent ordering in sequential chains.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from data_loader import load_data, FairWatchDataLoader
from itertools import permutations

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_order_impact_on_approval(df: pd.DataFrame, output_dir: Path):
    """Analyze how agent ordering affects final approval rates."""
    df_seq = df[df["experiment_type"] == "sequential"].copy()

    if df_seq.empty or "agent_order" not in df_seq.columns:
        print("No sequential data available")
        return

    # Get unique orderings
    orderings = df_seq["agent_order"].dropna().unique()

    # Calculate approval rate for each ordering (final agent decision)
    df_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    # Get the last agent in each chain
    max_positions = df_valid.groupby("chain_id")["agent_position"].transform("max")
    df_final = df_valid[df_valid["agent_position"] == max_positions].copy()

    stats = (
        df_final.groupby("agent_order")
        .agg({"approved": ["mean", "std", "count"]})
        .round(4)
    )
    stats.columns = ["approval_rate", "std", "count"]
    stats = stats.reset_index().sort_values("approval_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(stats)))
    bars = ax.barh(range(len(stats)), stats["approval_rate"], color=colors)

    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels(stats["agent_order"].str.replace("_", " → "), fontsize=9)
    ax.set_xlabel("Final Approval Rate")
    ax.set_title("Impact of Agent Ordering on Final Approval Rate")
    ax.set_xlim(0, 1)

    # Add value labels
    for i, (bar, rate, count) in enumerate(
        zip(bars, stats["approval_rate"], stats["count"])
    ):
        ax.annotate(
            f"{rate:.1%} (n={count})", xy=(rate + 0.01, i), va="center", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_dir / "order_impact_approval.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "order_impact_approval.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: order_impact_approval.png")


def plot_decision_drift(df: pd.DataFrame, output_dir: Path):
    """Analyze how decisions change through the sequential chain."""
    df_seq = df[df["experiment_type"] == "sequential"].copy()

    if df_seq.empty or "agent_position" not in df_seq.columns:
        print("No sequential data available")
        return

    df_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Average approval rate by position in chain
    ax = axes[0]
    stats = (
        df_valid.groupby("agent_position")["approved"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    ax.errorbar(
        stats["agent_position"],
        stats["mean"],
        yerr=stats["std"] / np.sqrt(stats["count"]),
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.fill_between(
        stats["agent_position"],
        stats["mean"] - stats["std"] / np.sqrt(stats["count"]),
        stats["mean"] + stats["std"] / np.sqrt(stats["count"]),
        alpha=0.3,
    )

    ax.set_xlabel("Position in Chain")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rate by Position in Sequential Chain")
    ax.set_xticks(stats["agent_position"])
    ax.set_ylim(0, 1)

    # 2. Decision changes between consecutive agents
    ax = axes[1]

    # Track changes
    changes = []
    for chain_id in df_valid["chain_id"].unique():
        chain = df_valid[df_valid["chain_id"] == chain_id].sort_values("agent_position")
        decisions = chain["approval_decision"].values

        for i in range(1, len(decisions)):
            changes.append(
                {
                    "position": i + 1,
                    "changed": decisions[i] != decisions[i - 1],
                    "from": decisions[i - 1],
                    "to": decisions[i],
                }
            )

    if changes:
        df_changes = pd.DataFrame(changes)
        change_rate = df_changes.groupby("position")["changed"].mean()

        ax.bar(change_rate.index, change_rate.values, color="coral", alpha=0.8)
        ax.set_xlabel("Position in Chain")
        ax.set_ylabel("Decision Change Rate")
        ax.set_title("Rate of Decision Changes by Position")
        ax.set_ylim(0, max(0.5, change_rate.max() * 1.1))

    plt.tight_layout()
    plt.savefig(output_dir / "decision_drift.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "decision_drift.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: decision_drift.png")


def plot_first_agent_influence(df: pd.DataFrame, output_dir: Path):
    """Analyze how the first agent influences the final decision."""
    df_seq = df[df["experiment_type"] == "sequential"].copy()

    if df_seq.empty:
        return

    df_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()

    # Get first and last agent decisions for each chain
    first_agents = df_valid[df_valid["agent_position"] == 1][
        ["chain_id", "approval_decision", "agent"]
    ].rename(columns={"approval_decision": "first_decision", "agent": "first_agent"})

    last_positions = df_valid.groupby("chain_id")["agent_position"].max().reset_index()
    last_positions.columns = ["chain_id", "last_pos"]

    df_last = df_valid.merge(last_positions, on="chain_id")
    df_last = df_last[df_last["agent_position"] == df_last["last_pos"]][
        ["chain_id", "approval_decision"]
    ].rename(columns={"approval_decision": "final_decision"})

    merged = first_agents.merge(df_last, on="chain_id")

    if merged.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Agreement between first and final decision
    ax = axes[0]
    merged["agreement"] = (merged["first_decision"] == merged["final_decision"]).astype(
        int
    )

    agreement_by_first = (
        merged.groupby("first_agent")["agreement"].mean().sort_values(ascending=False)
    )

    colors = plt.cm.RdYlGn(agreement_by_first.values)
    bars = ax.bar(
        range(len(agreement_by_first)), agreement_by_first.values, color=colors
    )
    ax.set_xticks(range(len(agreement_by_first)))
    ax.set_xticklabels(agreement_by_first.index, rotation=45, ha="right")
    ax.set_ylabel("Agreement Rate with Final Decision")
    ax.set_title("First Agent Influence on Final Decision")
    ax.set_ylim(0, 1)

    for bar, rate in zip(bars, agreement_by_first.values):
        ax.annotate(
            f"{rate:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )

    # 2. Transition matrix: first -> final decision
    ax = axes[1]

    transition = pd.crosstab(
        merged["first_decision"], merged["final_decision"], normalize="index"
    )

    sns.heatmap(
        transition,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Transition Rate"},
    )
    ax.set_xlabel("Final Decision")
    ax.set_ylabel("First Agent Decision")
    ax.set_title("First to Final Decision Transition Matrix")

    plt.tight_layout()
    plt.savefig(output_dir / "first_agent_influence.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "first_agent_influence.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: first_agent_influence.png")


def plot_interest_rate_evolution(df: pd.DataFrame, output_dir: Path):
    """Plot how interest rates evolve through the chain."""
    df_seq = df[
        (df["experiment_type"] == "sequential") & (df["approval_decision"] == "approve")
    ].copy()

    if df_seq.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Average interest rate by position
    ax = axes[0]
    stats = (
        df_seq.groupby("agent_position")["interest_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    ax.errorbar(
        stats["agent_position"],
        stats["mean"],
        yerr=stats["std"] / np.sqrt(stats["count"]),
        marker="s",
        capsize=5,
        linewidth=2,
        markersize=8,
        color="steelblue",
    )
    ax.fill_between(
        stats["agent_position"],
        stats["mean"] - stats["std"] / np.sqrt(stats["count"]),
        stats["mean"] + stats["std"] / np.sqrt(stats["count"]),
        alpha=0.3,
        color="steelblue",
    )

    ax.set_xlabel("Position in Chain")
    ax.set_ylabel("Average Interest Rate (%)")
    ax.set_title("Interest Rate Evolution in Sequential Chain")
    ax.set_xticks(stats["agent_position"])

    # 2. Interest rate by agent regardless of position
    ax = axes[1]

    stats_agent = (
        df_seq.groupby("agent")["interest_rate"].agg(["mean", "std"]).reset_index()
    )
    stats_agent = stats_agent.sort_values("mean")

    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(stats_agent)))
    ax.barh(
        range(len(stats_agent)),
        stats_agent["mean"],
        xerr=stats_agent["std"],
        color=colors,
        capsize=3,
    )
    ax.set_yticks(range(len(stats_agent)))
    ax.set_yticklabels(stats_agent["agent"])
    ax.set_xlabel("Average Interest Rate (%)")
    ax.set_title("Average Interest Rate by Agent")

    plt.tight_layout()
    plt.savefig(
        output_dir / "interest_rate_evolution.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "interest_rate_evolution.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: interest_rate_evolution.png")


def plot_order_bias_interaction(df: pd.DataFrame, output_dir: Path):
    """Analyze how ordering interacts with demographic bias."""
    df_seq = df[
        (df["experiment_type"] == "sequential") & (df["ethnicity_signal"] != "Unknown")
    ].copy()

    if df_seq.empty or "agent_order" not in df_seq.columns:
        print("No sequential data for order bias analysis")
        return

    df_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    # Get final decisions
    max_positions = df_valid.groupby("chain_id")["agent_position"].transform("max")
    df_final = df_valid[df_valid["agent_position"] == max_positions].copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate approval rate by ordering and ethnicity
    pivot = df_final.pivot_table(
        values="approved",
        index="agent_order",
        columns="ethnicity_signal",
        aggfunc="mean",
    )
    pivot.columns = [c.replace("_Signal", "") for c in pivot.columns]

    # Sort by variance (most biased orderings first)
    pivot["variance"] = pivot.var(axis=1)
    pivot = pivot.sort_values("variance", ascending=False).drop("variance", axis=1)

    # Show top 10 most and least biased
    if len(pivot) > 10:
        pivot_top = pivot.head(5)
        pivot_bottom = pivot.tail(5)
        pivot_show = pd.concat([pivot_top, pivot_bottom])
    else:
        pivot_show = pivot

    sns.heatmap(
        pivot_show,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Approval Rate"},
    )
    ax.set_xlabel("Ethnicity")
    ax.set_ylabel("Agent Order")
    ax.set_title(
        "Approval Rate by Ordering and Ethnicity\n(Top: Most Variance, Bottom: Least Variance)"
    )

    # Clean up y-axis labels
    labels = [item.get_text().replace("_", "→") for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "order_bias_interaction.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "order_bias_interaction.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: order_bias_interaction.png")


def plot_consensus_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze consensus patterns in sequential chains."""
    df_seq = df[df["experiment_type"] == "sequential"].copy()

    if df_seq.empty or "chain_id" not in df_seq.columns:
        print("No sequential data for consensus analysis")
        return

    # Calculate consensus for each chain
    consensus_data = []

    for chain_id in df_seq["chain_id"].dropna().unique():
        chain = df_seq[df_seq["chain_id"] == chain_id]
        decisions = chain["approval_decision"].dropna().values

        if len(decisions) < 2:
            continue

        # Full consensus: all agents agree
        full_consensus = len(set(decisions)) == 1

        # Majority: >50% agree
        from collections import Counter

        counts = Counter(decisions)
        majority_decision = counts.most_common(1)[0][0]
        majority_pct = counts.most_common(1)[0][1] / len(decisions)

        consensus_data.append(
            {
                "chain_id": chain_id,
                "agent_order": chain["agent_order"].iloc[0],
                "full_consensus": full_consensus,
                "majority_pct": majority_pct,
                "majority_decision": majority_decision,
                "n_agents": len(decisions),
            }
        )

    if not consensus_data:
        return

    df_consensus = pd.DataFrame(consensus_data)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Full consensus rate by ordering
    ax = axes[0]
    consensus_by_order = (
        df_consensus.groupby("agent_order")["full_consensus"]
        .mean()
        .sort_values(ascending=False)
    )

    if len(consensus_by_order) > 15:
        consensus_by_order = consensus_by_order.head(15)

    ax.barh(
        range(len(consensus_by_order)), consensus_by_order.values, color="steelblue"
    )
    ax.set_yticks(range(len(consensus_by_order)))
    ax.set_yticklabels(
        [o.replace("_", "→") for o in consensus_by_order.index], fontsize=8
    )
    ax.set_xlabel("Full Consensus Rate")
    ax.set_title("Full Consensus Rate by Ordering (Top 15)")
    ax.set_xlim(0, 1)

    # 2. Majority agreement distribution
    ax = axes[1]
    ax.hist(
        df_consensus["majority_pct"],
        bins=20,
        color="coral",
        alpha=0.8,
        edgecolor="black",
    )
    ax.axvline(x=0.75, color="green", linestyle="--", label="75% threshold")
    ax.axvline(
        x=df_consensus["majority_pct"].mean(), color="blue", linestyle="-", label="Mean"
    )
    ax.set_xlabel("Majority Agreement Percentage")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Majority Agreement")
    ax.legend()

    # 3. Consensus by majority decision
    ax = axes[2]
    pivot = df_consensus.groupby("majority_decision")["full_consensus"].mean()

    colors = ["#27ae60" if d == "approve" else "#e74c3c" for d in pivot.index]
    ax.bar(pivot.index, pivot.values, color=colors)
    ax.set_ylabel("Full Consensus Rate")
    ax.set_title("Consensus Rate by Majority Decision")
    ax.set_ylim(0, 1)

    for i, (decision, rate) in enumerate(pivot.items()):
        ax.annotate(f"{rate:.1%}", xy=(i, rate), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_dir / "consensus_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "consensus_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: consensus_analysis.png")


def main():
    """Run all sequential chain analysis plots."""
    output_dir = Path("plotting/outputs/sequential_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, _ = load_data("70bresults")
    print(f"Loaded {len(df)} records")

    print("\nGenerating sequential chain analysis plots...")
    plot_order_impact_on_approval(df, output_dir)
    plot_decision_drift(df, output_dir)
    plot_first_agent_influence(df, output_dir)
    plot_interest_rate_evolution(df, output_dir)
    plot_order_bias_interaction(df, output_dir)
    plot_consensus_analysis(df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
