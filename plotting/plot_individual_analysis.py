"""
Individual Analysis Plots for FairWatch.
Generates separate analysis plots for each baseline agent and each sequential ordering.
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# Add plotting directory to path
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import FairWatchDataLoader, load_data

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def create_baseline_analysis(df: pd.DataFrame, agent_name: str, output_dir: Path):
    """Create analysis plots for a single baseline agent."""
    df_agent = df[
        (df["experiment_type"] == "baseline") & (df["agent"] == agent_name)
    ].copy()

    if df_agent.empty:
        print(f"  No data for baseline {agent_name}")
        return

    agent_dir = output_dir / "baselines" / agent_name.lower().replace(" ", "_")
    agent_dir.mkdir(parents=True, exist_ok=True)

    df_valid = df_agent[df_agent["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)
    df_approved = df_valid[df_valid["approval_decision"] == "approve"]

    # Figure 1: Overview Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"{agent_name} - Baseline Analysis\n(n={len(df_valid):,})",
        fontsize=14,
        fontweight="bold",
    )

    # 1.1 Approval Rate by Ethnicity
    ax = axes[0, 0]
    eth_rates = df_valid.groupby("ethnicity_signal")["approved"].mean().sort_values()
    colors = [
        (
            "#e74c3c"
            if r < eth_rates.mean() - 0.05
            else "#27ae60" if r > eth_rates.mean() + 0.05 else "#3498db"
        )
        for r in eth_rates
    ]
    bars = ax.barh(eth_rates.index, eth_rates.values, color=colors)
    ax.set_xlabel("Approval Rate")
    ax.set_title("Approval Rate by Ethnicity")
    ax.set_xlim(0, 1)
    for bar, rate in zip(bars, eth_rates.values):
        ax.text(
            rate + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1%}",
            va="center",
            fontsize=9,
        )

    # 1.2 Interest Rate Distribution
    ax = axes[0, 1]
    if len(df_approved) > 0 and df_approved["interest_rate"].notna().any():
        sns.histplot(
            df_approved["interest_rate"].dropna(),
            ax=ax,
            bins=20,
            color="#3498db",
            edgecolor="white",
        )
        ax.axvline(
            df_approved["interest_rate"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {df_approved['interest_rate'].mean():.1f}%",
        )
        ax.legend()
    ax.set_xlabel("Interest Rate (%)")
    ax.set_title("Interest Rate Distribution")

    # 1.3 Approval Type Distribution
    ax = axes[0, 2]
    if df_valid["approval_type"].notna().any():
        type_counts = df_valid["approval_type"].value_counts()
        colors_type = {
            "STANDARD_TERMS": "#27ae60",
            "SUBOPTIMAL_TERMS": "#f39c12",
            "MANUAL_REVIEW": "#3498db",
            "DENIAL": "#e74c3c",
        }
        ax.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.1f%%",
            colors=[colors_type.get(t, "#95a5a6") for t in type_counts.index],
        )
        ax.set_title("Approval Type Distribution")

    # 1.4 Approval by Credit Score
    ax = axes[1, 0]
    if df_valid["credit_score"].notna().any():
        df_valid["credit_tier"] = pd.cut(
            df_valid["credit_score"],
            bins=[0, 580, 670, 740, 800, 900],
            labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        )
        credit_rates = df_valid.groupby("credit_tier", observed=True)["approved"].mean()
        ax.bar(
            credit_rates.index, credit_rates.values, color="#3498db", edgecolor="white"
        )
        ax.set_ylabel("Approval Rate")
        ax.set_title("Approval Rate by Credit Score")
        ax.set_ylim(0, 1)

    # 1.5 Approval by Age Group
    ax = axes[1, 1]
    if df_valid["age"].notna().any():
        df_valid["age_group"] = pd.cut(
            df_valid["age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "55+"],
        )
        age_rates = df_valid.groupby("age_group", observed=True)["approved"].mean()
        ax.bar(age_rates.index, age_rates.values, color="#9b59b6", edgecolor="white")
        ax.set_ylabel("Approval Rate")
        ax.set_title("Approval Rate by Age Group")
        ax.set_ylim(0, 1)

    # 1.6 Confidence Distribution
    ax = axes[1, 2]
    if df_valid["confidence_level"].notna().any():
        conf_counts = df_valid["confidence_level"].value_counts()
        colors_conf = {"high": "#27ae60", "medium": "#f39c12", "low": "#e74c3c"}
        conf_order = ["high", "medium", "low"]
        conf_counts = conf_counts.reindex(
            [c for c in conf_order if c in conf_counts.index]
        )
        ax.bar(
            conf_counts.index,
            conf_counts.values,
            color=[colors_conf.get(c, "#95a5a6") for c in conf_counts.index],
        )
        ax.set_ylabel("Count")
        ax.set_title("Confidence Level Distribution")

    plt.tight_layout()
    plt.savefig(agent_dir / "overview_dashboard.png", dpi=300, bbox_inches="tight")
    plt.savefig(agent_dir / "overview_dashboard.pdf", bbox_inches="tight")
    plt.close()

    # Figure 2: Ethnicity Bias Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    if (
        df_valid["ethnicity_signal"].notna().any()
        and df_valid["credit_score"].notna().any()
    ):
        df_valid["credit_tier"] = pd.cut(
            df_valid["credit_score"],
            bins=[0, 580, 670, 740, 800, 900],
            labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        )
        pivot = df_valid.pivot_table(
            values="approved",
            index="ethnicity_signal",
            columns="credit_tier",
            aggfunc="mean",
            observed=True,
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Approval Rate"},
        )
        ax.set_title(f"{agent_name}: Approval Rate by Ethnicity and Credit Score")

    plt.tight_layout()
    plt.savefig(
        agent_dir / "ethnicity_credit_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(agent_dir / "ethnicity_credit_heatmap.pdf", bbox_inches="tight")
    plt.close()

    # Figure 3: Interest Rate by Demographics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if len(df_approved) > 0:
        # Interest rate by ethnicity
        ax = axes[0]
        if (
            df_approved["ethnicity_signal"].notna().any()
            and df_approved["interest_rate"].notna().any()
        ):
            eth_ir = (
                df_approved.groupby("ethnicity_signal")["interest_rate"]
                .agg(["mean", "std"])
                .sort_values("mean")
            )
            ax.barh(
                eth_ir.index,
                eth_ir["mean"],
                xerr=eth_ir["std"],
                color="#3498db",
                capsize=3,
            )
            ax.set_xlabel("Interest Rate (%)")
            ax.set_title("Average Interest Rate by Ethnicity")

        # Interest rate by credit score
        ax = axes[1]
        if (
            df_approved["credit_score"].notna().any()
            and df_approved["interest_rate"].notna().any()
        ):
            df_approved["credit_tier"] = pd.cut(
                df_approved["credit_score"],
                bins=[0, 580, 670, 740, 800, 900],
                labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
            )
            credit_ir = df_approved.groupby("credit_tier", observed=True)[
                "interest_rate"
            ].agg(["mean", "std"])
            ax.bar(
                credit_ir.index,
                credit_ir["mean"],
                yerr=credit_ir["std"],
                color="#9b59b6",
                capsize=3,
            )
            ax.set_ylabel("Interest Rate (%)")
            ax.set_title("Average Interest Rate by Credit Score")

    plt.tight_layout()
    plt.savefig(
        agent_dir / "interest_rate_demographics.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(agent_dir / "interest_rate_demographics.pdf", bbox_inches="tight")
    plt.close()

    # Save summary statistics
    summary = {
        "agent": agent_name,
        "total_records": len(df_valid),
        "approval_rate": df_valid["approved"].mean(),
        "avg_interest_rate": (
            df_approved["interest_rate"].mean() if len(df_approved) > 0 else None
        ),
        "ethnicity_approval_rates": df_valid.groupby("ethnicity_signal")["approved"]
        .mean()
        .to_dict(),
        "approval_type_distribution": df_valid["approval_type"]
        .value_counts(normalize=True)
        .to_dict(),
    }

    with open(agent_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved {agent_name} baseline analysis to {agent_dir}")


def create_sequential_analysis(df: pd.DataFrame, ordering: str, output_dir: Path):
    """Create analysis plots for a single sequential ordering."""
    df_seq = df[
        (df["experiment_type"] == "sequential") & (df["agent_order"] == ordering)
    ].copy()

    if df_seq.empty:
        print(f"  No data for ordering {ordering}")
        return

    # Create clean folder name
    folder_name = ordering.lower().replace(" ", "_")
    seq_dir = output_dir / "sequential" / folder_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    df_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)
    df_approved = df_valid[df_valid["approval_decision"] == "approve"]

    # Get final decisions
    df_final = df_valid[df_valid["is_final_agent"] == True].copy()

    # Parse ordering for display
    agents_display = ordering.replace("_", " → ")

    # Figure 1: Chain Overview Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Sequential Chain Analysis\n{agents_display}\n(n={len(df_final):,} chains)",
        fontsize=12,
        fontweight="bold",
    )

    # 1.1 Approval Rate by Agent Position
    ax = axes[0, 0]
    pos_rates = df_valid.groupby("agent_position")["approved"].mean()
    ax.plot(
        pos_rates.index,
        pos_rates.values,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#3498db",
    )
    ax.fill_between(pos_rates.index, pos_rates.values, alpha=0.3, color="#3498db")
    ax.set_xlabel("Agent Position")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rate Evolution Through Chain")
    ax.set_ylim(0, 1)
    ax.set_xticks(pos_rates.index)

    # 1.2 Final Approval Rate by Ethnicity
    ax = axes[0, 1]
    if df_final["ethnicity_signal"].notna().any():
        eth_rates = (
            df_final.groupby("ethnicity_signal")["approved"].mean().sort_values()
        )
        colors = [
            (
                "#e74c3c"
                if r < eth_rates.mean() - 0.05
                else "#27ae60" if r > eth_rates.mean() + 0.05 else "#3498db"
            )
            for r in eth_rates
        ]
        bars = ax.barh(eth_rates.index, eth_rates.values, color=colors)
        ax.set_xlabel("Approval Rate")
        ax.set_title("Final Approval Rate by Ethnicity")
        ax.set_xlim(0, 1)
        for bar, rate in zip(bars, eth_rates.values):
            ax.text(
                rate + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}",
                va="center",
                fontsize=9,
            )

    # 1.3 Interest Rate Evolution
    ax = axes[0, 2]
    if df_approved["interest_rate"].notna().any():
        ir_by_pos = df_approved.groupby("agent_position")["interest_rate"].agg(
            ["mean", "std"]
        )
        ax.errorbar(
            ir_by_pos.index,
            ir_by_pos["mean"],
            yerr=ir_by_pos["std"],
            marker="s",
            linewidth=2,
            markersize=8,
            capsize=5,
            color="#e74c3c",
        )
        ax.set_xlabel("Agent Position")
        ax.set_ylabel("Interest Rate (%)")
        ax.set_title("Interest Rate Evolution")
        ax.set_xticks(ir_by_pos.index)

    # 1.4 Agent Agreement Analysis
    ax = axes[1, 0]
    # Check decision consistency across agents
    chain_decisions = df_valid.groupby("chain_id")["approval_decision"].agg(
        lambda x: x.nunique()
    )
    agreement_counts = chain_decisions.value_counts().sort_index()
    ax.bar(agreement_counts.index, agreement_counts.values, color="#9b59b6")
    ax.set_xlabel("Number of Unique Decisions in Chain")
    ax.set_ylabel("Number of Chains")
    ax.set_title("Agent Agreement Distribution")

    # 1.5 Approval Type Distribution (Final)
    ax = axes[1, 1]
    if df_final["approval_type"].notna().any():
        type_counts = df_final["approval_type"].value_counts()
        colors_type = {
            "STANDARD_TERMS": "#27ae60",
            "SUBOPTIMAL_TERMS": "#f39c12",
            "MANUAL_REVIEW": "#3498db",
            "DENIAL": "#e74c3c",
        }
        ax.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.1f%%",
            colors=[colors_type.get(t, "#95a5a6") for t in type_counts.index],
        )
        ax.set_title("Final Approval Type Distribution")

    # 1.6 Confidence Evolution
    ax = axes[1, 2]
    if df_valid["confidence_probability"].notna().any():
        conf_by_pos = df_valid.groupby("agent_position")["confidence_probability"].agg(
            ["mean", "std"]
        )
        ax.errorbar(
            conf_by_pos.index,
            conf_by_pos["mean"],
            yerr=conf_by_pos["std"],
            marker="D",
            linewidth=2,
            markersize=8,
            capsize=5,
            color="#27ae60",
        )
        ax.set_xlabel("Agent Position")
        ax.set_ylabel("Confidence Probability")
        ax.set_title("Confidence Evolution")
        ax.set_xticks(conf_by_pos.index)

    plt.tight_layout()
    plt.savefig(seq_dir / "chain_overview.png", dpi=300, bbox_inches="tight")
    plt.savefig(seq_dir / "chain_overview.pdf", bbox_inches="tight")
    plt.close()

    # Figure 2: Per-Agent Analysis
    agents = df_valid["agent"].unique()
    n_agents = len(agents)

    if n_agents > 0:
        fig, axes = plt.subplots(2, n_agents, figsize=(4 * n_agents, 8))
        if n_agents == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(
            f"Per-Agent Analysis: {agents_display}", fontsize=12, fontweight="bold"
        )

        for i, agent in enumerate(agents):
            df_agent = df_valid[df_valid["agent"] == agent]
            df_agent_approved = df_agent[df_agent["approval_decision"] == "approve"]

            # Approval rate by ethnicity
            ax = axes[0, i]
            if df_agent["ethnicity_signal"].notna().any():
                eth_rates = df_agent.groupby("ethnicity_signal")["approved"].mean()
                ax.bar(range(len(eth_rates)), eth_rates.values, color="#3498db")
                ax.set_xticks(range(len(eth_rates)))
                ax.set_xticklabels(
                    [e.replace("_Signal", "") for e in eth_rates.index],
                    rotation=45,
                    ha="right",
                )
                ax.set_ylim(0, 1)
                ax.set_ylabel("Approval Rate" if i == 0 else "")
                ax.set_title(f"{agent}\n(Position {i+1})")

            # Interest rate distribution
            ax = axes[1, i]
            if (
                len(df_agent_approved) > 0
                and df_agent_approved["interest_rate"].notna().any()
            ):
                sns.histplot(
                    df_agent_approved["interest_rate"].dropna(),
                    ax=ax,
                    bins=15,
                    color="#e74c3c",
                )
                ax.axvline(
                    df_agent_approved["interest_rate"].mean(),
                    color="black",
                    linestyle="--",
                )
            ax.set_xlabel("Interest Rate (%)" if i == n_agents // 2 else "")
            ax.set_ylabel("Count" if i == 0 else "")

        plt.tight_layout()
        plt.savefig(seq_dir / "per_agent_analysis.png", dpi=300, bbox_inches="tight")
        plt.savefig(seq_dir / "per_agent_analysis.pdf", bbox_inches="tight")
        plt.close()

    # Figure 3: Decision Flow Sankey-style (simplified as heatmap)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create decision transition matrix
    decision_by_position = df_valid.pivot_table(
        values="approved",
        index="agent_position",
        columns="ethnicity_signal",
        aggfunc="mean",
    )
    if not decision_by_position.empty:
        decision_by_position.columns = [
            c.replace("_Signal", "") for c in decision_by_position.columns
        ]
        sns.heatmap(
            decision_by_position,
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Approval Rate"},
        )
        ax.set_xlabel("Ethnicity")
        ax.set_ylabel("Agent Position")
        ax.set_title("Approval Rate by Position and Ethnicity")

    plt.tight_layout()
    plt.savefig(seq_dir / "decision_flow.png", dpi=300, bbox_inches="tight")
    plt.savefig(seq_dir / "decision_flow.pdf", bbox_inches="tight")
    plt.close()

    # Figure 4: First Agent vs Final Decision Transition Matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"First Agent → Final Decision Analysis\n{agents_display}",
        fontsize=12,
        fontweight="bold",
    )

    # Get first agent decisions (position 1)
    df_first = df_valid[df_valid["agent_position"] == 1][
        ["chain_id", "approval_decision"]
    ].copy()
    df_first.columns = ["chain_id", "first_decision"]

    # Merge with final decisions
    df_transitions = df_final[["chain_id", "approval_decision"]].merge(
        df_first, on="chain_id", how="inner"
    )
    df_transitions.columns = ["chain_id", "final_decision", "first_decision"]

    if len(df_transitions) > 0:
        # 4.1 Transition Matrix (Counts)
        ax = axes[0]
        transition_counts = pd.crosstab(
            df_transitions["first_decision"],
            df_transitions["final_decision"],
            margins=False,
        )
        # Ensure both approve and deny are present
        for col in ["approve", "deny"]:
            if col not in transition_counts.columns:
                transition_counts[col] = 0
        for idx in ["approve", "deny"]:
            if idx not in transition_counts.index:
                transition_counts.loc[idx] = 0
        transition_counts = transition_counts.reindex(
            index=["approve", "deny"], columns=["approve", "deny"], fill_value=0
        )

        # Create annotations with counts and percentages
        total = transition_counts.values.sum()
        annot_text = np.array(
            [
                [
                    f"{transition_counts.iloc[i, j]:,}\n({transition_counts.iloc[i, j]/total:.1%})"
                    for j in range(2)
                ]
                for i in range(2)
            ]
        )

        colors_matrix = np.array(
            [[0.8, 0.2], [0.2, 0.8]]
        )  # Green diagonal, red off-diagonal
        sns.heatmap(
            transition_counts,
            annot=annot_text,
            fmt="",
            cmap="RdYlGn",
            ax=ax,
            cbar_kws={"label": "Count"},
            xticklabels=["Approve", "Deny"],
            yticklabels=["Approve", "Deny"],
        )
        ax.set_xlabel("Final Decision", fontsize=11)
        ax.set_ylabel("First Agent Decision", fontsize=11)
        ax.set_title("Decision Transition Counts", fontsize=11, fontweight="bold")

        # 4.2 Conditional Probabilities
        ax = axes[1]
        # P(Final | First) - row-normalized
        transition_probs = transition_counts.div(
            transition_counts.sum(axis=1), axis=0
        ).fillna(0)

        annot_probs = np.array(
            [[f"{transition_probs.iloc[i, j]:.1%}" for j in range(2)] for i in range(2)]
        )

        sns.heatmap(
            transition_probs,
            annot=annot_probs,
            fmt="",
            cmap="RdYlGn",
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Probability"},
            xticklabels=["Approve", "Deny"],
            yticklabels=["Approve", "Deny"],
        )
        ax.set_xlabel("Final Decision", fontsize=11)
        ax.set_ylabel("First Agent Decision", fontsize=11)
        ax.set_title(
            "P(Final Decision | First Agent Decision)", fontsize=11, fontweight="bold"
        )

        # Add summary text
        first_approve = df_transitions["first_decision"] == "approve"
        final_approve = df_transitions["final_decision"] == "approve"

        agreement_rate = (
            (first_approve & final_approve) | (~first_approve & ~final_approve)
        ).mean()
        flip_approve_to_deny = (first_approve & ~final_approve).mean()
        flip_deny_to_approve = (~first_approve & final_approve).mean()

        summary_text = (
            f"Agreement Rate: {agreement_rate:.1%}\n"
            f"Approve→Deny Flip: {flip_approve_to_deny:.1%}\n"
            f"Deny→Approve Flip: {flip_deny_to_approve:.1%}"
        )
        fig.text(
            0.5,
            0.02,
            summary_text,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(seq_dir / "first_vs_final_transition.png", dpi=300, bbox_inches="tight")
    plt.savefig(seq_dir / "first_vs_final_transition.pdf", bbox_inches="tight")
    plt.close()

    # Figure 5: First vs Final by Ethnicity (4-box breakdown per ethnicity)
    ethnicities = df_transitions["chain_id"].map(
        df_final.set_index("chain_id")["ethnicity_signal"]
    )
    df_transitions["ethnicity"] = ethnicities

    unique_ethnicities = df_transitions["ethnicity"].dropna().unique()
    n_eth = len(unique_ethnicities)

    if n_eth > 0:
        n_cols = min(4, n_eth)
        n_rows = (n_eth + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_eth == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            f"First Agent → Final Decision by Ethnicity\n{agents_display}",
            fontsize=12,
            fontweight="bold",
        )

        for idx, ethnicity in enumerate(sorted(unique_ethnicities)):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            df_eth = df_transitions[df_transitions["ethnicity"] == ethnicity]
            eth_counts = pd.crosstab(df_eth["first_decision"], df_eth["final_decision"])

            # Ensure both approve and deny are present
            for c in ["approve", "deny"]:
                if c not in eth_counts.columns:
                    eth_counts[c] = 0
            for i in ["approve", "deny"]:
                if i not in eth_counts.index:
                    eth_counts.loc[i] = 0
            eth_counts = eth_counts.reindex(
                index=["approve", "deny"], columns=["approve", "deny"], fill_value=0
            )

            eth_total = eth_counts.values.sum()
            eth_annot = np.array(
                [
                    [
                        (
                            f"{eth_counts.iloc[i, j]:,}\n({eth_counts.iloc[i, j]/eth_total:.1%})"
                            if eth_total > 0
                            else "0"
                        )
                        for j in range(2)
                    ]
                    for i in range(2)
                ]
            )

            sns.heatmap(
                eth_counts,
                annot=eth_annot,
                fmt="",
                cmap="RdYlGn",
                ax=ax,
                cbar=False,
                xticklabels=["Approve", "Deny"],
                yticklabels=["Approve", "Deny"],
            )
            eth_display = ethnicity.replace("_Signal", "") if ethnicity else "Unknown"
            ax.set_title(f"{eth_display}\n(n={eth_total:,})", fontsize=10)
            ax.set_xlabel("Final" if row == n_rows - 1 else "")
            ax.set_ylabel("First" if col == 0 else "")

        # Hide unused subplots
        for idx in range(n_eth, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.savefig(
            seq_dir / "first_vs_final_by_ethnicity.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(seq_dir / "first_vs_final_by_ethnicity.pdf", bbox_inches="tight")
        plt.close()

    # Save summary statistics
    # Calculate transition stats if data exists
    transition_stats = {}
    if len(df_transitions) > 0:
        first_approve = df_transitions["first_decision"] == "approve"
        final_approve = df_transitions["final_decision"] == "approve"
        transition_stats = {
            "agreement_rate": float(
                (
                    (first_approve & final_approve) | (~first_approve & ~final_approve)
                ).mean()
            ),
            "approve_to_deny_flip_rate": float((first_approve & ~final_approve).mean()),
            "deny_to_approve_flip_rate": float((~first_approve & final_approve).mean()),
            "first_approve_final_approve": int((first_approve & final_approve).sum()),
            "first_approve_final_deny": int((first_approve & ~final_approve).sum()),
            "first_deny_final_approve": int((~first_approve & final_approve).sum()),
            "first_deny_final_deny": int((~first_approve & ~final_approve).sum()),
        }

    summary = {
        "ordering": ordering,
        "agents_display": agents_display,
        "total_chains": len(df_final),
        "total_records": len(df_valid),
        "final_approval_rate": (
            df_final["approved"].mean() if len(df_final) > 0 else None
        ),
        "avg_final_interest_rate": (
            df_approved[df_approved["is_final_agent"] == True]["interest_rate"].mean()
            if len(df_approved) > 0
            else None
        ),
        "approval_rate_by_position": df_valid.groupby("agent_position")["approved"]
        .mean()
        .to_dict(),
        "final_ethnicity_approval_rates": (
            df_final.groupby("ethnicity_signal")["approved"].mean().to_dict()
            if len(df_final) > 0
            else {}
        ),
        "chain_agreement_rate": (
            (chain_decisions == 1).mean() if len(chain_decisions) > 0 else None
        ),
        "first_vs_final_transitions": transition_stats,
    }

    with open(seq_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved sequential analysis to {seq_dir}")


def main():
    """Generate individual analysis plots."""
    print("=" * 60)
    print("FairWatch Individual Analysis - Plot Generation")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    df, raw = load_data("70bresults")
    print(f"Loaded {len(df):,} records")

    output_dir = Path("plotting/outputs/individual")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate baseline analysis for each agent
    print("\n[2/3] Generating baseline agent analysis...")
    baseline_agents = df[df["experiment_type"] == "baseline"]["agent"].unique()
    print(f"Found {len(baseline_agents)} baseline agents: {list(baseline_agents)}")

    for agent in baseline_agents:
        create_baseline_analysis(df, agent, output_dir)

    # Generate sequential analysis for each ordering
    print("\n[3/3] Generating sequential ordering analysis...")
    orderings = df[df["experiment_type"] == "sequential"]["agent_order"].unique()
    print(f"Found {len(orderings)} sequential orderings")

    for ordering in orderings:
        create_sequential_analysis(df, ordering, output_dir)

    # Count generated files
    baseline_files = (
        list((output_dir / "baselines").rglob("*.png"))
        if (output_dir / "baselines").exists()
        else []
    )
    sequential_files = (
        list((output_dir / "sequential").rglob("*.png"))
        if (output_dir / "sequential").exists()
        else []
    )

    print("\n" + "=" * 60)
    print("Individual Analysis Complete!")
    print("=" * 60)
    print(f"  Baseline plots: {len(baseline_files)} files")
    print(f"  Sequential plots: {len(sequential_files)} files")
    print(f"  Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
