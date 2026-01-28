"""
Qwen Validation Analysis Plots
Generates analysis plots for Qwen model validation runs (8 sequential orderings).
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import re

# Add plotting directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def load_qwen_data(data_dir: str = "qwen_validation") -> pd.DataFrame:
    """Load Qwen validation sequential data."""
    data_path = Path(data_dir)
    records = []

    # Agent name mapping
    agent_map = {
        "consumer": "Consumer Advocate",
        "data": "Data Science",
        "regulatory": "Regulatory",
        "risk": "Risk Manager",
    }

    for file in sorted(data_path.glob("sequential_*.json")):
        print(f"  Loading {file.name}...")

        # Extract agent order from filename
        name = file.stem.replace("sequential_", "")
        parts = name.split("_")
        order = []
        for part in parts:
            if part in agent_map:
                order.append(agent_map[part])

        order_str = "_".join([a.replace(" ", "_") for a in order])

        try:
            with open(file, "r") as f:
                data = json.load(f)

            for result in data.get("results", []):
                # Parse input
                input_str = result.get("input", "{}")
                try:
                    input_data = json.loads(input_str)
                except:
                    input_data = {}

                base_record = {
                    "chain_id": result.get("chain_id"),
                    "agent_order": order_str,
                    "agent_order_list": order,
                    "name": input_data.get("name"),
                    "ethnicity_signal": input_data.get("ethnicity_signal"),
                    "credit_score": input_data.get("credit_score"),
                    "visa_status": input_data.get("visa_status"),
                    "income": input_data.get("income"),
                    "age": input_data.get("age"),
                    "loan_amount": input_data.get("loan_amount"),
                    "dti_ratio": input_data.get("dti_ratio"),
                }

                # Get decisions
                decisions = result.get("decisions", {})

                if decisions:
                    for i, agent_name in enumerate(order):
                        agent_key = agent_name.lower().replace(" ", "_")
                        agent_data = decisions.get(agent_key, {})

                        if not agent_data:
                            continue

                        record = base_record.copy()
                        record["agent"] = agent_name
                        record["agent_position"] = i + 1
                        record["is_final_agent"] = i == len(order) - 1
                        record["approval_decision"] = agent_data.get(
                            "approval_decision"
                        )
                        record["approval_type"] = agent_data.get("approval_type")
                        record["interest_rate"] = agent_data.get("interest_rate")
                        record["confidence_probability"] = agent_data.get(
                            "confidence_probability"
                        )
                        record["confidence_level"] = agent_data.get("confidence_level")

                        records.append(record)

        except Exception as e:
            print(f"    Error loading {file.name}: {e}")
            continue

    df = pd.DataFrame(records)
    df["experiment_type"] = "sequential"
    return df


def create_sequential_analysis(df: pd.DataFrame, ordering: str, output_dir: Path):
    """Create analysis plots for a single sequential ordering."""
    df_seq = df[df["agent_order"] == ordering].copy()

    if df_seq.empty:
        print(f"  No data for ordering {ordering}")
        return

    folder_name = ordering.lower().replace(" ", "_")
    seq_dir = output_dir / "sequential" / folder_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    df_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)
    df_approved = df_valid[df_valid["approval_decision"] == "approve"]

    # Get final decisions
    df_final = df_valid[df_valid["is_final_agent"] == True].copy()

    if len(df_final) == 0:
        print(f"  No final decisions for {ordering}")
        return

    agents_display = ordering.replace("_", " → ")

    # Figure 1: Chain Overview Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Qwen - Sequential Chain Analysis\n{agents_display}\n(n={len(df_final):,} chains)",
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
        valid_ir = df_approved[df_approved["interest_rate"].between(0, 50)]
        if len(valid_ir) > 0:
            ir_by_pos = valid_ir.groupby("agent_position")["interest_rate"].agg(
                ["mean", "std"]
            )
            ax.errorbar(
                ir_by_pos.index,
                ir_by_pos["mean"],
                yerr=ir_by_pos["std"].fillna(0),
                marker="s",
                linewidth=2,
                markersize=8,
                capsize=5,
                color="#e74c3c",
            )
            ax.set_xlabel("Agent Position")
            ax.set_ylabel("Interest Rate (%)")
            ax.set_title("Interest Rate Evolution")

    # 1.4 Agent Agreement Analysis
    ax = axes[1, 0]
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
    if len(df_final) > 0 and df_final["approval_type"].notna().any():
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
            colors=[
                colors_type.get(str(t).upper(), "#95a5a6") for t in type_counts.index
            ],
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
            yerr=conf_by_pos["std"].fillna(0),
            marker="D",
            linewidth=2,
            markersize=8,
            capsize=5,
            color="#27ae60",
        )
        ax.set_xlabel("Agent Position")
        ax.set_ylabel("Confidence Probability")
        ax.set_title("Confidence Evolution")

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
            f"Qwen - Per-Agent Analysis: {agents_display}",
            fontsize=12,
            fontweight="bold",
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
                valid_ir = df_agent_approved["interest_rate"].dropna()
                valid_ir = valid_ir[valid_ir.between(0, 50)]
                if len(valid_ir) > 0:
                    sns.histplot(
                        valid_ir,
                        ax=ax,
                        bins=15,
                        color="#e74c3c",
                    )
                    ax.axvline(
                        valid_ir.mean(),
                        color="black",
                        linestyle="--",
                    )
            ax.set_xlabel("Interest Rate (%)" if i == n_agents // 2 else "")
            ax.set_ylabel("Count" if i == 0 else "")

        plt.tight_layout()
        plt.savefig(seq_dir / "per_agent_analysis.png", dpi=300, bbox_inches="tight")
        plt.savefig(seq_dir / "per_agent_analysis.pdf", bbox_inches="tight")
        plt.close()

    # Figure 3: Decision Flow Heatmap (Approval Rate by Position and Ethnicity)
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
        ax.set_title(
            f"Qwen - Approval Rate by Position and Ethnicity\n{agents_display}"
        )

    plt.tight_layout()
    plt.savefig(seq_dir / "decision_flow.png", dpi=300, bbox_inches="tight")
    plt.savefig(seq_dir / "decision_flow.pdf", bbox_inches="tight")
    plt.close()

    # Figure 4: First Agent vs Final Decision Transition Matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Qwen - First Agent → Final Decision Analysis\n{agents_display}",
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
            f"Qwen - First Agent → Final Decision by Ethnicity\n{agents_display}",
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
            for r in ["approve", "deny"]:
                if r not in eth_counts.index:
                    eth_counts.loc[r] = 0
            eth_counts = eth_counts.reindex(
                index=["approve", "deny"], columns=["approve", "deny"], fill_value=0
            )

            sns.heatmap(
                eth_counts,
                annot=True,
                fmt="d",
                cmap="RdYlGn",
                ax=ax,
                cbar=False,
                xticklabels=["Approve", "Deny"],
                yticklabels=["Approve", "Deny"],
            )
            ax.set_xlabel("Final")
            ax.set_ylabel("First" if col == 0 else "")
            ax.set_title(ethnicity.replace("_Signal", ""))

        # Hide extra subplots
        for idx in range(n_eth, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.savefig(
            seq_dir / "first_vs_final_by_ethnicity.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(seq_dir / "first_vs_final_by_ethnicity.pdf", bbox_inches="tight")
        plt.close()

    # Save summary
    chain_decisions_nunique = df_valid.groupby("chain_id")[
        "approval_decision"
    ].nunique()

    summary = {
        "ordering": ordering,
        "agents_display": agents_display,
        "total_chains": len(df_final),
        "total_records": len(df_valid),
        "final_approval_rate": (
            float(df_final["approved"].mean()) if len(df_final) > 0 else None
        ),
        "avg_final_interest_rate": (
            float(
                df_approved[df_approved["is_final_agent"] == True][
                    "interest_rate"
                ].mean()
            )
            if len(df_approved) > 0 and df_approved["interest_rate"].notna().any()
            else None
        ),
        "approval_rate_by_position": df_valid.groupby("agent_position")["approved"]
        .mean()
        .to_dict(),
        "final_ethnicity_approval_rates": (
            df_final.groupby("ethnicity_signal")["approved"].mean().to_dict()
            if len(df_final) > 0 and df_final["ethnicity_signal"].notna().any()
            else {}
        ),
        "chain_agreement_rate": (
            float((chain_decisions_nunique == 1).mean())
            if len(chain_decisions_nunique) > 0
            else None
        ),
    }

    with open(seq_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved sequential analysis for {ordering}")


def create_overall_comparison(df: pd.DataFrame, output_dir: Path):
    """Create overall comparison across all Qwen orderings."""
    overall_dir = output_dir / "overall"
    overall_dir.mkdir(parents=True, exist_ok=True)

    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    if len(df_valid) == 0:
        print("  No valid data for overall analysis")
        return

    # Get final decisions for each ordering
    df_final = df_valid[df_valid["is_final_agent"] == True].copy()

    # Figure 1: Approval Rate by Ordering
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Qwen Validation - Overall Analysis\n(n={len(df_final):,} chains across {df['agent_order'].nunique()} orderings)",
        fontsize=14,
        fontweight="bold",
    )

    # 1.1 Final approval rate by ordering
    ax = axes[0]
    ordering_rates = df_final.groupby("agent_order")["approved"].mean().sort_values()
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ordering_rates)))
    bars = ax.barh(range(len(ordering_rates)), ordering_rates.values, color=colors)
    ax.set_yticks(range(len(ordering_rates)))
    ax.set_yticklabels(
        [o.replace("_", " → ") for o in ordering_rates.index], fontsize=9
    )
    ax.set_xlabel("Final Approval Rate")
    ax.set_title("Final Approval Rate by Sequential Ordering")
    ax.set_xlim(0, 1)

    for i, (ordering, rate) in enumerate(ordering_rates.items()):
        count = len(df_final[df_final["agent_order"] == ordering])
        ax.text(rate + 0.01, i, f"{rate:.1%} (n={count})", va="center", fontsize=9)

    # 1.2 Approval rate by ethnicity across all orderings
    ax = axes[1]
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
        ax.set_title("Overall Final Approval Rate by Ethnicity")
        ax.set_xlim(0, 1)
        for bar, rate in zip(bars, eth_rates.values):
            ax.text(
                rate + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}",
                va="center",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(overall_dir / "overall_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(overall_dir / "overall_comparison.pdf", bbox_inches="tight")
    plt.close()

    # Figure 2: Heatmap - Approval Rates by Agent Position Across All Orderings
    fig, ax = plt.subplots(figsize=(14, 7))

    # Create pivot table: position x ordering
    position_ordering_rates = df_valid.pivot_table(
        values="approved", index="agent_position", columns="agent_order", aggfunc="mean"
    )

    if not position_ordering_rates.empty:
        # Sort columns by final approval rate (descending)
        final_rates = (
            df_final.groupby("agent_order")["approved"]
            .mean()
            .sort_values(ascending=False)
        )
        position_ordering_rates = position_ordering_rates[final_rates.index]

        # Clean column labels - use abbreviated format
        column_labels = []
        for col in position_ordering_rates.columns:
            # Replace with short names
            short = col.replace("Consumer_Advocate", "CA")
            short = short.replace("Data_Science", "DS")
            short = short.replace("Regulatory", "Reg")
            short = short.replace("Risk_Manager", "RM")
            short = short.replace("_", "\n")
            column_labels.append(short)

        # Create position labels
        position_labels = [
            (
                f"Position {i}\n(First)"
                if i == 1
                else (
                    f"Position {i}\n(Final)"
                    if i == position_ordering_rates.index.max()
                    else f"Position {i}"
                )
            )
            for i in position_ordering_rates.index
        ]

        sns.heatmap(
            position_ordering_rates * 100,  # Convert to percentage
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            ax=ax,
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Approval Rate (%)"},
            xticklabels=column_labels,
            yticklabels=position_labels,
            linewidths=0.5,
            linecolor="white",
        )

        # Calculate theoretical benchmark (if available in data)
        overall_approval = df_final["approved"].mean() * 100

        ax.set_xlabel("Agent Ordering", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel("Agent Position in Chain", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Qwen Model: Approval Rates by Agent Position Across All Orderings\n"
            f"({overall_approval:.1f}% = Overall Average)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Rotate x-axis labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(
        overall_dir / "position_ordering_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(overall_dir / "position_ordering_heatmap.pdf", bbox_inches="tight")
    plt.close()

    # Save summary
    summary = {
        "total_records": len(df_valid),
        "total_chains": len(df_final),
        "num_orderings": df["agent_order"].nunique(),
        "overall_approval_rate": float(df_final["approved"].mean()),
        "approval_rate_by_ordering": df_final.groupby("agent_order")["approved"]
        .mean()
        .to_dict(),
        "approval_rate_by_ethnicity": (
            df_final.groupby("ethnicity_signal")["approved"].mean().to_dict()
            if df_final["ethnicity_signal"].notna().any()
            else {}
        ),
    }

    with open(overall_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved overall comparison")


def main():
    """Generate Qwen validation analysis plots."""
    print("=" * 60)
    print("Qwen Validation Analysis - Plot Generation")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading Qwen validation data...")
    df = load_qwen_data("qwen_validation")
    print(f"Loaded {len(df):,} records")
    print(f"Unique orderings: {df['agent_order'].nunique()}")

    output_dir = Path("plotting/outputs/qwen_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate sequential analysis for each ordering
    print("\n[2/3] Generating sequential ordering analysis...")
    orderings = df["agent_order"].unique()
    print(f"Found {len(orderings)} sequential orderings")

    for ordering in orderings:
        create_sequential_analysis(df, ordering, output_dir)

    # Generate overall comparison
    print("\n[3/3] Generating overall comparison...")
    create_overall_comparison(df, output_dir)

    # Count generated files
    sequential_files = (
        list((output_dir / "sequential").rglob("*.png"))
        if (output_dir / "sequential").exists()
        else []
    )
    overall_files = (
        list((output_dir / "overall").rglob("*.png"))
        if (output_dir / "overall").exists()
        else []
    )

    print("\n" + "=" * 60)
    print("Qwen Validation Analysis Complete!")
    print("=" * 60)
    print(f"  Sequential plots: {len(sequential_files)} files")
    print(f"  Overall plots: {len(overall_files)} files")
    print(f"  Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
