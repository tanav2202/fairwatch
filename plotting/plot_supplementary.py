"""
Supplementary Analysis Plots
Additional detailed analysis plots for appendix/supplementary materials.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import sys

sys.path.append(str(Path(__file__).parent))
from data_loader import load_data, FairWatchDataLoader

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_income_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze approval rates by income levels."""
    df = df.copy()
    df["income_tier"] = pd.cut(
        df["income"],
        bins=[0, 35000, 65000, 120000, 500000],
        labels=[
            "Low\n(<35K)",
            "Medium\n(35-65K)",
            "High\n(65-120K)",
            "Very High\n(>120K)",
        ],
    )

    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Approval rate by income
    ax = axes[0, 0]
    stats = (
        df_valid.groupby("income_tier")["approved"].agg(["mean", "count"]).reset_index()
    )
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(stats)))
    bars = ax.bar(range(len(stats)), stats["mean"], color=colors)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats["income_tier"])
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rate by Income Tier")
    ax.set_ylim(0, 1)

    # 2. Interest rate by income
    ax = axes[0, 1]
    df_approved = df_valid[df_valid["approval_decision"] == "approve"]
    df_plot = df_approved[["income_tier", "interest_rate"]].dropna()

    if len(df_plot) > 0:
        sns.boxplot(
            data=df_plot, x="income_tier", y="interest_rate", ax=ax, palette="Greens"
        )
    ax.set_xlabel("Income Tier")
    ax.set_ylabel("Interest Rate (%)")
    ax.set_title("Interest Rate by Income Tier")

    # 3. Income x Ethnicity interaction
    ax = axes[1, 0]
    df_eth = df_valid[df_valid["ethnicity_signal"] != "Unknown"]
    pivot = df_eth.pivot_table(
        values="approved",
        index="income_tier",
        columns="ethnicity_signal",
        aggfunc="mean",
    )
    pivot.columns = [c.replace("_Signal", "") for c in pivot.columns]

    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Income Tier")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval by Income and Ethnicity")
    ax.legend(title="Ethnicity", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, 1)

    # 4. DTI ratio distribution
    ax = axes[1, 1]
    df_valid["dti_bucket"] = pd.cut(
        df_valid["dti_ratio"],
        bins=[0, 0.2, 0.35, 0.5, 1.0],
        labels=["<20%", "20-35%", "35-50%", ">50%"],
    )
    stats_dti = df_valid.groupby("dti_bucket")["approved"].mean()

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(stats_dti)))
    ax.bar(stats_dti.index, stats_dti.values, color=colors)
    ax.set_xlabel("Debt-to-Income Ratio")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rate by DTI Ratio")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "income_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "income_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: income_analysis.png")


def plot_loan_amount_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze decisions by loan amount."""
    df = df.copy()
    df["loan_tier"] = pd.cut(
        df["loan_amount"],
        bins=[0, 35000, 65000, 120000, 400000],
        labels=[
            "Small\n(<35K)",
            "Medium\n(35-65K)",
            "Large\n(65-120K)",
            "Very Large\n(>120K)",
        ],
    )

    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Approval by loan amount
    ax = axes[0]
    stats = df_valid.groupby("loan_tier")["approved"].mean()
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(stats)))
    ax.bar(stats.index, stats.values, color=colors)
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rate by Loan Amount")
    ax.set_ylim(0, 1)

    # 2. Interest rate by loan amount
    ax = axes[1]
    df_approved = df_valid[df_valid["approval_decision"] == "approve"]
    stats_ir = (
        df_approved.groupby("loan_tier")["interest_rate"]
        .agg(["mean", "std"])
        .reset_index()
    )

    ax.bar(
        range(len(stats_ir)),
        stats_ir["mean"],
        yerr=stats_ir["std"],
        color="steelblue",
        capsize=4,
    )
    ax.set_xticks(range(len(stats_ir)))
    ax.set_xticklabels(stats_ir["loan_tier"])
    ax.set_ylabel("Average Interest Rate (%)")
    ax.set_title("Interest Rate by Loan Amount")

    # 3. Loan-to-income ratio impact
    ax = axes[2]
    df_valid["lti_ratio"] = df_valid["loan_amount"] / df_valid["income"]
    df_valid["lti_bucket"] = pd.cut(
        df_valid["lti_ratio"],
        bins=[0, 1, 2, 3, 10],
        labels=["<1x", "1-2x", "2-3x", ">3x"],
    )

    stats_lti = df_valid.groupby("lti_bucket")["approved"].mean()
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(stats_lti)))
    ax.bar(stats_lti.index, stats_lti.values, color=colors)
    ax.set_xlabel("Loan-to-Income Ratio")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval by Loan-to-Income Ratio")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "loan_amount_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "loan_amount_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: loan_amount_analysis.png")


def plot_reasoning_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze decision distributions by approval type."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Approval type by experiment mode
    ax = axes[0, 0]
    df_valid = df[df["approval_decision"].isin(["approve", "deny"])]
    pivot = pd.crosstab(
        df_valid["experiment_type"], df_valid["approval_type"], normalize="index"
    )

    approval_types = ["STANDARD_TERMS", "SUBOPTIMAL_TERMS", "MANUAL_REVIEW", "DENIAL"]
    colors = {
        "STANDARD_TERMS": "#27ae60",
        "SUBOPTIMAL_TERMS": "#f39c12",
        "MANUAL_REVIEW": "#3498db",
        "DENIAL": "#e74c3c",
    }

    pivot = pivot[[c for c in approval_types if c in pivot.columns]]
    pivot.plot(
        kind="bar", stacked=True, ax=ax, color=[colors[c] for c in pivot.columns]
    )
    ax.set_ylabel("Proportion")
    ax.set_title("Approval Type Distribution by Mode")
    ax.legend(title="Type", bbox_to_anchor=(1.02, 1))
    ax.tick_params(axis="x", rotation=0)

    # 2. Approval type by credit score
    ax = axes[0, 1]
    df_valid = df_valid.copy()
    df_valid["credit_tier"] = pd.cut(
        df_valid["credit_score"],
        bins=[0, 650, 720, 850],
        labels=["Low", "Medium", "High"],
    )

    pivot_credit = pd.crosstab(
        df_valid["credit_tier"], df_valid["approval_type"], normalize="index"
    )
    pivot_credit = pivot_credit[
        [c for c in approval_types if c in pivot_credit.columns]
    ]
    pivot_credit.plot(
        kind="bar", stacked=True, ax=ax, color=[colors[c] for c in pivot_credit.columns]
    )
    ax.set_ylabel("Proportion")
    ax.set_title("Approval Type by Credit Score")
    ax.legend(title="Type")
    ax.tick_params(axis="x", rotation=0)

    # 3. Confidence by approval type
    ax = axes[1, 0]
    pivot_conf = pd.crosstab(
        df_valid["approval_type"], df_valid["confidence_level"], normalize="index"
    )
    conf_levels = ["high", "medium", "low"]
    pivot_conf = pivot_conf[[c for c in conf_levels if c in pivot_conf.columns]]

    pivot_conf.plot(
        kind="bar", stacked=True, ax=ax, color=["#27ae60", "#f39c12", "#e74c3c"]
    )
    ax.set_ylabel("Proportion")
    ax.set_title("Confidence Level by Approval Type")
    ax.legend(title="Confidence")
    ax.tick_params(axis="x", rotation=45)

    # 4. Interest rate by approval type (for approved)
    ax = axes[1, 1]
    df_approved = df_valid[df_valid["approval_decision"] == "approve"]
    df_plot = df_approved[["approval_type", "interest_rate"]].dropna()

    if len(df_plot) > 0:
        sns.boxplot(
            data=df_plot,
            x="approval_type",
            y="interest_rate",
            ax=ax,
            order=["STANDARD_TERMS", "SUBOPTIMAL_TERMS", "MANUAL_REVIEW"],
            palette=["#27ae60", "#f39c12", "#3498db"],
        )
    ax.set_xlabel("Approval Type")
    ax.set_ylabel("Interest Rate (%)")
    ax.set_title("Interest Rate by Approval Type")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "reasoning_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "reasoning_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: reasoning_analysis.png")


def plot_name_based_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze decisions by applicant names (proxies for demographics)."""
    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Approval rate by name
    ax = axes[0]
    name_stats = (
        df_valid.groupby("name")
        .agg({"approved": ["mean", "count"], "ethnicity_signal": "first"})
        .round(4)
    )
    name_stats.columns = ["rate", "count", "ethnicity"]
    name_stats = name_stats[name_stats["count"] >= 100].sort_values(
        "rate", ascending=False
    )

    if len(name_stats) > 0:
        eth_colors = {
            "White_Signal": "#3498db",
            "Black_Signal": "#e74c3c",
            "Hispanic_Signal": "#2ecc71",
            "Asian_Signal": "#f39c12",
            "Unknown": "#95a5a6",
        }
        colors = [eth_colors.get(e, "#95a5a6") for e in name_stats["ethnicity"]]

        bars = ax.barh(range(len(name_stats)), name_stats["rate"], color=colors)
        ax.set_yticks(range(len(name_stats)))
        ax.set_yticklabels(name_stats.index)
        ax.set_xlabel("Approval Rate")
        ax.set_title("Approval Rate by Applicant Name")
        ax.set_xlim(0, 1)

        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=c, label=e.replace("_Signal", ""))
            for e, c in eth_colors.items()
            if e != "Unknown"
        ]
        ax.legend(handles=legend_elements, title="Ethnicity", loc="lower right")

    # 2. Interest rate by name
    ax = axes[1]
    df_approved = df_valid[df_valid["approval_decision"] == "approve"]
    ir_by_name = df_approved.groupby("name")["interest_rate"].agg(
        ["mean", "std", "count"]
    )
    ir_by_name = ir_by_name[ir_by_name["count"] >= 50].sort_values("mean")

    if len(ir_by_name) > 0:
        ax.barh(
            range(len(ir_by_name)),
            ir_by_name["mean"],
            xerr=ir_by_name["std"] / np.sqrt(ir_by_name["count"]),
            color="steelblue",
            capsize=3,
        )
        ax.set_yticks(range(len(ir_by_name)))
        ax.set_yticklabels(ir_by_name.index)
        ax.set_xlabel("Average Interest Rate (%)")
        ax.set_title("Average Interest Rate by Name")

    plt.tight_layout()
    plt.savefig(output_dir / "name_based_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "name_based_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: name_based_analysis.png")


def plot_detailed_ethnicity_analysis(df: pd.DataFrame, output_dir: Path):
    """Detailed ethnicity-based analysis."""
    df_valid = df[
        (df["approval_decision"].isin(["approve", "deny"]))
        & (df["ethnicity_signal"] != "Unknown")
    ].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)
    df_valid["ethnicity"] = df_valid["ethnicity_signal"].str.replace("_Signal", "")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ethnicities = ["White", "Black", "Hispanic", "Asian"]
    colors = {
        "White": "#3498db",
        "Black": "#e74c3c",
        "Hispanic": "#2ecc71",
        "Asian": "#f39c12",
    }

    # 1. Approval rate with error bars
    ax = axes[0, 0]
    stats = df_valid.groupby("ethnicity").agg({"approved": ["mean", "std", "count"]})
    stats.columns = ["rate", "std", "count"]
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats = stats.reindex([e for e in ethnicities if e in stats.index])

    bars = ax.bar(
        stats.index,
        stats["rate"],
        yerr=stats["se"],
        color=[colors.get(e, "#95a5a6") for e in stats.index],
        capsize=5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Approval Rate")
    ax.set_title("Overall Approval Rate by Ethnicity")
    ax.set_ylim(0, 1)

    for bar, rate, se in zip(bars, stats["rate"], stats["se"]):
        ax.annotate(
            f"{rate:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.02),
            ha="center",
        )

    # 2. Interest rate distribution
    ax = axes[0, 1]
    df_approved = df_valid[df_valid["approval_decision"] == "approve"]

    for eth in ethnicities:
        data = df_approved[df_approved["ethnicity"] == eth]["interest_rate"].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.5, label=eth, color=colors[eth])

    ax.set_xlabel("Interest Rate (%)")
    ax.set_ylabel("Count")
    ax.set_title("Interest Rate Distribution by Ethnicity")
    ax.legend()

    # 3. Approval type by ethnicity
    ax = axes[1, 0]
    pivot_at = pd.crosstab(
        df_valid["ethnicity"], df_valid["approval_type"], normalize="index"
    )
    approval_types = ["STANDARD_TERMS", "SUBOPTIMAL_TERMS", "DENIAL"]
    pivot_at = pivot_at[[c for c in approval_types if c in pivot_at.columns]]
    pivot_at = pivot_at.reindex([e for e in ethnicities if e in pivot_at.index])

    pivot_at.plot(
        kind="bar", stacked=True, ax=ax, color=["#27ae60", "#f39c12", "#e74c3c"]
    )
    ax.set_ylabel("Proportion")
    ax.set_title("Approval Type by Ethnicity")
    ax.legend(title="Type")
    ax.tick_params(axis="x", rotation=0)

    # 4. Confidence by ethnicity
    ax = axes[1, 1]
    df_valid["conf_score"] = df_valid["confidence_level"].map(
        {"high": 3, "medium": 2, "low": 1}
    )
    conf_by_eth = df_valid.groupby("ethnicity")["conf_score"].mean()
    conf_by_eth = conf_by_eth.reindex(
        [e for e in ethnicities if e in conf_by_eth.index]
    )

    bars = ax.bar(
        conf_by_eth.index,
        conf_by_eth.values,
        color=[colors.get(e, "#95a5a6") for e in conf_by_eth.index],
    )
    ax.set_ylabel("Average Confidence Score")
    ax.set_title("Average Confidence by Ethnicity")
    ax.set_ylim(0, 3.5)
    ax.axhline(
        y=2.5,
        color="green",
        linestyle="--",
        alpha=0.5,
        label="High confidence threshold",
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        output_dir / "detailed_ethnicity_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "detailed_ethnicity_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: detailed_ethnicity_analysis.png")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path):
    """Plot correlation matrix of numerical features."""
    df_num = df.copy()

    # Create numerical columns
    df_num["approved"] = (df_num["approval_decision"] == "approve").astype(float)
    df_num["confidence_score"] = df_num["confidence_level"].map(
        {"high": 3, "medium": 2, "low": 1}
    )

    # Select numerical columns
    num_cols = [
        "credit_score",
        "income",
        "loan_amount",
        "age",
        "dti_ratio",
        "interest_rate",
        "confidence_probability",
        "confidence_score",
        "approved",
    ]

    df_corr = df_num[num_cols].dropna()

    if len(df_corr) < 100:
        print("Insufficient data for correlation matrix")
        return

    corr_matrix = df_corr.corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        mask=mask,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Correlation"},
        linewidths=0.5,
    )

    ax.set_title("Correlation Matrix of Key Variables", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "correlation_matrix.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: correlation_matrix.png")


def main():
    """Run all supplementary analysis plots."""
    output_dir = Path("plotting/outputs/supplementary")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, _ = load_data("70bresults")
    print(f"Loaded {len(df)} records")

    print("\nGenerating supplementary plots...")
    plot_income_analysis(df, output_dir)
    plot_loan_amount_analysis(df, output_dir)
    plot_reasoning_analysis(df, output_dir)
    plot_name_based_analysis(df, output_dir)
    plot_detailed_ethnicity_analysis(df, output_dir)
    plot_correlation_matrix(df, output_dir)

    print(f"\nAll supplementary plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
