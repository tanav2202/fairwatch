"""
Demographic Bias Analysis Plots
Analyzes and visualizes bias across ethnicity, age, and visa status.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from data_loader import load_data, FairWatchDataLoader

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def calculate_approval_rates(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Calculate approval rates by group."""
    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    stats = (
        df_valid.groupby(group_col)
        .agg({"approved": ["mean", "sum", "count"], "interest_rate": ["mean", "std"]})
        .round(4)
    )

    stats.columns = [
        "approval_rate",
        "approvals",
        "total",
        "avg_interest_rate",
        "std_interest_rate",
    ]
    return stats.reset_index()


def plot_ethnicity_approval_rates(df: pd.DataFrame, output_dir: Path):
    """Plot approval rates by ethnicity for each experiment type."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    exp_types = ["baseline", "sequential", "parallel"]

    for ax, exp_type in zip(axes, exp_types):
        df_exp = df[df["experiment_type"] == exp_type]
        if df_exp.empty:
            ax.set_title(f"{exp_type.title()} - No Data")
            continue

        stats = calculate_approval_rates(df_exp, "ethnicity_signal")
        stats = stats[stats["ethnicity_signal"] != "Unknown"]

        if stats.empty:
            continue

        colors = {
            "White_Signal": "#3498db",
            "Black_Signal": "#e74c3c",
            "Hispanic_Signal": "#2ecc71",
            "Asian_Signal": "#f39c12",
        }
        bar_colors = [colors.get(e, "#95a5a6") for e in stats["ethnicity_signal"]]

        bars = ax.bar(range(len(stats)), stats["approval_rate"], color=bar_colors)
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(
            [e.replace("_Signal", "") for e in stats["ethnicity_signal"]], rotation=45
        )
        ax.set_ylabel("Approval Rate")
        ax.set_title(f"{exp_type.title()} - Approval by Ethnicity")
        ax.set_ylim(0, 1)

        # Add value labels
        for bar, rate in zip(bars, stats["approval_rate"]):
            ax.annotate(
                f"{rate:.2%}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(
        output_dir / "ethnicity_approval_rates.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "ethnicity_approval_rates.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: ethnicity_approval_rates.png")


def plot_ethnicity_interest_rates(df: pd.DataFrame, output_dir: Path):
    """Plot interest rates by ethnicity using box plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    exp_types = ["baseline", "sequential", "parallel"]

    for ax, exp_type in zip(axes, exp_types):
        df_exp = df[
            (df["experiment_type"] == exp_type)
            & (df["approval_decision"] == "approve")
            & (df["ethnicity_signal"] != "Unknown")
        ]

        if df_exp.empty:
            ax.set_title(f"{exp_type.title()} - No Data")
            continue

        df_plot = df_exp[["ethnicity_signal", "interest_rate"]].dropna()
        df_plot["ethnicity"] = df_plot["ethnicity_signal"].str.replace("_Signal", "")

        sns.boxplot(
            data=df_plot, x="ethnicity", y="interest_rate", ax=ax, palette="husl"
        )
        ax.set_xlabel("Ethnicity")
        ax.set_ylabel("Interest Rate (%)")
        ax.set_title(f"{exp_type.title()} - Interest Rates by Ethnicity")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        output_dir / "ethnicity_interest_rates.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "ethnicity_interest_rates.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: ethnicity_interest_rates.png")


def plot_age_bias(df: pd.DataFrame, output_dir: Path):
    """Plot approval rates and interest rates by age groups."""
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    exp_types = ["baseline", "sequential", "parallel"]

    # Approval rates
    for ax, exp_type in zip(axes[0], exp_types):
        df_exp = df[df["experiment_type"] == exp_type]
        if df_exp.empty:
            continue

        stats = calculate_approval_rates(df_exp, "age_group")

        ax.bar(range(len(stats)), stats["approval_rate"], color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(stats["age_group"])
        ax.set_ylabel("Approval Rate")
        ax.set_title(f"{exp_type.title()} - Approval by Age")
        ax.set_ylim(0, 1)

    # Interest rates
    for ax, exp_type in zip(axes[1], exp_types):
        df_exp = df[
            (df["experiment_type"] == exp_type) & (df["approval_decision"] == "approve")
        ]
        if df_exp.empty:
            continue

        df_plot = df_exp[["age_group", "interest_rate"]].dropna()

        sns.boxplot(
            data=df_plot, x="age_group", y="interest_rate", ax=ax, palette="coolwarm"
        )
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Interest Rate (%)")
        ax.set_title(f"{exp_type.title()} - Interest Rates by Age")

    plt.tight_layout()
    plt.savefig(output_dir / "age_bias_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "age_bias_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: age_bias_analysis.png")


def plot_visa_status_bias(df: pd.DataFrame, output_dir: Path):
    """Plot approval rates by visa status."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    exp_types = ["baseline", "sequential", "parallel"]

    for ax, exp_type in zip(axes, exp_types):
        df_exp = df[df["experiment_type"] == exp_type]
        if df_exp.empty:
            continue

        stats = calculate_approval_rates(df_exp, "visa_status")
        stats = stats.dropna(subset=["visa_status"])

        if stats.empty:
            continue

        # Sort by approval rate
        stats = stats.sort_values("approval_rate", ascending=True)

        colors = plt.cm.RdYlGn(stats["approval_rate"])
        bars = ax.barh(range(len(stats)), stats["approval_rate"], color=colors)
        ax.set_yticks(range(len(stats)))
        ax.set_yticklabels(stats["visa_status"], fontsize=9)
        ax.set_xlabel("Approval Rate")
        ax.set_title(f"{exp_type.title()} - Approval by Visa Status")
        ax.set_xlim(0, 1)

        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars, stats["approval_rate"])):
            ax.annotate(f"{rate:.1%}", xy=(rate + 0.02, i), va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "visa_status_bias.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "visa_status_bias.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: visa_status_bias.png")


def plot_credit_score_impact(df: pd.DataFrame, output_dir: Path):
    """Plot approval rates and interest rates by credit score ranges."""
    df = df.copy()
    df["credit_tier"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 620, 680, 720, 780, 850],
        labels=[
            "Poor (<580)",
            "Fair (580-620)",
            "Good (620-680)",
            "Very Good (680-720)",
            "Excellent (720-780)",
            "Exceptional (780+)",
        ],
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Overall approval rate by credit tier
    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    ax = axes[0, 0]
    stats = (
        df_valid.groupby("credit_tier")["approved"].agg(["mean", "count"]).reset_index()
    )
    colors = plt.cm.RdYlGn(stats["mean"])
    bars = ax.bar(range(len(stats)), stats["mean"], color=colors)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats["credit_tier"], rotation=45, ha="right")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rate by Credit Score Tier")
    ax.set_ylim(0, 1)

    # Interest rate by credit tier
    ax = axes[0, 1]
    df_approved = df[(df["approval_decision"] == "approve")]
    df_plot = df_approved[["credit_tier", "interest_rate"]].dropna()

    sns.boxplot(
        data=df_plot, x="credit_tier", y="interest_rate", ax=ax, palette="coolwarm_r"
    )
    ax.set_xlabel("Credit Tier")
    ax.set_ylabel("Interest Rate (%)")
    ax.set_title("Interest Rate by Credit Score Tier")
    ax.tick_params(axis="x", rotation=45)

    # Approval rates by credit tier and ethnicity
    ax = axes[1, 0]
    df_eth = df_valid[df_valid["ethnicity_signal"] != "Unknown"]
    pivot = df_eth.pivot_table(
        values="approved",
        index="credit_tier",
        columns="ethnicity_signal",
        aggfunc="mean",
    )
    pivot.columns = [c.replace("_Signal", "") for c in pivot.columns]
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Credit Tier")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rate by Credit Tier & Ethnicity")
    ax.legend(title="Ethnicity", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1)

    # Interest rate distribution by experiment type
    ax = axes[1, 1]
    df_approved = df[(df["approval_decision"] == "approve")]
    sns.boxplot(
        data=df_approved, x="experiment_type", y="interest_rate", ax=ax, palette="Set2"
    )
    ax.set_xlabel("Experiment Type")
    ax.set_ylabel("Interest Rate (%)")
    ax.set_title("Interest Rate Distribution by Experiment Type")

    plt.tight_layout()
    plt.savefig(output_dir / "credit_score_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "credit_score_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: credit_score_analysis.png")


def plot_intersectional_bias(df: pd.DataFrame, output_dir: Path):
    """Plot intersectional bias analysis (ethnicity + credit score)."""
    df = df.copy()
    df_valid = df[
        (df["approval_decision"].isin(["approve", "deny"]))
        & (df["ethnicity_signal"] != "Unknown")
    ].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)
    df_valid["credit_tier"] = pd.cut(
        df_valid["credit_score"],
        bins=[0, 650, 720, 850],
        labels=["Low (<650)", "Medium (650-720)", "High (720+)"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    exp_types = ["baseline", "sequential", "parallel"]

    for ax, exp_type in zip(axes, exp_types):
        df_exp = df_valid[df_valid["experiment_type"] == exp_type]
        if df_exp.empty:
            continue

        pivot = df_exp.pivot_table(
            values="approved",
            index="credit_tier",
            columns="ethnicity_signal",
            aggfunc="mean",
        )
        pivot.columns = [c.replace("_Signal", "") for c in pivot.columns]

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2%",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Approval Rate"},
        )
        ax.set_title(f"{exp_type.title()} - Intersectional Approval Rates")
        ax.set_xlabel("Ethnicity")
        ax.set_ylabel("Credit Tier")

    plt.tight_layout()
    plt.savefig(output_dir / "intersectional_bias.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "intersectional_bias.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: intersectional_bias.png")


def plot_bias_disparity_metrics(df: pd.DataFrame, output_dir: Path):
    """Calculate and plot disparity metrics across demographic groups."""
    df_valid = df[
        (df["approval_decision"].isin(["approve", "deny"]))
        & (df["ethnicity_signal"] != "Unknown")
    ].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    # Calculate disparity ratio (min/max approval rate)
    disparities = []

    for exp_type in ["baseline", "sequential", "parallel"]:
        df_exp = df_valid[df_valid["experiment_type"] == exp_type]
        if df_exp.empty:
            continue

        eth_rates = df_exp.groupby("ethnicity_signal")["approved"].mean()
        if len(eth_rates) >= 2:
            disparity = eth_rates.min() / eth_rates.max() if eth_rates.max() > 0 else 0
            disparities.append(
                {
                    "experiment_type": exp_type,
                    "metric": "Ethnicity Disparity Ratio",
                    "value": disparity,
                }
            )

        # Age disparity
        df_exp["age_group"] = pd.cut(
            df_exp["age"], bins=[0, 35, 55, 100], labels=["Young", "Middle", "Senior"]
        )
        age_rates = df_exp.groupby("age_group")["approved"].mean()
        if len(age_rates) >= 2:
            disparity = age_rates.min() / age_rates.max() if age_rates.max() > 0 else 0
            disparities.append(
                {
                    "experiment_type": exp_type,
                    "metric": "Age Disparity Ratio",
                    "value": disparity,
                }
            )

    if not disparities:
        print("No disparity data available")
        return

    df_disp = pd.DataFrame(disparities)

    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = df_disp.pivot(index="metric", columns="experiment_type", values="value")
    pivot.plot(kind="bar", ax=ax, width=0.8, color=["#3498db", "#e74c3c", "#2ecc71"])

    ax.axhline(y=0.8, color="green", linestyle="--", label="Fair threshold (0.8)")
    ax.set_ylabel("Disparity Ratio (min/max)")
    ax.set_xlabel("")
    ax.set_title(
        "Demographic Disparity Ratios\n(Higher is Fairer, 1.0 = Perfect Parity)"
    )
    ax.set_ylim(0, 1.1)
    ax.legend(title="Experiment Type")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "disparity_metrics.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "disparity_metrics.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: disparity_metrics.png")


def main():
    """Run all demographic bias analysis plots."""
    output_dir = Path("plotting/outputs/demographic_bias")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, _ = load_data("70bresults")
    print(f"Loaded {len(df)} records")

    print("\nGenerating demographic bias plots...")
    plot_ethnicity_approval_rates(df, output_dir)
    plot_ethnicity_interest_rates(df, output_dir)
    plot_age_bias(df, output_dir)
    plot_visa_status_bias(df, output_dir)
    plot_credit_score_impact(df, output_dir)
    plot_intersectional_bias(df, output_dir)
    plot_bias_disparity_metrics(df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
