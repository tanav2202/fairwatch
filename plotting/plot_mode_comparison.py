"""
Parallel vs Sequential vs Baseline Comparison Plots
Comprehensive comparison between different evaluation modes.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
import sys

sys.path.append(str(Path(__file__).parent))
from data_loader import load_data, FairWatchDataLoader

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_mode_comparison_overview(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive overview comparing all modes."""
    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    modes = ["baseline", "sequential", "parallel"]
    colors = {"baseline": "#3498db", "sequential": "#e74c3c", "parallel": "#2ecc71"}

    # 1. Approval rate comparison
    ax1 = fig.add_subplot(gs[0, 0])
    stats = df_valid.groupby("experiment_type")["approved"].mean()
    bars = ax1.bar(
        modes, [stats.get(m, 0) for m in modes], color=[colors[m] for m in modes]
    )
    ax1.set_ylabel("Approval Rate")
    ax1.set_title("Overall Approval Rate")
    ax1.set_ylim(0, 1)
    for bar, rate in zip(bars, [stats.get(m, 0) for m in modes]):
        ax1.annotate(
            f"{rate:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )

    # 2. Average interest rate
    ax2 = fig.add_subplot(gs[0, 1])
    df_approved = df_valid[df_valid["approval_decision"] == "approve"]
    stats_ir = df_approved.groupby("experiment_type")["interest_rate"].mean()
    bars = ax2.bar(
        modes, [stats_ir.get(m, 0) for m in modes], color=[colors[m] for m in modes]
    )
    ax2.set_ylabel("Interest Rate (%)")
    ax2.set_title("Average Interest Rate (Approved)")
    for bar, rate in zip(bars, [stats_ir.get(m, 0) for m in modes]):
        ax2.annotate(
            f"{rate:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )

    # 3. Confidence distribution
    ax3 = fig.add_subplot(gs[0, 2])
    conf_data = []
    for mode in modes:
        df_mode = df_valid[df_valid["experiment_type"] == mode]
        for level in ["high", "medium", "low"]:
            count = (df_mode["confidence_level"] == level).sum()
            total = len(df_mode)
            conf_data.append(
                {
                    "mode": mode,
                    "level": level,
                    "proportion": count / total if total > 0 else 0,
                }
            )
    df_conf = pd.DataFrame(conf_data)
    pivot_conf = df_conf.pivot(index="mode", columns="level", values="proportion")
    pivot_conf = pivot_conf.reindex(modes)[["high", "medium", "low"]]
    pivot_conf.plot(
        kind="bar", stacked=True, ax=ax3, color=["#27ae60", "#f39c12", "#e74c3c"]
    )
    ax3.set_ylabel("Proportion")
    ax3.set_title("Confidence Level Distribution")
    ax3.legend(title="Level")
    ax3.tick_params(axis="x", rotation=0)

    # 4. Approval type distribution
    ax4 = fig.add_subplot(gs[1, 0])
    approval_types = ["STANDARD_TERMS", "SUBOPTIMAL_TERMS", "MANUAL_REVIEW", "DENIAL"]
    at_colors = ["#27ae60", "#f39c12", "#3498db", "#e74c3c"]
    at_data = []
    for mode in modes:
        df_mode = df_valid[df_valid["experiment_type"] == mode]
        for at in approval_types:
            count = (df_mode["approval_type"] == at).sum()
            total = len(df_mode)
            at_data.append(
                {
                    "mode": mode,
                    "type": at,
                    "proportion": count / total if total > 0 else 0,
                }
            )
    df_at = pd.DataFrame(at_data)
    pivot_at = df_at.pivot(index="mode", columns="type", values="proportion")
    pivot_at = pivot_at.reindex(modes)
    pivot_at = pivot_at[[c for c in approval_types if c in pivot_at.columns]]
    pivot_at.plot(
        kind="bar", stacked=True, ax=ax4, color=at_colors[: len(pivot_at.columns)]
    )
    ax4.set_ylabel("Proportion")
    ax4.set_title("Approval Type Distribution")
    ax4.legend(title="Type", fontsize=8)
    ax4.tick_params(axis="x", rotation=0)

    # 5. Interest rate distribution (violin plot)
    ax5 = fig.add_subplot(gs[1, 1])
    df_plot = df_approved[["experiment_type", "interest_rate"]].dropna()
    if len(df_plot) > 0:
        sns.violinplot(
            data=df_plot,
            x="experiment_type",
            y="interest_rate",
            ax=ax5,
            palette=colors,
            order=modes,
        )
    ax5.set_xlabel("Mode")
    ax5.set_ylabel("Interest Rate (%)")
    ax5.set_title("Interest Rate Distribution")

    # 6. Confidence probability distribution
    ax6 = fig.add_subplot(gs[1, 2])
    df_plot = df_valid[["experiment_type", "confidence_probability"]].dropna()
    if len(df_plot) > 0:
        sns.boxplot(
            data=df_plot,
            x="experiment_type",
            y="confidence_probability",
            ax=ax6,
            palette=colors,
            order=modes,
        )
    ax6.set_xlabel("Mode")
    ax6.set_ylabel("Confidence Probability")
    ax6.set_title("Confidence Probability Distribution")

    # 7-9. Bias comparison across modes
    for idx, (ax_idx, group_col) in enumerate(
        [(gs[2, 0], "ethnicity_signal"), (gs[2, 1], "age"), (gs[2, 2], "credit_score")]
    ):
        ax = fig.add_subplot(ax_idx)

        if group_col == "ethnicity_signal":
            df_group = df_valid[df_valid["ethnicity_signal"] != "Unknown"]
            group_vals = df_group[group_col].unique()
        elif group_col == "age":
            df_valid["age_group"] = pd.cut(
                df_valid["age"],
                bins=[0, 35, 55, 100],
                labels=["Young", "Middle", "Senior"],
            )
            df_group = df_valid.dropna(subset=["age_group"])
            group_col = "age_group"
            group_vals = ["Young", "Middle", "Senior"]
        else:
            df_valid["credit_tier"] = pd.cut(
                df_valid["credit_score"],
                bins=[0, 650, 720, 850],
                labels=["Low", "Medium", "High"],
            )
            df_group = df_valid.dropna(subset=["credit_tier"])
            group_col = "credit_tier"
            group_vals = ["Low", "Medium", "High"]

        pivot = df_group.pivot_table(
            values="approved",
            index="experiment_type",
            columns=group_col,
            aggfunc="mean",
        )

        if group_col == "ethnicity_signal":
            pivot.columns = [c.replace("_Signal", "") for c in pivot.columns]

        pivot = pivot.reindex(modes)
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("Approval Rate")
        ax.set_title(f'Approval by {group_col.replace("_", " ").title()}')
        ax.legend(title=group_col.replace("_", " ").title(), fontsize=8)
        ax.tick_params(axis="x", rotation=0)
        ax.set_ylim(0, 1)

    plt.suptitle(
        "Multi-Agent System: Mode Comparison Overview",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.savefig(
        output_dir / "mode_comparison_overview.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "mode_comparison_overview.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: mode_comparison_overview.png")


def plot_statistical_comparison(df: pd.DataFrame, output_dir: Path):
    """Perform and visualize statistical comparisons between modes."""
    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    modes = ["baseline", "sequential", "parallel"]

    # Statistical tests
    test_results = []

    # Compare approval rates between modes
    for i, mode1 in enumerate(modes):
        for mode2 in modes[i + 1 :]:
            data1 = df_valid[df_valid["experiment_type"] == mode1]["approved"]
            data2 = df_valid[df_valid["experiment_type"] == mode2]["approved"]

            if len(data1) > 0 and len(data2) > 0:
                # Chi-square test for approval rates
                contingency = pd.crosstab(
                    df_valid["experiment_type"].isin([mode1, mode2]),
                    df_valid["approved"],
                )
                # Proportion test
                n1, n2 = len(data1), len(data2)
                p1, p2 = data1.mean(), data2.mean()

                # Z-test for proportions
                p_pooled = (data1.sum() + data2.sum()) / (n1 + n2)
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
                z_stat = (p1 - p2) / se if se > 0 else 0
                p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

                test_results.append(
                    {
                        "comparison": f"{mode1} vs {mode2}",
                        "metric": "Approval Rate",
                        "diff": p1 - p2,
                        "z_stat": z_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }
                )

    # Compare interest rates
    df_approved = df_valid[df_valid["approval_decision"] == "approve"]

    for i, mode1 in enumerate(modes):
        for mode2 in modes[i + 1 :]:
            data1 = df_approved[df_approved["experiment_type"] == mode1][
                "interest_rate"
            ].dropna()
            data2 = df_approved[df_approved["experiment_type"] == mode2][
                "interest_rate"
            ].dropna()

            if len(data1) > 10 and len(data2) > 10:
                t_stat, p_value = scipy_stats.ttest_ind(data1, data2)

                test_results.append(
                    {
                        "comparison": f"{mode1} vs {mode2}",
                        "metric": "Interest Rate",
                        "diff": data1.mean() - data2.mean(),
                        "z_stat": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }
                )

    if not test_results:
        print("Insufficient data for statistical tests")
        return

    df_tests = pd.DataFrame(test_results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Difference plot
    ax = axes[0]
    df_approval = df_tests[df_tests["metric"] == "Approval Rate"]
    if not df_approval.empty:
        colors = ["green" if s else "gray" for s in df_approval["significant"]]
        bars = ax.barh(df_approval["comparison"], df_approval["diff"], color=colors)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Difference in Approval Rate")
        ax.set_title(
            "Pairwise Approval Rate Differences\n(Green = Significant at p<0.05)"
        )

    # Add significance stars
    for i, (idx, row) in enumerate(df_approval.iterrows()):
        if row["significant"]:
            ax.annotate(
                "*" if row["p_value"] < 0.05 else "",
                xy=(row["diff"], i),
                ha="left" if row["diff"] > 0 else "right",
            )

    # 2. P-value visualization
    ax = axes[1]
    pivot = df_tests.pivot(index="comparison", columns="metric", values="p_value")

    # Create heatmap with significance threshold
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn_r",
        ax=ax,
        vmin=0,
        vmax=0.1,
        cbar_kws={"label": "p-value"},
    )
    ax.set_title("Statistical Significance (p-values)\n(Lower = More Significant)")

    # Add significance threshold line
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "statistical_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "statistical_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: statistical_comparison.png")

    # Save results to CSV
    df_tests.to_csv(output_dir / "statistical_tests.csv", index=False)
    print("Saved: statistical_tests.csv")


def plot_fairness_metrics_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare fairness metrics across different modes."""
    df_valid = df[
        (df["approval_decision"].isin(["approve", "deny"]))
        & (df["ethnicity_signal"] != "Unknown")
    ].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    modes = ["baseline", "sequential", "parallel"]

    # Calculate fairness metrics
    fairness_data = []

    for mode in modes:
        df_mode = df_valid[df_valid["experiment_type"] == mode]

        if df_mode.empty:
            continue

        # 1. Demographic Parity (approval rate by ethnicity)
        eth_rates = df_mode.groupby("ethnicity_signal")["approved"].mean()
        if len(eth_rates) >= 2:
            dp_ratio = eth_rates.min() / eth_rates.max() if eth_rates.max() > 0 else 0
            dp_diff = eth_rates.max() - eth_rates.min()
        else:
            dp_ratio, dp_diff = 1, 0

        # 2. Equalized Odds approximation (per credit tier)
        df_mode["credit_tier"] = pd.cut(
            df_mode["credit_score"],
            bins=[0, 650, 720, 850],
            labels=["Low", "Medium", "High"],
        )

        eo_diffs = []
        for tier in df_mode["credit_tier"].dropna().unique():
            tier_data = df_mode[df_mode["credit_tier"] == tier]
            tier_rates = tier_data.groupby("ethnicity_signal")["approved"].mean()
            if len(tier_rates) >= 2:
                eo_diffs.append(tier_rates.max() - tier_rates.min())
        eo_avg_diff = np.mean(eo_diffs) if eo_diffs else 0

        fairness_data.append(
            {"mode": mode, "metric": "Demographic Parity Ratio", "value": dp_ratio}
        )
        fairness_data.append(
            {"mode": mode, "metric": "Demographic Parity Gap", "value": dp_diff}
        )
        fairness_data.append(
            {"mode": mode, "metric": "Equalized Odds Gap", "value": eo_avg_diff}
        )

    if not fairness_data:
        print("No fairness data available")
        return

    df_fair = pd.DataFrame(fairness_data)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        "Demographic Parity Ratio",
        "Demographic Parity Gap",
        "Equalized Odds Gap",
    ]

    for ax, metric in zip(axes, metrics):
        data = df_fair[df_fair["metric"] == metric]

        colors = ["#3498db", "#e74c3c", "#2ecc71"]
        bars = ax.bar(data["mode"], data["value"], color=colors)

        if "Ratio" in metric:
            ax.axhline(
                y=0.8, color="green", linestyle="--", label="Fair threshold (0.8)"
            )
            ax.set_ylim(0, 1.1)
        elif "Gap" in metric:
            ax.axhline(y=0.1, color="red", linestyle="--", label="Fair threshold (0.1)")
            ax.set_ylim(0, max(data["value"].max() * 1.2, 0.2))

        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(loc="upper right")

        for bar, val in zip(bars, data["value"]):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
            )

    plt.suptitle(
        "Fairness Metrics Comparison Across Modes", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "fairness_metrics_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "fairness_metrics_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fairness_metrics_comparison.png")


def plot_sample_size_comparison(df: pd.DataFrame, output_dir: Path):
    """Visualize sample sizes across different categories."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. By experiment type
    ax = axes[0, 0]
    counts = df["experiment_type"].value_counts()
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(counts.index, counts.values, color=colors[: len(counts)])
    ax.set_ylabel("Sample Count")
    ax.set_title("Sample Size by Experiment Type")
    for bar, count in zip(bars, counts.values):
        ax.annotate(
            f"{count:,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )

    # 2. By agent
    ax = axes[0, 1]
    counts = df["agent"].value_counts().head(10)
    ax.barh(range(len(counts)), counts.values, color="steelblue")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.set_xlabel("Sample Count")
    ax.set_title("Sample Size by Agent (Top 10)")

    # 3. By ethnicity
    ax = axes[1, 0]
    df_eth = df[df["ethnicity_signal"] != "Unknown"]
    counts = df_eth["ethnicity_signal"].value_counts()
    counts.index = [e.replace("_Signal", "") for e in counts.index]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors[: len(counts)])
    ax.set_title("Sample Distribution by Ethnicity")

    # 4. By credit score
    ax = axes[1, 1]
    df["credit_tier"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 620, 680, 720, 780, 850],
        labels=["<580", "580-620", "620-680", "680-720", "720-780", "780+"],
    )
    counts = df["credit_tier"].value_counts().sort_index()
    ax.bar(
        range(len(counts)),
        counts.values,
        color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(counts))),
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45)
    ax.set_ylabel("Sample Count")
    ax.set_title("Sample Size by Credit Score Tier")

    plt.tight_layout()
    plt.savefig(
        output_dir / "sample_size_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "sample_size_distribution.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: sample_size_distribution.png")


def plot_parallel_vs_sequential_detail(df: pd.DataFrame, output_dir: Path):
    """Detailed comparison between parallel and sequential modes."""
    df_seq = df[df["experiment_type"] == "sequential"]
    df_par = df[df["experiment_type"] == "parallel"]

    if df_seq.empty or df_par.empty:
        print("Missing sequential or parallel data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Approval rate by agent - sequential vs parallel
    ax = axes[0, 0]

    seq_rates = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
    seq_rates["approved"] = (seq_rates["approval_decision"] == "approve").astype(int)
    seq_by_agent = seq_rates.groupby("agent")["approved"].mean()

    par_rates = df_par[df_par["approval_decision"].isin(["approve", "deny"])].copy()
    par_rates["approved"] = (par_rates["approval_decision"] == "approve").astype(int)
    par_by_agent = par_rates.groupby("agent")["approved"].mean()

    agents = list(set(seq_by_agent.index) & set(par_by_agent.index))
    agents = [a for a in agents if a != "Business Decision"]

    x = np.arange(len(agents))
    width = 0.35

    ax.bar(
        x - width / 2,
        [seq_by_agent.get(a, 0) for a in agents],
        width,
        label="Sequential",
        color="#e74c3c",
    )
    ax.bar(
        x + width / 2,
        [par_by_agent.get(a, 0) for a in agents],
        width,
        label="Parallel",
        color="#2ecc71",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rate by Agent: Sequential vs Parallel")
    ax.legend()
    ax.set_ylim(0, 1)

    # 2. Interest rate by agent
    ax = axes[0, 1]

    seq_ir = (
        df_seq[df_seq["approval_decision"] == "approve"]
        .groupby("agent")["interest_rate"]
        .mean()
    )
    par_ir = (
        df_par[df_par["approval_decision"] == "approve"]
        .groupby("agent")["interest_rate"]
        .mean()
    )

    ax.bar(
        x - width / 2,
        [seq_ir.get(a, 0) for a in agents],
        width,
        label="Sequential",
        color="#e74c3c",
    )
    ax.bar(
        x + width / 2,
        [par_ir.get(a, 0) for a in agents],
        width,
        label="Parallel",
        color="#2ecc71",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.set_ylabel("Average Interest Rate (%)")
    ax.set_title("Interest Rate by Agent: Sequential vs Parallel")
    ax.legend()

    # 3. Scatter plot: Sequential vs Parallel approval rates by demographic group
    ax = axes[1, 0]

    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)
    df_valid = df_valid[df_valid["ethnicity_signal"] != "Unknown"]

    # Calculate rates by ethnicity and mode
    rates = (
        df_valid.groupby(["experiment_type", "ethnicity_signal"])["approved"]
        .mean()
        .unstack(level=0)
    )

    if "sequential" in rates.columns and "parallel" in rates.columns:
        ethnicities = rates.index
        colors = plt.cm.Set1(np.linspace(0, 1, len(ethnicities)))

        for eth, color in zip(ethnicities, colors):
            ax.scatter(
                rates.loc[eth, "sequential"],
                rates.loc[eth, "parallel"],
                label=eth.replace("_Signal", ""),
                color=color,
                s=100,
            )

        # Add diagonal line (perfect agreement)
        lims = [0, 1]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Equal rates")

        ax.set_xlabel("Sequential Approval Rate")
        ax.set_ylabel("Parallel Approval Rate")
        ax.set_title("Approval Rates: Sequential vs Parallel by Ethnicity")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # 4. Confidence comparison
    ax = axes[1, 1]

    seq_conf = seq_rates["confidence_probability"].dropna()
    par_conf = par_rates["confidence_probability"].dropna()

    ax.hist(seq_conf, bins=20, alpha=0.5, label="Sequential", color="#e74c3c")
    ax.hist(par_conf, bins=20, alpha=0.5, label="Parallel", color="#2ecc71")
    ax.axvline(
        x=seq_conf.mean(),
        color="#e74c3c",
        linestyle="--",
        label=f"Seq mean: {seq_conf.mean():.1f}",
    )
    ax.axvline(
        x=par_conf.mean(),
        color="#2ecc71",
        linestyle="--",
        label=f"Par mean: {par_conf.mean():.1f}",
    )

    ax.set_xlabel("Confidence Probability")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Probability Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        output_dir / "parallel_vs_sequential_detail.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "parallel_vs_sequential_detail.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: parallel_vs_sequential_detail.png")


def main():
    """Run all mode comparison plots."""
    output_dir = Path("plotting/outputs/mode_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, _ = load_data("70bresults")
    print(f"Loaded {len(df)} records")

    print("\nGenerating mode comparison plots...")
    plot_mode_comparison_overview(df, output_dir)
    plot_statistical_comparison(df, output_dir)
    plot_fairness_metrics_comparison(df, output_dir)
    plot_sample_size_comparison(df, output_dir)
    plot_parallel_vs_sequential_detail(df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
