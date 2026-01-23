"""
Paper-Ready Publication Figures
Creates high-quality figures suitable for academic publication.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch
import sys

sys.path.append(str(Path(__file__).parent))
from data_loader import load_data, FairWatchDataLoader

# Publication-quality settings
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def create_main_results_figure(df: pd.DataFrame, output_dir: Path):
    """Create the main results figure for the paper."""
    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    modes = ["baseline", "sequential", "parallel"]
    mode_labels = ["Baseline", "Sequential", "Parallel"]
    colors = ["#2C3E50", "#E74C3C", "#27AE60"]

    # (a) Overall approval rates
    ax1 = fig.add_subplot(gs[0, 0])
    stats = df_valid.groupby("experiment_type")["approved"].agg(
        ["mean", "std", "count"]
    )
    rates = [stats.loc[m, "mean"] if m in stats.index else 0 for m in modes]
    errors = [
        stats.loc[m, "std"] / np.sqrt(stats.loc[m, "count"]) if m in stats.index else 0
        for m in modes
    ]

    bars = ax1.bar(
        mode_labels,
        rates,
        color=colors,
        yerr=errors,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("Approval Rate")
    ax1.set_ylim(0, 1)
    ax1.set_title("(a) Overall Approval Rate")

    for bar, rate in zip(bars, rates):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.1%}",
            ha="center",
            fontsize=9,
        )

    # (b) Interest rates
    ax2 = fig.add_subplot(gs[0, 1])
    df_approved = df_valid[df_valid["approval_decision"] == "approve"]

    ir_data = [
        df_approved[df_approved["experiment_type"] == m]["interest_rate"].dropna()
        for m in modes
    ]

    bp = ax2.boxplot(ir_data, labels=mode_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Interest Rate (%)")
    ax2.set_title("(b) Interest Rate Distribution")

    # (c) Confidence levels
    ax3 = fig.add_subplot(gs[0, 2])
    conf_data = []
    for mode, label in zip(modes, mode_labels):
        df_mode = df_valid[df_valid["experiment_type"] == mode]
        total = len(df_mode)
        for level in ["high", "medium", "low"]:
            conf_data.append(
                {
                    "Mode": label,
                    "Level": level.title(),
                    "Proportion": (
                        (df_mode["confidence_level"] == level).sum() / total
                        if total > 0
                        else 0
                    ),
                }
            )

    df_conf = pd.DataFrame(conf_data)
    pivot = df_conf.pivot(index="Mode", columns="Level", values="Proportion")
    pivot = pivot.reindex(mode_labels)[["High", "Medium", "Low"]]
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax3,
        color=["#27AE60", "#F39C12", "#E74C3C"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.set_ylabel("Proportion")
    ax3.set_title("(c) Confidence Distribution")
    ax3.legend(title="Level", loc="upper right")
    ax3.tick_params(axis="x", rotation=0)
    ax3.set_ylim(0, 1)

    # (d) Ethnicity bias
    ax4 = fig.add_subplot(gs[1, 0])
    df_eth = df_valid[df_valid["ethnicity_signal"] != "Unknown"]

    eth_map = {
        "White_Signal": "White",
        "Black_Signal": "Black",
        "Hispanic_Signal": "Hispanic",
        "Asian_Signal": "Asian",
    }
    df_eth["ethnicity"] = df_eth["ethnicity_signal"].map(eth_map)

    pivot_eth = df_eth.pivot_table(
        values="approved", index="experiment_type", columns="ethnicity", aggfunc="mean"
    )
    pivot_eth = pivot_eth.reindex(modes)
    pivot_eth.index = mode_labels

    x = np.arange(len(mode_labels))
    width = 0.2
    eth_colors = ["#3498DB", "#E67E22", "#9B59B6", "#1ABC9C"]

    for i, (eth, color) in enumerate(zip(pivot_eth.columns, eth_colors)):
        ax4.bar(
            x + i * width,
            pivot_eth[eth],
            width,
            label=eth,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(mode_labels)
    ax4.set_ylabel("Approval Rate")
    ax4.set_title("(d) Approval by Ethnicity")
    ax4.legend(title="Ethnicity", ncol=2, loc="upper right")
    ax4.set_ylim(0, 1)

    # (e) Credit score impact
    ax5 = fig.add_subplot(gs[1, 1])
    df_valid["credit_tier"] = pd.cut(
        df_valid["credit_score"],
        bins=[0, 650, 720, 850],
        labels=["Low\n(<650)", "Med\n(650-720)", "High\n(720+)"],
    )

    pivot_credit = df_valid.pivot_table(
        values="approved",
        index="experiment_type",
        columns="credit_tier",
        aggfunc="mean",
    )
    pivot_credit = pivot_credit.reindex(modes)
    pivot_credit.index = mode_labels

    tier_colors = ["#E74C3C", "#F39C12", "#27AE60"]
    for i, (tier, color) in enumerate(zip(pivot_credit.columns, tier_colors)):
        ax5.bar(
            x + i * width,
            pivot_credit[tier],
            width,
            label=tier,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax5.set_xticks(x + width)
    ax5.set_xticklabels(mode_labels)
    ax5.set_ylabel("Approval Rate")
    ax5.set_title("(e) Approval by Credit Score")
    ax5.legend(title="Credit Tier", loc="upper right")
    ax5.set_ylim(0, 1)

    # (f) Fairness metrics
    ax6 = fig.add_subplot(gs[1, 2])

    fairness_metrics = []
    for mode, label in zip(modes, mode_labels):
        df_mode = df_eth[df_eth["experiment_type"] == mode]
        eth_rates = df_mode.groupby("ethnicity")["approved"].mean()
        if len(eth_rates) >= 2:
            dp_ratio = eth_rates.min() / eth_rates.max() if eth_rates.max() > 0 else 1
            fairness_metrics.append({"Mode": label, "DP Ratio": dp_ratio})

    df_fair = pd.DataFrame(fairness_metrics)
    bars = ax6.bar(
        df_fair["Mode"],
        df_fair["DP Ratio"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax6.axhline(
        y=0.8,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label="Fair threshold (0.8)",
    )
    ax6.set_ylabel("Demographic Parity Ratio")
    ax6.set_title("(f) Fairness: DP Ratio")
    ax6.set_ylim(0, 1.1)
    ax6.legend(loc="lower right")

    for bar, val in zip(bars, df_fair["DP Ratio"]):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            fontsize=9,
        )

    plt.suptitle(
        "Multi-Agent System Performance Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.savefig(output_dir / "figure_main_results.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure_main_results.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: figure_main_results.png/pdf")


def create_sequential_ordering_figure(df: pd.DataFrame, output_dir: Path):
    """Create figure showing impact of agent ordering."""
    df_seq = df[(df["experiment_type"] == "sequential")].copy()

    if df_seq.empty:
        print("No sequential data for ordering figure")
        return

    df_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Decision drift through chain
    ax = axes[0]
    stats = df_valid.groupby("agent_position").agg(
        {"approved": ["mean", "std", "count"]}
    )
    stats.columns = ["rate", "std", "count"]
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats = stats.reset_index()

    ax.errorbar(
        stats["agent_position"],
        stats["rate"],
        yerr=stats["se"],
        marker="o",
        markersize=8,
        linewidth=2,
        capsize=5,
        color="#2C3E50",
    )
    ax.fill_between(
        stats["agent_position"],
        stats["rate"] - stats["se"],
        stats["rate"] + stats["se"],
        alpha=0.2,
        color="#2C3E50",
    )

    ax.set_xlabel("Position in Sequential Chain")
    ax.set_ylabel("Approval Rate")
    ax.set_title("(a) Decision Evolution Through Chain")
    ax.set_xticks(stats["agent_position"])
    ax.set_ylim(0, 1)

    # (b) First agent influence
    ax = axes[1]

    # Get first and last decisions
    first = df_valid[df_valid["agent_position"] == 1][
        ["chain_id", "approval_decision", "agent"]
    ].rename(columns={"approval_decision": "first", "agent": "first_agent"})

    last_pos = df_valid.groupby("chain_id")["agent_position"].max().reset_index()
    last_pos.columns = ["chain_id", "max_pos"]
    df_last = df_valid.merge(last_pos, on="chain_id")
    df_last = df_last[df_last["agent_position"] == df_last["max_pos"]][
        ["chain_id", "approval_decision"]
    ].rename(columns={"approval_decision": "final"})

    merged = first.merge(df_last, on="chain_id")
    merged["agree"] = (merged["first"] == merged["final"]).astype(int)

    agreement = (
        merged.groupby("first_agent")["agree"].mean().sort_values(ascending=False)
    )

    colors = plt.cm.RdYlGn(agreement.values)
    bars = ax.barh(
        range(len(agreement)),
        agreement.values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(agreement)))
    ax.set_yticklabels(agreement.index)
    ax.set_xlabel("Agreement with Final Decision")
    ax.set_title("(b) First Agent Influence")
    ax.set_xlim(0, 1)

    for i, (bar, val) in enumerate(zip(bars, agreement.values)):
        ax.text(val + 0.02, i, f"{val:.1%}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        output_dir / "figure_sequential_ordering.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "figure_sequential_ordering.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: figure_sequential_ordering.png/pdf")


def create_agent_agreement_figure(df: pd.DataFrame, output_dir: Path):
    """Create figure showing agent agreement patterns."""
    df_par = df[
        (df["experiment_type"] == "parallel") & (df["agent"] != "Business Decision")
    ]

    if df_par.empty:
        print("No parallel data for agreement figure")
        return

    # Create pivot: prompt_id x agent -> approval_decision
    pivot = df_par.pivot_table(
        index="prompt_id", columns="agent", values="approval_decision", aggfunc="first"
    )

    agents = pivot.columns.tolist()
    n_agents = len(agents)

    # Calculate agreement matrix
    agreement = np.zeros((n_agents, n_agents))
    for i, a1 in enumerate(agents):
        for j, a2 in enumerate(agents):
            if i == j:
                agreement[i, j] = 1.0
            else:
                mask = pivot[[a1, a2]].notna().all(axis=1)
                if mask.sum() > 0:
                    agreement[i, j] = (
                        pivot.loc[mask, a1] == pivot.loc[mask, a2]
                    ).mean()

    fig, ax = plt.subplots(figsize=(8, 7))

    # Custom colormap
    mask = np.triu(np.ones_like(agreement, dtype=bool), k=1)

    sns.heatmap(
        agreement,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        xticklabels=agents,
        yticklabels=agents,
        ax=ax,
        vmin=0.6,
        vmax=1.0,
        cbar_kws={"label": "Agreement Rate"},
        mask=mask,
        square=True,
        linewidths=0.5,
    )

    # Mirror the values
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{agreement[i, j]:.1%}",
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_title(
        "Agent Pairwise Agreement on Loan Decisions", fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "figure_agent_agreement.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure_agent_agreement.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: figure_agent_agreement.png/pdf")


def create_bias_heatmap_figure(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive bias heatmap figure."""
    df_valid = df[
        (df["approval_decision"].isin(["approve", "deny"]))
        & (df["ethnicity_signal"] != "Unknown")
    ].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    df_valid["credit_tier"] = pd.cut(
        df_valid["credit_score"],
        bins=[0, 650, 720, 850],
        labels=["Low", "Medium", "High"],
    )

    eth_map = {
        "White_Signal": "White",
        "Black_Signal": "Black",
        "Hispanic_Signal": "Hispanic",
        "Asian_Signal": "Asian",
    }
    df_valid["ethnicity"] = df_valid["ethnicity_signal"].map(eth_map)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    modes = ["baseline", "sequential", "parallel"]
    titles = ["(a) Baseline", "(b) Sequential", "(c) Parallel"]

    for ax, mode, title in zip(axes, modes, titles):
        df_mode = df_valid[df_valid["experiment_type"] == mode]

        if df_mode.empty:
            ax.set_title(f"{title} - No Data")
            continue

        pivot = df_mode.pivot_table(
            values="approved", index="credit_tier", columns="ethnicity", aggfunc="mean"
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
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Ethnicity")
        ax.set_ylabel("Credit Score Tier")

    plt.suptitle(
        "Intersectional Bias: Approval Rates by Credit Score and Ethnicity",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "figure_bias_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure_bias_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: figure_bias_heatmap.png/pdf")


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create summary statistics table."""
    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    modes = ["baseline", "sequential", "parallel"]

    summary = []

    for mode in modes:
        df_mode = df_valid[df_valid["experiment_type"] == mode]
        df_approved = df_mode[df_mode["approval_decision"] == "approve"]
        df_eth = df_mode[df_mode["ethnicity_signal"] != "Unknown"]

        if df_mode.empty:
            continue

        # Ethnicity disparity
        eth_rates = df_eth.groupby("ethnicity_signal")["approved"].mean()
        dp_ratio = (
            eth_rates.min() / eth_rates.max()
            if len(eth_rates) >= 2 and eth_rates.max() > 0
            else 1
        )

        summary.append(
            {
                "Mode": mode.title(),
                "N Samples": len(df_mode),
                "Approval Rate": f"{df_mode['approved'].mean():.1%}",
                "Avg Interest Rate": f"{df_approved['interest_rate'].mean():.2f}%",
                "Interest Rate Std": f"{df_approved['interest_rate'].std():.2f}%",
                "High Confidence %": f"{(df_mode['confidence_level'] == 'high').mean():.1%}",
                "DP Ratio": f"{dp_ratio:.3f}",
            }
        )

    df_summary = pd.DataFrame(summary)

    # Save as CSV
    df_summary.to_csv(output_dir / "table_summary_stats.csv", index=False)

    # Create styled table figure
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        cellLoc="center",
        loc="center",
        colColours=["#3498db"] * len(df_summary.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    plt.title(
        "Summary Statistics by Evaluation Mode", fontsize=12, fontweight="bold", pad=20
    )

    plt.savefig(output_dir / "table_summary_stats.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "table_summary_stats.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: table_summary_stats.png/pdf/csv")


def main():
    """Generate all publication-ready figures."""
    output_dir = Path("plotting/outputs/publication")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, _ = load_data("70bresults")
    print(f"Loaded {len(df)} records")

    print("\nGenerating publication figures...")
    create_main_results_figure(df, output_dir)
    create_sequential_ordering_figure(df, output_dir)
    create_agent_agreement_figure(df, output_dir)
    create_bias_heatmap_figure(df, output_dir)
    create_summary_table(df, output_dir)

    print(f"\nAll publication figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
