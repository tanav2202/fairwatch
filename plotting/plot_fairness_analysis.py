"""
Fairness-Specific Analysis Plots
Deep dive into fairness metrics and bias mitigation analysis.
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


def calculate_fairness_metrics(
    df: pd.DataFrame, protected_attr: str = "ethnicity_signal"
) -> dict:
    """Calculate comprehensive fairness metrics."""
    df_valid = df[
        (df["approval_decision"].isin(["approve", "deny"]))
        & (df[protected_attr] != "Unknown")
    ].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    group_rates = df_valid.groupby(protected_attr)["approved"].mean()

    if len(group_rates) < 2:
        return {}

    # Demographic Parity
    dp_ratio = group_rates.min() / group_rates.max() if group_rates.max() > 0 else 1
    dp_diff = group_rates.max() - group_rates.min()

    # Statistical Parity Difference
    spd = group_rates.max() - group_rates.min()

    # Disparate Impact
    disparate_impact = (
        group_rates.min() / group_rates.max() if group_rates.max() > 0 else 1
    )

    # Four-Fifths Rule Check
    four_fifths_pass = disparate_impact >= 0.8

    return {
        "demographic_parity_ratio": dp_ratio,
        "demographic_parity_diff": dp_diff,
        "statistical_parity_diff": spd,
        "disparate_impact": disparate_impact,
        "four_fifths_pass": four_fifths_pass,
        "group_rates": group_rates.to_dict(),
        "min_rate_group": group_rates.idxmin(),
        "max_rate_group": group_rates.idxmax(),
    }


def plot_comprehensive_fairness_dashboard(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive fairness metrics dashboard."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    modes = ["baseline", "sequential", "parallel"]
    colors = {"baseline": "#3498db", "sequential": "#e74c3c", "parallel": "#27ae60"}

    # Calculate metrics for each mode
    metrics_by_mode = {}
    for mode in modes:
        df_mode = df[df["experiment_type"] == mode]
        metrics_by_mode[mode] = calculate_fairness_metrics(df_mode, "ethnicity_signal")

    # 1. Four-Fifths Rule Compliance
    ax = fig.add_subplot(gs[0, 0])
    disparate_impacts = [metrics_by_mode[m].get("disparate_impact", 0) for m in modes]

    bars = ax.bar(
        modes,
        disparate_impacts,
        color=[colors[m] for m in modes],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(
        y=0.8,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Four-Fifths Rule (0.8)",
    )
    ax.axhline(
        y=1.0, color="blue", linestyle=":", linewidth=1, label="Perfect Parity (1.0)"
    )
    ax.set_ylabel("Disparate Impact Ratio")
    ax.set_title("Four-Fifths Rule Compliance")
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper right", fontsize=9)

    for bar, val in zip(bars, disparate_impacts):
        color = "green" if val >= 0.8 else "red"
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontweight="bold",
            color=color,
        )

    # 2. Statistical Parity Difference
    ax = fig.add_subplot(gs[0, 1])
    spd_vals = [metrics_by_mode[m].get("statistical_parity_diff", 0) for m in modes]

    bars = ax.bar(
        modes,
        spd_vals,
        color=[colors[m] for m in modes],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(
        y=0.1,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Acceptable threshold (0.1)",
    )
    ax.set_ylabel("Statistical Parity Difference")
    ax.set_title("Statistical Parity Difference\n(Lower is Better)")
    ax.legend(loc="upper right", fontsize=9)

    for bar, val in zip(bars, spd_vals):
        color = "green" if val <= 0.1 else "red"
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontweight="bold",
            color=color,
        )

    # 3. Group-wise approval rates
    ax = fig.add_subplot(gs[0, 2])

    ethnicities = ["White_Signal", "Black_Signal", "Hispanic_Signal", "Asian_Signal"]
    eth_labels = [e.replace("_Signal", "") for e in ethnicities]

    x = np.arange(len(eth_labels))
    width = 0.25

    for i, mode in enumerate(modes):
        rates = [
            metrics_by_mode[mode].get("group_rates", {}).get(e, 0) for e in ethnicities
        ]
        ax.bar(
            x + i * width,
            rates,
            width,
            label=mode.title(),
            color=colors[mode],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(eth_labels)
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rate by Ethnicity")
    ax.legend()
    ax.set_ylim(0, 1)

    # 4. Most/Least Favored Groups
    ax = fig.add_subplot(gs[1, 0])

    groups_data = []
    for mode in modes:
        min_group = metrics_by_mode[mode].get("min_rate_group", "")
        max_group = metrics_by_mode[mode].get("max_rate_group", "")
        groups_data.append(
            {
                "Mode": mode.title(),
                "Most Favored": (
                    max_group.replace("_Signal", "") if max_group else "N/A"
                ),
                "Least Favored": (
                    min_group.replace("_Signal", "") if min_group else "N/A"
                ),
            }
        )

    df_groups = pd.DataFrame(groups_data)
    ax.axis("off")

    table = ax.table(
        cellText=df_groups.values,
        colLabels=df_groups.columns,
        cellLoc="center",
        loc="center",
        colColours=["#3498db"] * 3,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for i in range(3):
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Most/Least Favored Groups by Mode", y=0.95)

    # 5. Fairness by Credit Tier
    ax = fig.add_subplot(gs[1, 1])

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

    fairness_by_tier = []
    for tier in ["Low", "Medium", "High"]:
        df_tier = df_valid[df_valid["credit_tier"] == tier]
        for mode in modes:
            df_mode = df_tier[df_tier["experiment_type"] == mode]
            metrics = calculate_fairness_metrics(df_mode, "ethnicity_signal")
            fairness_by_tier.append(
                {
                    "Credit Tier": tier,
                    "Mode": mode,
                    "DP Ratio": metrics.get("disparate_impact", 0),
                }
            )

    df_tier_fair = pd.DataFrame(fairness_by_tier)
    pivot = df_tier_fair.pivot(index="Credit Tier", columns="Mode", values="DP Ratio")
    pivot = pivot.reindex(["Low", "Medium", "High"])[modes]

    pivot.plot(
        kind="bar",
        ax=ax,
        color=[colors[m] for m in modes],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(
        y=0.8, color="green", linestyle="--", linewidth=2, label="Fair threshold"
    )
    ax.set_ylabel("Demographic Parity Ratio")
    ax.set_title("Fairness by Credit Score Tier")
    ax.legend(title="Mode")
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, 1.2)

    # 6. Fairness by Age Group
    ax = fig.add_subplot(gs[1, 2])

    df_valid["age_group"] = pd.cut(
        df_valid["age"], bins=[0, 35, 55, 100], labels=["Young", "Middle", "Senior"]
    )

    fairness_by_age = []
    for age in ["Young", "Middle", "Senior"]:
        df_age = df_valid[df_valid["age_group"] == age]
        for mode in modes:
            df_mode = df_age[df_age["experiment_type"] == mode]
            metrics = calculate_fairness_metrics(df_mode, "ethnicity_signal")
            fairness_by_age.append(
                {
                    "Age Group": age,
                    "Mode": mode,
                    "DP Ratio": metrics.get("disparate_impact", 0),
                }
            )

    df_age_fair = pd.DataFrame(fairness_by_age)
    pivot_age = df_age_fair.pivot(index="Age Group", columns="Mode", values="DP Ratio")
    pivot_age = pivot_age.reindex(["Young", "Middle", "Senior"])[modes]

    pivot_age.plot(
        kind="bar",
        ax=ax,
        color=[colors[m] for m in modes],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(y=0.8, color="green", linestyle="--", linewidth=2)
    ax.set_ylabel("Demographic Parity Ratio")
    ax.set_title("Fairness by Age Group")
    ax.legend(title="Mode")
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, 1.2)

    # 7. Interest Rate Equity
    ax = fig.add_subplot(gs[2, 0])

    df_approved = df_valid[df_valid["approval_decision"] == "approve"]

    ir_equity = []
    for mode in modes:
        df_mode = df_approved[df_approved["experiment_type"] == mode]
        ir_by_eth = df_mode.groupby("ethnicity_signal")["interest_rate"].mean()
        if len(ir_by_eth) >= 2:
            ir_diff = ir_by_eth.max() - ir_by_eth.min()
            ir_equity.append({"Mode": mode, "IR Spread": ir_diff})

    if ir_equity:
        df_ir = pd.DataFrame(ir_equity)
        bars = ax.bar(
            df_ir["Mode"],
            df_ir["IR Spread"],
            color=[colors[m] for m in df_ir["Mode"]],
            edgecolor="black",
        )
        ax.set_ylabel("Interest Rate Spread (%)")
        ax.set_title("Interest Rate Equity\n(Lower Spread = More Fair)")

        for bar, val in zip(bars, df_ir["IR Spread"]):
            ax.annotate(
                f"{val:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
            )

    # 8. Confidence Equity
    ax = fig.add_subplot(gs[2, 1])

    conf_equity = []
    for mode in modes:
        df_mode = df_valid[df_valid["experiment_type"] == mode]
        high_conf_by_eth = df_mode.groupby("ethnicity_signal").apply(
            lambda x: (x["confidence_level"] == "high").mean()
        )
        if len(high_conf_by_eth) >= 2:
            conf_diff = high_conf_by_eth.max() - high_conf_by_eth.min()
            conf_equity.append({"Mode": mode, "High Conf Spread": conf_diff})

    if conf_equity:
        df_conf = pd.DataFrame(conf_equity)
        bars = ax.bar(
            df_conf["Mode"],
            df_conf["High Conf Spread"],
            color=[colors[m] for m in df_conf["Mode"]],
            edgecolor="black",
        )
        ax.set_ylabel("High Confidence Rate Spread")
        ax.set_title("Confidence Equity by Ethnicity\n(Lower Spread = More Fair)")

    # 9. Summary radar chart (simplified as bar)
    ax = fig.add_subplot(gs[2, 2])

    summary_metrics = ["DP Ratio", "Fair IR", "Fair Conf", "Four-Fifths"]

    summary_data = []
    for mode in modes:
        dp = metrics_by_mode[mode].get("disparate_impact", 0)
        ir_spread = [x["IR Spread"] for x in ir_equity if x["Mode"] == mode]
        ir_fair = 1 - (ir_spread[0] / 5 if ir_spread else 0)  # Normalize
        conf_spread = [x["High Conf Spread"] for x in conf_equity if x["Mode"] == mode]
        conf_fair = 1 - (conf_spread[0] if conf_spread else 0)
        four_fifth = 1 if dp >= 0.8 else dp / 0.8

        summary_data.append(
            {
                "Mode": mode,
                "DP Ratio": dp,
                "Fair IR": ir_fair,
                "Fair Conf": conf_fair,
                "Four-Fifths": four_fifth,
            }
        )

    df_summary = pd.DataFrame(summary_data)

    x = np.arange(len(summary_metrics))
    width = 0.25

    for i, mode in enumerate(modes):
        vals = df_summary[df_summary["Mode"] == mode][summary_metrics].values[0]
        ax.bar(
            x + i * width,
            vals,
            width,
            label=mode.title(),
            color=colors[mode],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(summary_metrics, rotation=45, ha="right")
    ax.set_ylabel("Fairness Score")
    ax.set_title("Comprehensive Fairness Summary\n(Higher = More Fair)")
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5)

    plt.suptitle(
        "Multi-Agent System Fairness Dashboard", fontsize=14, fontweight="bold", y=1.02
    )

    plt.savefig(output_dir / "fairness_dashboard.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fairness_dashboard.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fairness_dashboard.png")


def plot_fairness_over_orderings(df: pd.DataFrame, output_dir: Path):
    """Analyze fairness metrics across different sequential orderings."""
    df_seq = df[df["experiment_type"] == "sequential"].copy()

    if df_seq.empty:
        print("No sequential data for ordering fairness analysis")
        return

    df_valid = df_seq[
        (df_seq["approval_decision"].isin(["approve", "deny"]))
        & (df_seq["ethnicity_signal"] != "Unknown")
    ].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    # Get final decisions for each chain
    last_pos = (
        df_valid.groupby(["chain_id", "agent_order"])["agent_position"]
        .max()
        .reset_index()
    )
    last_pos.columns = ["chain_id", "agent_order", "max_pos"]

    df_final = df_valid.merge(last_pos, on=["chain_id", "agent_order"])
    df_final = df_final[df_final["agent_position"] == df_final["max_pos"]]

    # Calculate fairness for each ordering
    ordering_fairness = []

    for order in df_final["agent_order"].unique():
        df_order = df_final[df_final["agent_order"] == order]
        metrics = calculate_fairness_metrics(df_order, "ethnicity_signal")

        if metrics:
            ordering_fairness.append(
                {
                    "ordering": order,
                    "dp_ratio": metrics.get("disparate_impact", 0),
                    "spd": metrics.get("statistical_parity_diff", 0),
                    "n_samples": len(df_order),
                    "first_agent": order.split("_")[0] if order else "Unknown",
                }
            )

    if not ordering_fairness:
        print("Could not calculate ordering fairness")
        return

    df_order_fair = pd.DataFrame(ordering_fairness)
    df_order_fair = df_order_fair.sort_values("dp_ratio", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. DP Ratio by Ordering (top orderings)
    ax = axes[0, 0]
    top_orders = df_order_fair.head(15)

    colors = plt.cm.RdYlGn(top_orders["dp_ratio"] / top_orders["dp_ratio"].max())
    bars = ax.barh(range(len(top_orders)), top_orders["dp_ratio"], color=colors)
    ax.set_yticks(range(len(top_orders)))
    ax.set_yticklabels(
        [o.replace("_", "→") for o in top_orders["ordering"]], fontsize=8
    )
    ax.axvline(x=0.8, color="green", linestyle="--", label="Four-Fifths Rule")
    ax.set_xlabel("Demographic Parity Ratio")
    ax.set_title("Most Fair Orderings (DP Ratio)")
    ax.legend()

    # 2. Least fair orderings
    ax = axes[0, 1]
    bottom_orders = df_order_fair.tail(15)

    colors = plt.cm.RdYlGn(bottom_orders["dp_ratio"] / df_order_fair["dp_ratio"].max())
    bars = ax.barh(range(len(bottom_orders)), bottom_orders["dp_ratio"], color=colors)
    ax.set_yticks(range(len(bottom_orders)))
    ax.set_yticklabels(
        [o.replace("_", "→") for o in bottom_orders["ordering"]], fontsize=8
    )
    ax.axvline(x=0.8, color="green", linestyle="--")
    ax.set_xlabel("Demographic Parity Ratio")
    ax.set_title("Least Fair Orderings (DP Ratio)")

    # 3. First agent impact on fairness
    ax = axes[1, 0]
    first_agent_fair = df_order_fair.groupby("first_agent")["dp_ratio"].agg(
        ["mean", "std", "count"]
    )
    first_agent_fair = first_agent_fair[first_agent_fair["count"] >= 2].sort_values(
        "mean", ascending=False
    )

    if len(first_agent_fair) > 0:
        colors = plt.cm.RdYlGn(
            first_agent_fair["mean"] / first_agent_fair["mean"].max()
        )
        bars = ax.bar(
            range(len(first_agent_fair)),
            first_agent_fair["mean"],
            yerr=first_agent_fair["std"],
            color=colors,
            capsize=4,
        )
        ax.set_xticks(range(len(first_agent_fair)))
        ax.set_xticklabels(first_agent_fair.index, rotation=45, ha="right")
        ax.axhline(y=0.8, color="green", linestyle="--")
        ax.set_ylabel("Average DP Ratio")
        ax.set_title("Fairness by First Agent in Chain")
        ax.set_ylim(0, 1.1)

    # 4. Fairness distribution
    ax = axes[1, 1]
    ax.hist(
        df_order_fair["dp_ratio"],
        bins=20,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    ax.axvline(
        x=0.8,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Four-Fifths Rule (0.8)",
    )
    ax.axvline(
        x=df_order_fair["dp_ratio"].mean(),
        color="red",
        linestyle="-",
        linewidth=2,
        label=f'Mean ({df_order_fair["dp_ratio"].mean():.3f})',
    )
    ax.axvline(
        x=df_order_fair["dp_ratio"].median(),
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f'Median ({df_order_fair["dp_ratio"].median():.3f})',
    )
    ax.set_xlabel("Demographic Parity Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Fairness Across Orderings")
    ax.legend()

    # Add stats annotation
    n_fair = (df_order_fair["dp_ratio"] >= 0.8).sum()
    n_total = len(df_order_fair)
    ax.annotate(
        f"{n_fair}/{n_total} orderings meet\nFour-Fifths Rule",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fairness_by_ordering.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fairness_by_ordering.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fairness_by_ordering.png")

    # Save ordering rankings to CSV
    df_order_fair.to_csv(output_dir / "ordering_fairness_rankings.csv", index=False)
    print("Saved: ordering_fairness_rankings.csv")


def plot_protected_attribute_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze multiple protected attributes together."""
    df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Define protected attributes
    protected_attrs = {
        "ethnicity_signal": ("Ethnicity", lambda x: x != "Unknown"),
        "age_group": ("Age", lambda x: pd.notna(x)),
        "visa_status": ("Visa Status", lambda x: pd.notna(x)),
    }

    # Create age groups
    df_valid["age_group"] = pd.cut(
        df_valid["age"], bins=[0, 35, 55, 100], labels=["Young", "Middle", "Senior"]
    )

    modes = ["baseline", "sequential", "parallel"]
    colors = ["#3498db", "#e74c3c", "#27ae60"]

    # 1-3. Fairness metrics for each protected attribute
    metrics_data = []

    for attr, (label, filter_fn) in protected_attrs.items():
        df_attr = df_valid[filter_fn(df_valid[attr])]

        for mode in modes:
            df_mode = df_attr[df_attr["experiment_type"] == mode]
            metrics = calculate_fairness_metrics(df_mode, attr)

            if metrics:
                metrics_data.append(
                    {
                        "Attribute": label,
                        "Mode": mode.title(),
                        "DP Ratio": metrics.get("disparate_impact", 0),
                        "SPD": metrics.get("statistical_parity_diff", 0),
                    }
                )

    df_metrics = pd.DataFrame(metrics_data)

    # DP Ratio comparison
    ax = axes[0, 0]
    pivot_dp = df_metrics.pivot(index="Attribute", columns="Mode", values="DP Ratio")
    pivot_dp.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(
        y=0.8, color="green", linestyle="--", linewidth=2, label="Fair threshold"
    )
    ax.set_ylabel("Demographic Parity Ratio")
    ax.set_title("Fairness Across Protected Attributes")
    ax.legend(title="Mode")
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, 1.2)

    # SPD comparison
    ax = axes[0, 1]
    pivot_spd = df_metrics.pivot(index="Attribute", columns="Mode", values="SPD")
    pivot_spd.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(
        y=0.1, color="red", linestyle="--", linewidth=2, label="Acceptable threshold"
    )
    ax.set_ylabel("Statistical Parity Difference")
    ax.set_title("Bias Magnitude Across Attributes")
    ax.legend(title="Mode")
    ax.tick_params(axis="x", rotation=0)

    # 4. Compound fairness (ethnicity + age)
    ax = axes[1, 0]

    df_compound = df_valid[(df_valid["ethnicity_signal"] != "Unknown")]
    df_compound["eth_age"] = (
        df_compound["ethnicity_signal"].str.replace("_Signal", "")
        + "_"
        + df_compound["age_group"].astype(str)
    )

    compound_rates = (
        df_compound.groupby(["experiment_type", "eth_age"])["approved"]
        .mean()
        .unstack(level=0)
    )

    if not compound_rates.empty:
        compound_var = compound_rates.var()
        bars = ax.bar(
            compound_var.index, compound_var.values, color=colors, edgecolor="black"
        )
        ax.set_ylabel("Variance in Approval Rates")
        ax.set_title("Intersectional Fairness\n(Lower Variance = More Fair)")

        for bar, val in zip(bars, compound_var.values):
            ax.annotate(
                f"{val:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # 5. Summary table
    ax = axes[1, 1]
    ax.axis("off")

    # Create summary
    summary = []
    for mode in modes:
        df_mode = df_valid[df_valid["experiment_type"] == mode]

        eth_metrics = calculate_fairness_metrics(
            df_mode[df_mode["ethnicity_signal"] != "Unknown"], "ethnicity_signal"
        )
        age_metrics = calculate_fairness_metrics(df_mode, "age_group")

        summary.append(
            {
                "Mode": mode.title(),
                "Eth DP": f"{eth_metrics.get('disparate_impact', 0):.3f}",
                "Age DP": f"{age_metrics.get('disparate_impact', 0):.3f}",
                "Eth Fair": (
                    "✓" if eth_metrics.get("disparate_impact", 0) >= 0.8 else "✗"
                ),
                "Age Fair": (
                    "✓" if age_metrics.get("disparate_impact", 0) >= 0.8 else "✗"
                ),
            }
        )

    df_summary = pd.DataFrame(summary)

    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        cellLoc="center",
        loc="center",
        colColours=["#3498db"] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.5)

    for i in range(5):
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Fairness Summary (✓ = Meets Four-Fifths Rule)", y=0.95)

    plt.tight_layout()
    plt.savefig(
        output_dir / "protected_attribute_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "protected_attribute_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: protected_attribute_analysis.png")


def main():
    """Run all fairness analysis plots."""
    output_dir = Path("plotting/outputs/fairness")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, _ = load_data("70bresults")
    print(f"Loaded {len(df)} records")

    print("\nGenerating fairness analysis plots...")
    plot_comprehensive_fairness_dashboard(df, output_dir)
    plot_fairness_over_orderings(df, output_dir)
    plot_protected_attribute_analysis(df, output_dir)

    print(f"\nAll fairness plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
