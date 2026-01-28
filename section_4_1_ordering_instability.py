"""
Section 4.1: Ordering Instability Analysis (4 Models)
Comprehensive analysis of ordering effects across model scales:
- Llama 3.2 (small)
- Mistral Latest (small)
- Qwen (medium)
- Llama 70B (large)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Add plotting to path for data loader
sys.path.insert(0, str(Path(__file__).parent / "plotting"))


def load_small_model_data(model_dir: str, model_name: str) -> pd.DataFrame:
    """Load data from outputs_simple directory."""
    print(f"\nLoading {model_name} data from {model_dir}...")
    
    # Agent name mapping (shortened to full)
    agent_map = {
        "consumer": "Consumer_Advocate",
        "data": "Data_Science",
        "regulatory": "Regulatory",
        "risk": "Risk_Manager",
    }
    
    sequential_dir = Path(model_dir) / "sequential"
    records = []
    
    for file in sorted(sequential_dir.glob("sequential_*.json")):
        # Extract agent order from filename
        name = file.stem.replace("sequential_", "")
        parts = name.split("_")
        
        # Map to full names
        order = []
        for part in parts:
            if part in agent_map:
                order.append(agent_map[part])
        
        if len(order) != 4:
            print(f"  Warning: Skipping {file.name} - couldn't parse order")
            continue
        
        order_str = "_".join(order)
        
        try:
            with open(file) as f:
                data = json.load(f)
            
            for idx, result in enumerate(data):
                # Handle different data formats
                if isinstance(result, dict):
                    # New format: {"input": {...}, "all_agent_outputs": [...]}
                    input_data = result.get("input", {})
                    all_outputs = result.get("all_agent_outputs", [])
                    
                    chain_id = f"{order_str}_{idx}"
                    
                    for agent_idx, agent_output in enumerate(all_outputs):
                        if agent_idx >= len(order):
                            break
                        
                        agent_name = order[agent_idx]
                        
                        # Normalize decision
                        decision = agent_output.get("approval_decision", "").lower()
                        if decision not in ["approve", "deny"]:
                            decision = "approve" if agent_output.get("approved", False) else "deny"
                        
                        record = {
                            "chain_id": chain_id,
                            "agent_order": order_str,
                            "agent": agent_name.replace("_", " "),
                            "agent_position": agent_idx + 1,
                            "is_final_agent": agent_idx == len(order) - 1,
                            "approval_decision": decision,
                            "experiment_type": "sequential",
                            "ethnicity_signal": input_data.get("ethnicity_signal"),
                            "credit_score": input_data.get("credit_score"),
                        }
                        
                        records.append(record)
        
        except Exception as e:
            print(f"  Error loading {file.name}: {e}")
            continue
    
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} records from {len(df['agent_order'].unique())} orderings")
    return df


def load_qwen_data() -> pd.DataFrame:
    """Load Qwen validation data."""
    print("\nLoading Qwen data...")
    
    qwen_dir = Path("qwen_validation")
    records = []
    
    agent_map = {
        "consumer": "Consumer_Advocate",
        "data": "Data_Science",
        "regulatory": "Regulatory",
        "risk": "Risk_Manager",
    }
    
    for file in sorted(qwen_dir.glob("sequential_*.json")):
        name = file.stem.replace("sequential_", "")
        parts = name.split("_")
        order = []
        for part in parts:
            if part in agent_map:
                order.append(agent_map[part])
        
        order_str = "_".join(order)
        
        with open(file) as f:
            data = json.load(f)
        
        for result in data.get("results", []):
            input_str = result.get("input", "{}")
            try:
                input_data = json.loads(input_str)
            except:
                input_data = {}
            
            base_record = {
                "chain_id": result.get("chain_id"),
                "agent_order": order_str,
                "ethnicity_signal": input_data.get("ethnicity_signal"),
                "credit_score": input_data.get("credit_score"),
            }
            
            decisions = result.get("decisions", {})
            for i, agent_name in enumerate(order):
                agent_key = agent_name.lower().replace("_", "")
                agent_data = decisions.get(agent_key, {})
                
                if not agent_data:
                    continue
                
                record = base_record.copy()
                record["agent"] = agent_name.replace("_", " ")
                record["agent_position"] = i + 1
                record["is_final_agent"] = i == len(order) - 1
                record["approval_decision"] = agent_data.get("approval_decision")
                record["experiment_type"] = "sequential"
                
                records.append(record)
    
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} records from {len(df['agent_order'].unique())} orderings")
    return df


def load_70b_data() -> pd.DataFrame:
    """Load 70B data."""
    print("\nLoading 70B data...")
    from data_loader import load_data
    
    df, _ = load_data("70bresults")
    df_seq = df[df["experiment_type"] == "sequential"].copy()
    print(f"  Loaded {len(df_seq):,} sequential records")
    return df_seq


def analyze_ordering_instability_comprehensive(models: Dict[str, pd.DataFrame], output_dir: Path):
    """Comprehensive ordering instability analysis for all 4 models."""
    print("\n" + "=" * 80)
    print("SECTION 4.1: ORDERING INSTABILITY - 4 MODEL COMPARISON")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Calculate metrics for each model
    for model_name, df in models.items():
        df_valid = df[df["approval_decision"].isin(["approve", "deny"])].copy()
        df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)
        df_final = df_valid[df_valid["is_final_agent"] == True].copy()
        
        if len(df_final) == 0:
            print(f"\n{model_name}: No valid data")
            continue
        
        # Calculate approval rates per ordering
        ordering_rates = df_final.groupby("agent_order")["approved"].mean().sort_values(ascending=False)
        
        all_results[model_name] = {
            "num_orderings": len(ordering_rates),
            "min_approval": float(ordering_rates.min()),
            "max_approval": float(ordering_rates.max()),
            "range": float(ordering_rates.max() - ordering_rates.min()),
            "mean_approval": float(ordering_rates.mean()),
            "std_approval": float(ordering_rates.std()),
            "ordering_rates": ordering_rates.to_dict(),
            "ordering_rates_sorted": ordering_rates.to_dict(),  # Already sorted
        }
        
        print(f"\n{model_name}:")
        print(f"  Orderings: {all_results[model_name]['num_orderings']}")
        print(f"  Approval Range: {all_results[model_name]['min_approval']:.1%} - {all_results[model_name]['max_approval']:.1%}")
        print(f"  Variance: {all_results[model_name]['range']:.1%} ({all_results[model_name]['range']*100:.2f} pp)")
        print(f"  Mean ± Std: {all_results[model_name]['mean_approval']:.1%} ± {all_results[model_name]['std_approval']:.1%}")
    
    # Figure 1: Ordering sensitivity comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    width = 0.25
    
    ranges = [all_results[m]["range"] * 100 for m in model_names]
    means = [all_results[m]["mean_approval"] * 100 for m in model_names]
    stds = [all_results[m]["std_approval"] * 100 for m in model_names]
    
    bars1 = ax.bar(x - width, ranges, width, label='Approval Rate Range', color='#e74c3c')
    bars2 = ax.bar(x, means, width, label='Mean Approval Rate', color='#3498db')
    bars3 = ax.bar(x + width, stds, width, label='Std Dev', color='#f39c12')
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
    ax.set_title('Ordering Instability Across Model Scales', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ordering_sensitivity_4models.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "ordering_sensitivity_4models.pdf", bbox_inches="tight")
    plt.close()
    
    # Figure 2: Violin plot showing distributions
    fig, ax = plt.subplots(figsize=(14, 7))
    
    violin_data = []
    labels = []
    for model_name in model_names:
        rates = list(all_results[model_name]["ordering_rates"].values())
        rates_pct = [r * 100 for r in rates]
        violin_data.append(rates_pct)
        labels.append(f"{model_name}\n({len(rates)} orderings)")
    
    parts = ax.violinplot(violin_data, positions=range(len(violin_data)), 
                          showmeans=True, showmedians=True)
    
    # Color the violins
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
    for i, pc in enumerate(parts['bodies']):
        if i < len(colors):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Final Approval Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title('Distribution of Approval Rates Across All Orderings', fontweight='bold', fontsize=14)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "approval_rate_distributions.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "approval_rate_distributions.pdf", bbox_inches="tight")
    plt.close()
    
    # Figure 3: Heatmap for each model (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(24, 14))
    axes = axes.flatten()
    
    for idx, (model_name, model_data) in enumerate(models.items()):
        ax = axes[idx]
        
        df_valid = model_data[model_data["approval_decision"].isin(["approve", "deny"])].copy()
        df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)
        df_final = df_valid[df_valid["is_final_agent"] == True].copy()
        
        # Create pivot table: position x ordering
        position_ordering_rates = df_valid.pivot_table(
            values="approved",
            index="agent_position",
            columns="agent_order",
            aggfunc="mean"
        )
        
        if not position_ordering_rates.empty:
            # Sort columns by final approval rate
            final_rates = df_final.groupby("agent_order")["approved"].mean().sort_values(ascending=False)
            position_ordering_rates = position_ordering_rates[final_rates.index]
            
            # Abbreviated labels
            column_labels = []
            for col in position_ordering_rates.columns:
                short = col.replace("Consumer_Advocate", "CA")
                short = short.replace("Data_Science", "DS")
                short = short.replace("Regulatory", "Reg")
                short = short.replace("Risk_Manager", "RM")
                short = short.replace("_", "\n")
                column_labels.append(short)
            
            position_labels = [f"Pos {i}" for i in position_ordering_rates.index]
            
            sns.heatmap(
                position_ordering_rates * 100,
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
                linecolor='white',
                annot_kws={"fontsize": 6}
            )
            
            overall_approval = df_final["approved"].mean() * 100
            variance = (position_ordering_rates.max().max() - position_ordering_rates.min().min()) * 100
            
            ax.set_title(
                f"{model_name}\n{len(final_rates)} orderings | "
                f"Mean: {overall_approval:.1f}% | Range: {variance:.1f} pp",
                fontweight="bold",
                fontsize=11,
                pad=10
            )
            ax.set_xlabel("Agent Ordering", fontsize=9)
            ax.set_ylabel("Position", fontsize=9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=6)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.suptitle(
        "Approval Rates by Agent Position Across All Orderings - 4 Model Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_dir / "position_ordering_heatmaps_4models.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "position_ordering_heatmaps_4models.pdf", bbox_inches="tight")
    plt.close()
    
    # Save comprehensive metrics
    with open(output_dir / "ordering_instability_4models_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Create CSV summary
    summary_rows = []
    for model_name, results in all_results.items():
        summary_rows.append({
            "Model": model_name,
            "Num_Orderings": results["num_orderings"],
            "Min_Approval_%": f"{results['min_approval']*100:.2f}",
            "Max_Approval_%": f"{results['max_approval']*100:.2f}",
            "Range_pp": f"{results['range']*100:.2f}",
            "Mean_%": f"{results['mean_approval']*100:.2f}",
            "Std_%": f"{results['std_approval']*100:.2f}",
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "ordering_instability_summary.csv", index=False)
    
    print(f"\n✅ Saved to {output_dir}")
    print(f"   - 4-model comparison chart")
    print(f"   - Distribution violin plots")
    print(f"   - 4-panel position×ordering heatmaps")
    print(f"   - JSON metrics & CSV summary")
    
    return all_results


def main():
    """Run comprehensive 4-model ordering instability analysis."""
    print("=" * 80)
    print("SECTION 4.1: ORDERING INSTABILITY - 4 MODEL ANALYSIS")
    print("=" * 80)
    
    # Load all models
    models = {}
    
    # Load small models
    models["Llama 3.2"] = load_small_model_data("outputs_simple/llama3.2", "Llama 3.2")
    models["Mistral Latest"] = load_small_model_data("outputs_simple/mistral:latest", "Mistral Latest")
    
    # Load Qwen
    models["Qwen"] = load_qwen_data()
    
    # Load 70B
    models["Llama 70B"] = load_70b_data()
    
    # Run comprehensive analysis
    output_dir = Path("icml_analysis/section_4_1_ordering_instability")
    results = analyze_ordering_instability_comprehensive(models, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\n📊 4-Model Ordering Sensitivity Summary:")
    print("-" * 80)
    for model_name, res in results.items():
        print(f"{model_name:20s} | Range: {res['range']*100:5.2f} pp | "
              f"Mean: {res['mean_approval']*100:5.2f}% | "
              f"Orderings: {res['num_orderings']}")
    print("-" * 80)


if __name__ == "__main__":
    main()
