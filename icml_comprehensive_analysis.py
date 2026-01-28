"""
Comprehensive Analysis for ICML Paper
Generates all metrics and plots needed for paper sections 4.1-5.2
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add plotting to path
sys.path.insert(0, str(Path(__file__).parent / "plotting"))
from data_loader import load_data

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def load_all_models() -> Dict[str, pd.DataFrame]:
    """Load data from all available models."""
    models = {}
    
    # 70B Model
    print("Loading 70B model data...")
    df_70b, _ = load_data("70bresults")
    models["70B"] = df_70b
    print(f"  Loaded {len(df_70b):,} records")
    
    # Qwen Model
    print("\nLoading Qwen model data...")
    qwen_dir = Path("qwen_validation")
    if qwen_dir.exists():
        records = []
        agent_map = {
            "consumer": "Consumer Advocate",
            "data": "Data Science",
            "regulatory": "Regulatory",
            "risk": "Risk Manager",
        }
        
        for file in sorted(qwen_dir.glob("sequential_*.json")):
            name = file.stem.replace("sequential_", "")
            parts = name.split("_")
            order = []
            for part in parts:
                if part in agent_map:
                    order.append(agent_map[part])
            
            order_str = "_".join([a.replace(" ", "_") for a in order])
            
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
                    agent_key = agent_name.lower().replace(" ", "_")
                    agent_data = decisions.get(agent_key, {})
                    
                    if not agent_data:
                        continue
                    
                    record = base_record.copy()
                    record["agent"] = agent_name
                    record["agent_position"] = i + 1
                    record["is_final_agent"] = i == len(order) - 1
                    record["approval_decision"] = agent_data.get("approval_decision")
                    record["experiment_type"] = "sequential"
                    
                    records.append(record)
        
        df_qwen = pd.DataFrame(records)
        models["Qwen"] = df_qwen
        print(f"  Loaded {len(df_qwen):,} records")
    
    # Small models if available
    for model_name in ["Llama 3.2", "Mistral Latest"]:
        model_dir = Path(f"outputs_{model_name.lower().replace(' ', '_').replace('.', '')}")
        if model_dir.exists():
            print(f"\nLoading {model_name} data...")
            # TODO: Add loader if needed
    
    return models


def analyze_ordering_instability(models: Dict[str, pd.DataFrame], output_dir: Path):
    """Section 4.1: Ordering Instability Analysis"""
    print("\n" + "=" * 80)
    print("SECTION 4.1: ORDERING INSTABILITY")
    print("=" * 80)
    
    results = {}
    
    for model_name, df in models.items():
        df_seq = df[df["experiment_type"] == "sequential"].copy()
        df_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
        df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)
        df_final = df_valid[df_valid["is_final_agent"] == True].copy()
        
        if len(df_final) == 0:
            continue
        
        # Calculate approval rates per ordering
        ordering_rates = df_final.groupby("agent_order")["approved"].mean()
        
        results[model_name] = {
            "num_orderings": len(ordering_rates),
            "min_approval": float(ordering_rates.min()),
            "max_approval": float(ordering_rates.max()),
            "range": float(ordering_rates.max() - ordering_rates.min()),
            "mean_approval": float(ordering_rates.mean()),
            "std_approval": float(ordering_rates.std()),
            "ordering_rates": ordering_rates.to_dict(),
        }
        
        print(f"\n{model_name}:")
        print(f"  Orderings: {results[model_name]['num_orderings']}")
        print(f"  Approval Range: {results[model_name]['min_approval']:.1%} - {results[model_name]['max_approval']:.1%}")
        print(f"  Variance: {results[model_name]['range']:.1%} ({results[model_name]['range']*100:.2f} pp)")
        print(f"  Mean ± Std: {results[model_name]['mean_approval']:.1%} ± {results[model_name]['std_approval']:.1%}")
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(results.keys())
    ranges = [results[m]["range"] * 100 for m in model_names]
    means = [results[m]["mean_approval"] * 100 for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ranges, width, label='Approval Rate Range', color='#e74c3c')
    bars2 = ax.bar(x + width/2, means, width, label='Mean Approval Rate', color='#3498db')
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Ordering Instability Across Models', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ordering_instability_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "ordering_instability_comparison.pdf", bbox_inches="tight")
    plt.close()
    
    # Save metrics
    with open(output_dir / "ordering_instability_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to {output_dir}")
    return results


def analyze_information_cascades(models: Dict[str, pd.DataFrame], output_dir: Path):
    """Section 4.2: Information Cascades Analysis"""
    print("\n" + "=" * 80)
    print("SECTION 4.2: INFORMATION CASCADES")
    print("=" * 80)
    
    results = {}
    
    for model_name, df in models.items():
        df_seq = df[df["experiment_type"] == "sequential"].copy()
        df_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
        df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)
        
        if len(df_valid) == 0:
            continue
        
        # Get first and final decisions per chain
        df_first = df_valid[df_valid["agent_position"] == 1][["chain_id", "approval_decision"]].copy()
        df_first.columns = ["chain_id", "first_decision"]
        
        df_final = df_valid[df_valid["is_final_agent"] == True][["chain_id", "approval_decision"]].copy()
        df_final.columns = ["chain_id", "final_decision"]
        
        df_transitions = df_final.merge(df_first, on="chain_id", how="inner")
        
        # Confusion matrix
        confusion = pd.crosstab(
            df_transitions["first_decision"],
            df_transitions["final_decision"],
            normalize='all'
        )
        
        # Calculate metrics
        first_approve = df_transitions["first_decision"] == "approve"
        final_approve = df_transitions["final_decision"] == "approve"
        
        agreement_rate = ((first_approve & final_approve) | (~first_approve & ~final_approve)).mean()
        flip_approve_to_deny = (first_approve & ~final_approve).mean()
        flip_deny_to_approve = (~first_approve & final_approve).mean()
        overturn_rate = flip_approve_to_deny + flip_deny_to_approve
        
        # Position-wise agreement
        position_agreement = []
        max_pos = int(df_valid["agent_position"].max())
        for pos in range(1, max_pos):
            df_pos1 = df_valid[df_valid["agent_position"] == pos][["chain_id", "approval_decision"]].copy()
            df_pos2 = df_valid[df_valid["agent_position"] == pos + 1][["chain_id", "approval_decision"]].copy()
            df_pos1.columns = ["chain_id", f"pos_{pos}"]
            df_pos2.columns = ["chain_id", f"pos_{pos+1}"]
            
            df_compare = df_pos1.merge(df_pos2, on="chain_id", how="inner")
            if len(df_compare) > 0:
                agreement = (df_compare[f"pos_{pos}"] == df_compare[f"pos_{pos+1}"]).mean()
                position_agreement.append({
                    "transition": f"{pos}→{pos+1}",
                    "agreement_rate": float(agreement)
                })
        
        results[model_name] = {
            "confusion_matrix": confusion.to_dict(),
            "first_final_agreement": float(agreement_rate),
            "flip_approve_to_deny": float(flip_approve_to_deny),
            "flip_deny_to_approve": float(flip_deny_to_approve),
            "overturn_rate": float(overturn_rate),
            "position_agreement": position_agreement,
        }
        
        print(f"\n{model_name}:")
        print(f"  First→Final Agreement: {agreement_rate:.1%}")
        print(f"  Overturn Rate: {overturn_rate:.1%}")
        print(f"  Approve→Deny Flips: {flip_approve_to_deny:.1%}")
        print(f"  Deny→Approve Flips: {flip_deny_to_approve:.1%}")
        print(f"  Position-wise Agreement:")
        for item in position_agreement:
            print(f"    {item['transition']}: {item['agreement_rate']:.1%}")
    
    # Create cascade visualization
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for ax, (model_name, model_results) in zip(axes, results.items()):
        # Plot position agreement rates
        transitions = [item["transition"] for item in model_results["position_agreement"]]
        agreements = [item["agreement_rate"] * 100 for item in model_results["position_agreement"]]
        
        ax.plot(transitions, agreements, marker='o', linewidth=2, markersize=10, color='#3498db')
        ax.fill_between(range(len(transitions)), agreements, alpha=0.3, color='#3498db')
        ax.axhline(y=100 * model_results["first_final_agreement"], color='red', 
                   linestyle='--', label=f'First→Final: {model_results["first_final_agreement"]:.1%}')
        
        ax.set_xlabel('Agent Position Transition', fontweight='bold')
        ax.set_ylabel('Agreement Rate (%)', fontweight='bold')
        ax.set_title(f'{model_name} - Information Cascade', fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "information_cascades.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "information_cascades.pdf", bbox_inches="tight")
    plt.close()
    
    # Save metrics
    with open(output_dir / "information_cascades_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to {output_dir}")
    return results


def analyze_parallel_mode(models: Dict[str, pd.DataFrame], output_dir: Path):
    """Section 4.3: Parallel Mode Analysis"""
    print("\n" + "=" * 80)
    print("SECTION 4.3: PARALLEL MODE ANALYSIS")
    print("=" * 80)
    
    results = {}
    
    for model_name, df in models.items():
        df_seq = df[df["experiment_type"] == "sequential"].copy()
        df_par = df[df["experiment_type"] == "parallel"].copy()
        
        if len(df_par) == 0:
            print(f"\n{model_name}: No parallel data available")
            continue
        
        # Sequential stats
        df_seq_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
        df_seq_valid["approved"] = (df_seq_valid["approval_decision"] == "approve").astype(int)
        df_seq_final = df_seq_valid[df_seq_valid["is_final_agent"] == True].copy()
        
        seq_approval_rates = df_seq_final.groupby("agent_order")["approved"].mean()
        seq_mean = seq_approval_rates.mean()
        seq_std = seq_approval_rates.std()
        seq_range = seq_approval_rates.max() - seq_approval_rates.min()
        
        # Parallel stats
        df_par_valid = df_par[df_par["approval_decision"].isin(["approve", "deny"])].copy()
        df_par_valid["approved"] = (df_par_valid["approval_decision"] == "approve").astype(int)
        df_par_final = df_par_valid[df_par_valid["is_final_agent"] == True].copy()
        
        par_approval = df_par_final["approved"].mean()
        
        results[model_name] = {
            "sequential_mean": float(seq_mean),
            "sequential_std": float(seq_std),
            "sequential_range": float(seq_range),
            "parallel_approval": float(par_approval),
            "parallel_vs_seq_mean_diff": float(par_approval - seq_mean),
            "variance_reduction": float((seq_std ** 2) if seq_std > 0 else 0),
        }
        
        print(f"\n{model_name}:")
        print(f"  Sequential Mean: {seq_mean:.1%} ± {seq_std:.1%}")
        print(f"  Sequential Range: {seq_range:.1%}")
        print(f"  Parallel Approval: {par_approval:.1%}")
        print(f"  Diff (Parallel - Seq Mean): {results[model_name]['parallel_vs_seq_mean_diff']:.1%}")
    
    # Save metrics
    with open(output_dir / "parallel_analysis_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to {output_dir}")
    return results


def analyze_functional_collapse(models: Dict[str, pd.DataFrame], output_dir: Path):
    """Section 4.3: Functional Collapse Analysis"""
    print("\n" + "=" * 80)
    print("SECTION 4.3: FUNCTIONAL COLLAPSE")
    print("=" * 80)
    
    results = {}
    
    for model_name, df in models.items():
        df_seq = df[df["experiment_type"] == "sequential"].copy()
        df_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
        df_valid["approved"] = (df_valid["approval_decision"] == "approve").astype(int)
        df_final = df_valid[df_valid["is_final_agent"] == True].copy()
        
        if len(df_final) == 0:
            continue
        
        # Ethnicity fairness (demographic parity)
        if "ethnicity_signal" in df_final.columns and df_final["ethnicity_signal"].notna().any():
            ethnicity_rates = df_final.groupby("ethnicity_signal")["approved"].mean()
            ethnicity_max_diff = ethnicity_rates.max() - ethnicity_rates.min()
            
            ethnicity_metrics = {
                "rates": ethnicity_rates.to_dict(),
                "max_disparity": float(ethnicity_max_diff),
                "mean": float(ethnicity_rates.mean()),
                "std": float(ethnicity_rates.std()),
            }
        else:
            ethnicity_metrics = None
        
        # Credit score correlation
        if "credit_score" in df_final.columns and df_final["credit_score"].notna().any():
            # Calculate quartiles
            df_credit = df_final[df_final["credit_score"].notna()].copy()
            df_credit["credit_quartile"] = pd.qcut(df_credit["credit_score"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
            
            quartile_rates = df_credit.groupby("credit_quartile")["approved"].mean()
            
            # Correlation
            correlation = df_credit["credit_score"].corr(df_credit["approved"])
            
            credit_metrics = {
                "quartile_rates": quartile_rates.to_dict(),
                "correlation": float(correlation),
                "q4_q1_diff": float(quartile_rates.iloc[-1] - quartile_rates.iloc[0]),
            }
        else:
            credit_metrics = None
        
        # Find orderings with fairness but weak credit correlation
        ordering_analysis = []
        for ordering in df_final["agent_order"].unique():
            df_ord = df_final[df_final["agent_order"] == ordering].copy()
            
            if len(df_ord) < 100:  # Skip small samples
                continue
            
            # Ethnicity disparity
            if "ethnicity_signal" in df_ord.columns and df_ord["ethnicity_signal"].notna().any():
                eth_rates = df_ord.groupby("ethnicity_signal")["approved"].mean()
                eth_disparity = eth_rates.max() - eth_rates.min()
            else:
                eth_disparity = None
            
            # Credit correlation
            if "credit_score" in df_ord.columns and df_ord["credit_score"].notna().any():
                df_ord_credit = df_ord[df_ord["credit_score"].notna()].copy()
                credit_corr = df_ord_credit["credit_score"].corr(df_ord_credit["approved"])
            else:
                credit_corr = None
            
            if eth_disparity is not None and credit_corr is not None:
                ordering_analysis.append({
                    "ordering": ordering,
                    "ethnicity_disparity": float(eth_disparity),
                    "credit_correlation": float(credit_corr),
                    "is_fair_weak_credit": bool(eth_disparity < 0.1 and abs(credit_corr) < 0.3),  # Thresholds
                })
        
        results[model_name] = {
            "ethnicity_metrics": ethnicity_metrics,
            "credit_metrics": credit_metrics,
            "ordering_analysis": ordering_analysis,
            "num_fair_weak_credit": sum(1 for x in ordering_analysis if x["is_fair_weak_credit"]),
        }
        
        print(f"\n{model_name}:")
        if ethnicity_metrics:
            print(f"  Ethnicity Max Disparity: {ethnicity_metrics['max_disparity']:.1%}")
        if credit_metrics:
            print(f"  Credit Score Correlation: {credit_metrics['correlation']:.3f}")
            print(f"  Q4-Q1 Difference: {credit_metrics['q4_q1_diff']:.1%}")
        print(f"  Orderings with Fair+Weak Credit: {results[model_name]['num_fair_weak_credit']}/{len(ordering_analysis)}")
    
    # Save metrics
    with open(output_dir / "functional_collapse_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to {output_dir}")
    return results


def analyze_scale_paradox(models: Dict[str, pd.DataFrame], output_dir: Path):
    """Section 5.2: Scale Paradox Analysis"""
    print("\n" + "=" * 80)
    print("SECTION 5.2: SCALE PARADOX")
    print("=" * 80)
    
    results = {}
    
    for model_name, df in models.items():
        df_seq = df[df["experiment_type"] == "sequential"].copy()
        df_valid = df_seq[df_seq["approval_decision"].isin(["approve", "deny"])].copy()
        
        if len(df_valid) == 0:
            continue
        
        # Inter-agent agreement rate (all agents in chain agree)
        chain_decisions = df_valid.groupby("chain_id")["approval_decision"].agg(lambda x: x.nunique())
        inter_agent_agreement = (chain_decisions == 1).mean()
        
        # Error correction analysis
        # Define "error" as deviation from majority/final decision
        error_correction_rates = []
        
        for ordering in df_valid["agent_order"].unique():
            df_ord = df_valid[df_valid["agent_order"] == ordering].copy()
            
            # Get first and final decisions
            df_first = df_ord[df_ord["agent_position"] == 1][["chain_id", "approval_decision"]].copy()
            df_final = df_ord[df_ord["is_final_agent"] == True][["chain_id", "approval_decision"]].copy()
            
            df_first.columns = ["chain_id", "first"]
            df_final.columns = ["chain_id", "final"]
            
            df_compare = df_first.merge(df_final, on="chain_id", how="inner")
            
            if len(df_compare) > 0:
                # Count corrections (first != final)
                correction_rate = (df_compare["first"] != df_compare["final"]).mean()
                error_correction_rates.append(correction_rate)
        
        avg_correction_rate = np.mean(error_correction_rates) if error_correction_rates else 0
        
        results[model_name] = {
            "inter_agent_agreement": float(inter_agent_agreement),
            "avg_correction_rate": float(avg_correction_rate),
            "consensus_vs_correction": {
                "high_consensus": float(inter_agent_agreement),
                "low_correction": float(1 - avg_correction_rate),
                "paradox_score": float(inter_agent_agreement - avg_correction_rate),
            }
        }
        
        print(f"\n{model_name}:")
        print(f"  Inter-Agent Agreement: {inter_agent_agreement:.1%}")
        print(f"  Avg Correction Rate: {avg_correction_rate:.1%}")
        print(f"  Paradox Score (Agreement - Correction): {results[model_name]['consensus_vs_correction']['paradox_score']:.1%}")
    
    # Create scale paradox visualization
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = list(results.keys())
        agreement = [results[m]["inter_agent_agreement"] * 100 for m in model_names]
        correction = [results[m]["avg_correction_rate"] * 100 for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, agreement, width, label='Agreement Rate', color='#27ae60')
        bars2 = ax.bar(x + width/2, correction, width, label='Correction Rate', color='#e74c3c')
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title('Scale Paradox: Agreement vs Correction Rates', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "scale_paradox.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / "scale_paradox.pdf", bbox_inches="tight")
        plt.close()
    
    # Save metrics
    with open(output_dir / "scale_paradox_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to {output_dir}")
    return results


def main():
    """Run comprehensive ICML analysis."""
    print("=" * 80)
    print("ICML COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Load all models
    models = load_all_models()
    
    base_dir = Path("icml_analysis")
    
    # Section 4.1: Ordering Instability
    analyze_ordering_instability(
        models,
        base_dir / "section_4_1_ordering_instability"
    )
    
    # Section 4.2: Information Cascades
    analyze_information_cascades(
        models,
        base_dir / "section_4_2_information_cascades"
    )
    
    # Section 4.3: Parallel Mode
    analyze_parallel_mode(
        models,
        base_dir / "section_4_3_parallel_functional_collapse"
    )
    
    # Section 4.3: Functional Collapse
    analyze_functional_collapse(
        models,
        base_dir / "section_4_3_parallel_functional_collapse"
    )
    
    # Section 5.2: Scale Paradox
    analyze_scale_paradox(
        models,
        base_dir / "section_5_2_scale_paradox"
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {base_dir.absolute()}")


if __name__ == "__main__":
    main()
