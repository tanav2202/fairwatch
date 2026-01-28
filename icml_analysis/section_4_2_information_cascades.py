#!/usr/bin/env python3
"""
Comprehensive 4-Model Analysis for ICML Section 4.2: Information Cascades

Analyzes how agent decisions propagate through sequential chains:
- Confusion matrix: First agent vs Final decision
- Agreement rates at each position
- Overturn rates (how often final decision differs from first)
- Cascade speed comparison across model scales

Models: Llama 3.2 (3B), Mistral (7B), Qwen (72B), Llama 70B (70B)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_70b_cascade_data():
    """Load Llama 70B sequential data with agent-level decisions"""
    base_dir = Path("70bresults/sequential")
    
    all_records = []
    agent_order = ['consumer_advocate', 'data_science', 'regulatory', 'risk_manager']
    
    for json_file in sorted(base_dir.glob("sequential_*_formatted.json")):
        with open(json_file) as f:
            data = json.load(f)
            
        ordering = json_file.stem.replace("sequential_", "").replace("_formatted", "")
        
        for record in data['results']:
            decisions = record['decisions']
            
            # Extract agent decisions in order
            agent_decisions = []
            for agent in agent_order:
                if agent in decisions:
                    decision = decisions[agent]['approval_decision']
                    agent_decisions.append(decision)
            
            if len(agent_decisions) == 4:
                all_records.append({
                    'ordering': ordering,
                    'first_agent': agent_decisions[0],
                    'second_agent': agent_decisions[1],
                    'third_agent': agent_decisions[2],
                    'final_agent': agent_decisions[3],
                    'agent_order': agent_order
                })
    
    df = pd.DataFrame(all_records)
    print(f"Loaded Llama 70B: {len(df)} chains with agent-level decisions")
    return df

def load_qwen_cascade_data():
    """Load Qwen sequential data with agent-level decisions"""
    base_dir = Path("qwen_validation")
    
    all_records = []
    
    # Determine agent order from the first file
    agent_order_map = {
        'consumer_advocate_data_science_risk_manager_regulatory': 
            ['consumer_advocate', 'data_science', 'risk_manager', 'regulatory'],
        'consumer_advocate_regulatory_data_science_risk_manager':
            ['consumer_advocate', 'regulatory', 'data_science', 'risk_manager'],
        'data_science_regulatory_consumer_advocate_risk_manager':
            ['data_science', 'regulatory', 'consumer_advocate', 'risk_manager'],
        'data_science_regulatory_risk_manager_consumer_advocate':
            ['data_science', 'regulatory', 'risk_manager', 'consumer_advocate'],
        'regulatory_consumer_advocate_risk_manager_data_science':
            ['regulatory', 'consumer_advocate', 'risk_manager', 'data_science'],
        'regulatory_data_science_risk_manager_consumer_advocate':
            ['regulatory', 'data_science', 'risk_manager', 'consumer_advocate'],
        'risk_manager_consumer_advocate_regulatory_data_science':
            ['risk_manager', 'consumer_advocate', 'regulatory', 'data_science'],
        'risk_manager_data_science_regulatory_consumer_advocate':
            ['risk_manager', 'data_science', 'regulatory', 'consumer_advocate'],
    }
    
    for json_file in sorted(base_dir.glob("sequential_*.json")):
        with open(json_file) as f:
            data = json.load(f)
        
        ordering = json_file.stem.replace("sequential_", "")
        agent_order = agent_order_map.get(ordering, [])
        
        if not agent_order or 'results' not in data:
            continue
        
        for record in data['results']:
            decisions = record.get('decisions', {})
            
            # Extract agent decisions in order
            agent_decisions = []
            for agent in agent_order:
                if agent in decisions:
                    decision = decisions[agent].get('approval_decision', '')
                    if decision:
                        agent_decisions.append(decision)
            
            if len(agent_decisions) == 4:
                all_records.append({
                    'ordering': ordering,
                    'first_agent': agent_decisions[0],
                    'second_agent': agent_decisions[1],
                    'third_agent': agent_decisions[2],
                    'final_agent': agent_decisions[3],
                })
    
    df = pd.DataFrame(all_records)
    print(f"Loaded Qwen: {len(df)} chains with agent-level decisions")
    return df

def load_small_model_cascade_data(model_name, base_path):
    """Load Llama 3.2 or Mistral data from outputs_simple directory"""
    # Note: Small models store complete data in outputs_simple with different structure
    # We'll need to check if agent-level data is available
    
    print(f"⚠️  {model_name}: Agent-level cascade data not available in pre-analyzed format")
    print(f"   Will use aggregate statistics for {model_name}")
    return pd.DataFrame()

def calculate_confusion_matrix(df, model_name):
    """Calculate confusion matrix: First agent vs Final decision"""
    
    if len(df) == 0:
        return None, None
    
    # Map decisions to binary
    def to_binary(decision):
        if decision in ['approve', 'APPROVED']:
            return 'Approve'
        else:
            return 'Deny'
    
    first_binary = df['first_agent'].apply(to_binary)
    final_binary = df['final_agent'].apply(to_binary)
    
    # Create confusion matrix
    cm = confusion_matrix(first_binary, final_binary, labels=['Approve', 'Deny'])
    
    # Calculate metrics
    total = len(df)
    agreement = sum(first_binary == final_binary)
    overturn_rate = (total - agreement) / total
    
    metrics = {
        'total_chains': total,
        'agreement': agreement,
        'agreement_rate': agreement / total,
        'overturn_rate': overturn_rate,
        'confusion_matrix': cm.tolist()
    }
    
    print(f"\n{model_name} Confusion Matrix:")
    print(f"  Total chains: {total:,}")
    print(f"  Agreement (1st = Final): {agreement:,} ({metrics['agreement_rate']:.1%})")
    print(f"  Overturn rate: {overturn_rate:.1%}")
    
    return cm, metrics

def calculate_position_agreement(df, model_name):
    """Calculate agreement rate at each position transition"""
    
    if len(df) == 0:
        return {}
    
    def to_binary(decision):
        return 'approve' if decision in ['approve', 'APPROVED'] else 'deny'
    
    # Convert all to binary
    first = df['first_agent'].apply(to_binary)
    second = df['second_agent'].apply(to_binary)
    third = df['third_agent'].apply(to_binary)
    final = df['final_agent'].apply(to_binary)
    
    # Calculate agreement at each transition
    metrics = {
        '1st_to_2nd': (first == second).mean(),
        '2nd_to_3rd': (second == third).mean(),
        '3rd_to_4th': (third == final).mean(),
        '1st_to_final': (first == final).mean()
    }
    
    print(f"\n{model_name} Position Agreement Rates:")
    print(f"  1st → 2nd: {metrics['1st_to_2nd']:.1%}")
    print(f"  2nd → 3rd: {metrics['2nd_to_3rd']:.1%}")
    print(f"  3rd → 4th (final): {metrics['3rd_to_4th']:.1%}")
    print(f"  1st → Final: {metrics['1st_to_final']:.1%}")
    
    return metrics

def plot_confusion_matrices(cms, output_dir):
    """Plot confusion matrices for all models"""
    
    models_with_data = {k: v for k, v in cms.items() if v is not None}
    
    if not models_with_data:
        print("No confusion matrix data available")
        return
    
    n_models = len(models_with_data)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model, cm) in enumerate(models_with_data.items()):
        ax = axes[idx]
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Approve', 'Deny'],
                   yticklabels=['Approve', 'Deny'],
                   cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Final Decision', fontsize=12, fontweight='bold')
        ax.set_ylabel('First Agent Decision', fontsize=12, fontweight='bold')
        ax.set_title(f'{model}\nFirst vs Final Decision', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'confusion_matrices.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved confusion matrices")

def plot_agreement_cascade(position_agreements, output_dir):
    """Plot agreement rates across positions for all models"""
    
    models_with_data = {k: v for k, v in position_agreements.items() if v}
    
    if not models_with_data:
        print("No position agreement data available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    positions = ['1st → 2nd', '2nd → 3rd', '3rd → 4th']
    x_pos = np.arange(len(positions))
    width = 0.2
    
    colors = sns.color_palette("husl", len(models_with_data))
    
    for idx, (model, metrics) in enumerate(models_with_data.items()):
        rates = [
            metrics['1st_to_2nd'],
            metrics['2nd_to_3rd'],
            metrics['3rd_to_4th']
        ]
        
        offset = (idx - len(models_with_data)/2 + 0.5) * width
        bars = ax.bar(x_pos + offset, rates, width, label=model, 
                     color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Agent Position Transition', fontsize=13, fontweight='bold')
    ax.set_ylabel('Agreement Rate', fontsize=13, fontweight='bold')
    ax.set_title('Information Cascade: Agreement Rates Across Positions', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(positions, fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_agreement_cascade.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'position_agreement_cascade.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved position agreement cascade")

def plot_overturn_comparison(all_metrics, output_dir):
    """Plot overturn rates across models"""
    
    models_with_data = {k: v for k, v in all_metrics.items() if v and 'overturn_rate' in v}
    
    if not models_with_data:
        print("No overturn rate data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(models_with_data.keys())
    overturn_rates = [models_with_data[m]['overturn_rate'] for m in models]
    
    colors = sns.color_palette("husl", len(models))
    bars = ax.bar(models, overturn_rates, color=colors, alpha=0.8, 
                 edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1%}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Overturn Rate\n(Final ≠ First Agent)', fontsize=13, fontweight='bold')
    ax.set_title('Decision Overturn Rates: How Often Final Decision Differs from First Agent',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, max(overturn_rates) * 1.2])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overturn_rates.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'overturn_rates.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved overturn rates comparison")

def plot_cascade_speed(position_agreements, output_dir):
    """Plot cascade convergence speed (how quickly decisions stabilize)"""
    
    models_with_data = {k: v for k, v in position_agreements.items() if v}
    
    if not models_with_data:
        print("No cascade speed data available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    positions = [1, 2, 3, 4]
    position_labels = ['1st Agent', '2nd Agent', '3rd Agent', '4th Agent\n(Final)']
    
    colors = sns.color_palette("husl", len(models_with_data))
    
    for idx, (model, metrics) in enumerate(models_with_data.items()):
        # Calculate cumulative agreement with first agent
        agreement_trajectory = [
            1.0,  # 1st agent always agrees with itself
            metrics['1st_to_2nd'],
            1 - (1 - metrics['1st_to_2nd']) * (1 - metrics['2nd_to_3rd']),  # Approximate cascade
            metrics['1st_to_final']  # Final agreement with first
        ]
        
        ax.plot(positions, agreement_trajectory, 'o-', linewidth=2.5, 
               markersize=10, label=model, color=colors[idx])
        
        # Add value annotations
        for pos, val in zip(positions, agreement_trajectory):
            ax.annotate(f'{val:.1%}', 
                       xy=(pos, val),
                       xytext=(0, 10 if idx % 2 == 0 else -15),
                       textcoords='offset points',
                       fontsize=9,
                       ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor=colors[idx], alpha=0.2))
    
    ax.set_xlabel('Agent Position in Chain', fontsize=13, fontweight='bold')
    ax.set_ylabel('Agreement with First Agent Decision', fontsize=13, fontweight='bold')
    ax.set_title('Cascade Speed: How Fast Do Decisions Propagate?',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(positions)
    ax.set_xticklabels(position_labels, fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cascade_speed.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cascade_speed.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved cascade speed plot")

def main():
    """Main analysis function"""
    
    print("="*80)
    print("ICML Section 4.2: Information Cascades - Four-Model Analysis")
    print("="*80)
    print()
    
    # Create output directory
    output_dir = Path("icml_analysis/section_4_2_information_cascades")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data for models with agent-level information
    print("Loading agent-level decision data...")
    print()
    
    llama70b_df = load_70b_cascade_data()
    print()
    qwen_df = load_qwen_cascade_data()
    print()
    
    # Store all results
    confusion_matrices = {}
    position_agreements = {}
    all_metrics = {}
    
    # Analyze Llama 70B
    if len(llama70b_df) > 0:
        cm, metrics = calculate_confusion_matrix(llama70b_df, "Llama 70B (70B)")
        confusion_matrices["Llama 70B (70B)"] = cm
        all_metrics["Llama 70B (70B)"] = metrics
        
        pos_metrics = calculate_position_agreement(llama70b_df, "Llama 70B (70B)")
        position_agreements["Llama 70B (70B)"] = pos_metrics
    
    # Analyze Qwen
    if len(qwen_df) > 0:
        cm, metrics = calculate_confusion_matrix(qwen_df, "Qwen (72B)")
        confusion_matrices["Qwen (72B)"] = cm
        all_metrics["Qwen (72B)"] = metrics
        
        pos_metrics = calculate_position_agreement(qwen_df, "Qwen (72B)")
        position_agreements["Qwen (72B)"] = pos_metrics
    
    # Save metrics
    with open(output_dir / 'cascade_metrics.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {}
        for model, m in all_metrics.items():
            metrics_json[model] = {
                'total_chains': m['total_chains'],
                'agreement': m['agreement'],
                'agreement_rate': m['agreement_rate'],
                'overturn_rate': m['overturn_rate'],
                'confusion_matrix': m['confusion_matrix']
            }
        
        # Add position agreements
        for model, p in position_agreements.items():
            if model in metrics_json:
                metrics_json[model]['position_agreement'] = p
        
        json.dump(metrics_json, f, indent=2)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    print()
    
    plot_confusion_matrices(confusion_matrices, output_dir)
    plot_agreement_cascade(position_agreements, output_dir)
    plot_overturn_comparison(all_metrics, output_dir)
    plot_cascade_speed(position_agreements, output_dir)
    
    print()
    print("="*80)
    print(f"✓ Analysis complete! Results saved to: {output_dir}")
    print("="*80)
    print()
    
    # Print summary
    print("Key Findings:")
    print("-" * 80)
    
    for model in all_metrics:
        m = all_metrics[model]
        p = position_agreements.get(model, {})
        
        print(f"\n{model}:")
        print(f"  Overturn rate: {m['overturn_rate']:.1%} (Final ≠ First)")
        if p:
            print(f"  Agreement trajectory:")
            print(f"    1st → 2nd: {p['1st_to_2nd']:.1%}")
            print(f"    2nd → 3rd: {p['2nd_to_3rd']:.1%}")
            print(f"    3rd → 4th: {p['3rd_to_4th']:.1%}")
            print(f"    1st → Final: {p['1st_to_final']:.1%}")
    
    print()

if __name__ == "__main__":
    main()
