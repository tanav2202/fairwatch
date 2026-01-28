#!/usr/bin/env python3
"""
Comprehensive 4-Model Analysis for ICML Section 4.1: Ordering Instability

Analyzes ordering sensitivity across model scales:
- Llama 3.2 (3B) - 24 orderings
- Mistral Latest (7B) - 24 orderings  
- Qwen (72B) - 8 orderings
- Llama 70B (70B) - 24 orderings

Tests hypothesis: Does ordering sensitivity increase with model scale?
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_llama32_data():
    """Load Llama 3.2 (3B) sequential ordering data"""
    base_dir = Path("plotting/outputs/outputs_simple/llama3.2/sequential")
    
    orderings = []
    for ordering_dir in sorted(base_dir.iterdir()):
        if ordering_dir.is_dir():
            summary_file = ordering_dir / "summary_stats.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                    orderings.append({
                        'ordering': ordering_dir.name,
                        'approval_rate': data['final_approval_rate'],
                        'total_chains': data['total_chains']
                    })
    
    df = pd.DataFrame(orderings)
    print(f"Loaded Llama 3.2: {len(df)} orderings")
    print(f"  Range: {df['approval_rate'].min():.1%} - {df['approval_rate'].max():.1%}")
    print(f"  Std: {df['approval_rate'].std():.3f}")
    return df

def load_mistral_data():
    """Load Mistral Latest (7B) sequential ordering data (excluding 0% anomalies)"""
    base_dir = Path("plotting/outputs/outputs_simple/mistral_latest/sequential")
    
    orderings = []
    filtered_count = 0
    for ordering_dir in sorted(base_dir.iterdir()):
        if ordering_dir.is_dir():
            summary_file = ordering_dir / "summary_stats.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                    approval_rate = data['final_approval_rate']
                    
                    # Filter out 0% approval orderings for consistency
                    if approval_rate == 0.0:
                        filtered_count += 1
                        continue
                    
                    orderings.append({
                        'ordering': ordering_dir.name,
                        'approval_rate': approval_rate,
                        'total_chains': data['total_chains']
                    })
    
    df = pd.DataFrame(orderings)
    print(f"Loaded Mistral Latest: {len(df)} orderings (filtered {filtered_count} with 0% approval)")
    print(f"  Range: {df['approval_rate'].min():.1%} - {df['approval_rate'].max():.1%}")
    print(f"  Std: {df['approval_rate'].std():.3f}")
    return df

def load_qwen_data():
    """Load Qwen (72B) sequential ordering data"""
    base_dir = Path("plotting/outputs/qwen_validation/sequential")
    
    orderings = []
    for ordering_dir in sorted(base_dir.iterdir()):
        if ordering_dir.is_dir():
            summary_file = ordering_dir / "summary_stats.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                    orderings.append({
                        'ordering': ordering_dir.name,
                        'approval_rate': data['final_approval_rate'],
                        'total_chains': data['total_chains']
                    })
    
    df = pd.DataFrame(orderings)
    print(f"Loaded Qwen: {len(df)} orderings")
    print(f"  Range: {df['approval_rate'].min():.1%} - {df['approval_rate'].max():.1%}")
    print(f"  Std: {df['approval_rate'].std():.3f}")
    return df

def load_llama70b_data():
    """Load Llama 70B sequential ordering data"""
    base_dir = Path("plotting/outputs/individual/sequential")
    
    orderings = []
    for ordering_dir in sorted(base_dir.iterdir()):
        if ordering_dir.is_dir():
            summary_file = ordering_dir / "summary_stats.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                    orderings.append({
                        'ordering': ordering_dir.name,
                        'approval_rate': data['final_approval_rate'],
                        'total_chains': data['total_chains']
                    })
    
    df = pd.DataFrame(orderings)
    print(f"Loaded Llama 70B: {len(df)} orderings")
    print(f"  Range: {df['approval_rate'].min():.1%} - {df['approval_rate'].max():.1%}")
    print(f"  Std: {df['approval_rate'].std():.3f}")
    return df

def calculate_ordering_instability_metrics(llama32_df, mistral_df, qwen_df, llama70b_df):
    """Calculate comprehensive ordering instability metrics"""
    
    metrics = {
        'Llama 3.2 (3B)': {
            'size': '3B',
            'num_orderings': len(llama32_df),
            'min_approval': llama32_df['approval_rate'].min(),
            'max_approval': llama32_df['approval_rate'].max(),
            'range': llama32_df['approval_rate'].max() - llama32_df['approval_rate'].min(),
            'mean': llama32_df['approval_rate'].mean(),
            'std': llama32_df['approval_rate'].std(),
            'cv': llama32_df['approval_rate'].std() / llama32_df['approval_rate'].mean(),
        },
        'Mistral (7B)': {
            'size': '7B',
            'num_orderings': len(mistral_df),
            'min_approval': mistral_df['approval_rate'].min(),
            'max_approval': mistral_df['approval_rate'].max(),
            'range': mistral_df['approval_rate'].max() - mistral_df['approval_rate'].min(),
            'mean': mistral_df['approval_rate'].mean(),
            'std': mistral_df['approval_rate'].std(),
            'cv': mistral_df['approval_rate'].std() / mistral_df['approval_rate'].mean(),
        },
        'Llama 70B (70B)': {
            'size': '70B',
            'num_orderings': len(llama70b_df),
            'min_approval': llama70b_df['approval_rate'].min(),
            'max_approval': llama70b_df['approval_rate'].max(),
            'range': llama70b_df['approval_rate'].max() - llama70b_df['approval_rate'].min(),
            'mean': llama70b_df['approval_rate'].mean(),
            'std': llama70b_df['approval_rate'].std(),
            'cv': llama70b_df['approval_rate'].std() / llama70b_df['approval_rate'].mean(),
        },
        'Qwen (72B)': {
            'size': '72B',
            'num_orderings': len(qwen_df),
            'min_approval': qwen_df['approval_rate'].min(),
            'max_approval': qwen_df['approval_rate'].max(),
            'range': qwen_df['approval_rate'].max() - qwen_df['approval_rate'].min(),
            'mean': qwen_df['approval_rate'].mean(),
            'std': qwen_df['approval_rate'].std(),
            'cv': qwen_df['approval_rate'].std() / qwen_df['approval_rate'].mean(),
        }
    }
    
    return metrics

def plot_model_comparison(metrics, output_dir):
    """Create comparison bar charts across models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = list(metrics.keys())
    colors = sns.color_palette("husl", len(models))
    
    # 1. Min-Max Range
    ax = axes[0, 0]
    ranges = [metrics[m]['range'] for m in models]
    bars = ax.bar(models, ranges, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Approval Rate Range (Max - Min)', fontsize=12, fontweight='bold')
    ax.set_title('Ordering Sensitivity: Approval Rate Range', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. Standard Deviation
    ax = axes[0, 1]
    stds = [metrics[m]['std'] for m in models]
    bars = ax.bar(models, stds, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax.set_title('Ordering Variability: Standard Deviation', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 3. Coefficient of Variation
    ax = axes[1, 0]
    cvs = [metrics[m]['cv'] for m in models]
    bars = ax.bar(models, cvs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
    ax.set_title('Relative Ordering Sensitivity: CV = σ/μ', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 4. Mean Approval Rate
    ax = axes[1, 1]
    means = [metrics[m]['mean'] for m in models]
    bars = ax.bar(models, means, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Mean Approval Rate', fontsize=12, fontweight='bold')
    ax.set_title('Average Approval Rate Across Orderings', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_model_ordering_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '4_model_ordering_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved 4-model comparison chart")

def plot_violin_distributions(llama32_df, mistral_df, qwen_df, llama70b_df, output_dir):
    """Create violin plots showing approval rate distributions"""
    
    # Prepare data for violin plot
    plot_data = []
    
    for rate in llama32_df['approval_rate']:
        plot_data.append({'Model': 'Llama 3.2\n(3B)', 'Approval Rate': rate})
    
    for rate in mistral_df['approval_rate']:
        plot_data.append({'Model': 'Mistral\n(7B)', 'Approval Rate': rate})
    
    for rate in llama70b_df['approval_rate']:
        plot_data.append({'Model': 'Llama 70B\n(70B)', 'Approval Rate': rate})
    
    for rate in qwen_df['approval_rate']:
        plot_data.append({'Model': 'Qwen\n(72B)', 'Approval Rate': rate})
    
    df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create violin plot
    parts = ax.violinplot(
        [llama32_df['approval_rate'].values,
         mistral_df['approval_rate'].values,
         llama70b_df['approval_rate'].values,
         qwen_df['approval_rate'].values],
        positions=[1, 2, 3, 4],
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # Customize colors
    colors = sns.color_palette("husl", 4)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Add labels
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Llama 3.2\n(3B)', 'Mistral\n(7B)', 'Llama 70B\n(70B)', 'Qwen\n(72B)'],
                       fontsize=12, fontweight='bold')
    ax.set_ylabel('Approval Rate', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Approval Rates Across Orderings by Model',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add statistics annotations
    stats_text = []
    for i, (name, df_model) in enumerate([
        ('Llama 3.2', llama32_df),
        ('Mistral', mistral_df),
        ('Llama 70B', llama70b_df),
        ('Qwen', qwen_df)
    ], 1):
        median = df_model['approval_rate'].median()
        ax.text(i, 1.05, f'Median: {median:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_model_violin_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '4_model_violin_distributions.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved violin distribution plot")

def plot_scale_sensitivity_trend(metrics, output_dir):
    """Plot ordering sensitivity vs model size"""
    
    # Extract model sizes and sensitivities
    model_sizes = []
    ranges = []
    stds = []
    cvs = []
    names = []
    
    size_order = {
        'Llama 3.2 (3B)': (3, '3B'),
        'Mistral (7B)': (7, '7B'),
        'Llama 70B (70B)': (70, '70B'),
        'Qwen (72B)': (72, '72B')
    }
    
    for model in sorted(metrics.keys(), key=lambda x: size_order[x][0]):
        size_num, size_label = size_order[model]
        model_sizes.append(size_num)
        ranges.append(metrics[model]['range'])
        stds.append(metrics[model]['std'])
        cvs.append(metrics[model]['cv'])
        names.append(model.split('(')[0].strip())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Range vs Model Size
    ax = axes[0]
    ax.plot(model_sizes, ranges, 'o-', linewidth=2.5, markersize=10, color='#e74c3c')
    ax.set_xlabel('Model Size (Billions of Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Approval Rate Range', fontsize=12, fontweight='bold')
    ax.set_title('Ordering Sensitivity vs Model Scale', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    for i, (size, rng, name) in enumerate(zip(model_sizes, ranges, names)):
        ax.annotate(f'{name}\n{rng:.1%}',
                   xy=(size, rng),
                   xytext=(10, 10 if i % 2 == 0 else -20),
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # 2. Std Dev vs Model Size
    ax = axes[1]
    ax.plot(model_sizes, stds, 'o-', linewidth=2.5, markersize=10, color='#3498db')
    ax.set_xlabel('Model Size (Billions of Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax.set_title('Ordering Variability vs Model Scale', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    for i, (size, std, name) in enumerate(zip(model_sizes, stds, names)):
        ax.annotate(f'{name}\n{std:.3f}',
                   xy=(size, std),
                   xytext=(10, 10 if i % 2 == 0 else -20),
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))
    
    # 3. CV vs Model Size
    ax = axes[2]
    ax.plot(model_sizes, cvs, 'o-', linewidth=2.5, markersize=10, color='#2ecc71')
    ax.set_xlabel('Model Size (Billions of Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation', fontsize=12, fontweight='bold')
    ax.set_title('Relative Sensitivity vs Model Scale', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    for i, (size, cv, name) in enumerate(zip(model_sizes, cvs, names)):
        ax.annotate(f'{name}\n{cv:.3f}',
                   xy=(size, cv),
                   xytext=(10, 10 if i % 2 == 0 else -20),
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_model_scale_sensitivity_trend.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / '4_model_scale_sensitivity_trend.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved scale sensitivity trend plot")

def main():
    """Main analysis function"""
    
    print("="*80)
    print("ICML Section 4.1: Four-Model Ordering Instability Analysis")
    print("="*80)
    print()
    
    # Create output directory
    output_dir = Path("icml_analysis/section_4_1_ordering_instability")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all model data
    print("Loading model data...")
    print()
    llama32_df = load_llama32_data()
    print()
    mistral_df = load_mistral_data()
    print()
    qwen_df = load_qwen_data()
    print()
    llama70b_df = load_llama70b_data()
    print()
    
    # Calculate metrics
    print("Calculating ordering instability metrics...")
    metrics = calculate_ordering_instability_metrics(llama32_df, mistral_df, qwen_df, llama70b_df)
    
    # Save metrics
    with open(output_dir / 'four_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nOrdering Instability Summary:")
    print("-" * 80)
    for model, m in metrics.items():
        print(f"\n{model}:")
        print(f"  Orderings: {m['num_orderings']}")
        print(f"  Range: {m['min_approval']:.1%} - {m['max_approval']:.1%} (Δ = {m['range']:.1%})")
        print(f"  Mean ± Std: {m['mean']:.1%} ± {m['std']:.3f}")
        print(f"  CV: {m['cv']:.3f}")
    
    # Generate plots
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    print()
    
    plot_model_comparison(metrics, output_dir)
    plot_violin_distributions(llama32_df, mistral_df, qwen_df, llama70b_df, output_dir)
    plot_scale_sensitivity_trend(metrics, output_dir)
    
    print()
    print("="*80)
    print(f"✓ Analysis complete! Results saved to: {output_dir}")
    print("="*80)
    print()
    print("Key Findings:")
    print("-" * 80)
    
    # Analyze scale trend
    ranges_by_size = [
        (3, metrics['Llama 3.2 (3B)']['range']),
        (7, metrics['Mistral (7B)']['range']),
        (70, metrics['Llama 70B (70B)']['range']),
        (72, metrics['Qwen (72B)']['range'])
    ]
    
    print(f"1. Ordering sensitivity (range) across scales:")
    for size, rng in ranges_by_size:
        print(f"   {size}B: {rng:.1%}")
    
    # Check if sensitivity increases with scale
    if ranges_by_size[-1][1] > ranges_by_size[0][1]:
        print(f"\n2. ✓ Larger models show GREATER ordering sensitivity")
        print(f"   Small model (3B) range: {ranges_by_size[0][1]:.1%}")
        print(f"   Large model (72B) range: {ranges_by_size[-1][1]:.1%}")
    else:
        print(f"\n2. ✗ Larger models show LOWER ordering sensitivity")
        print(f"   This contradicts the hypothesis!")
    
    print()

if __name__ == "__main__":
    main()
