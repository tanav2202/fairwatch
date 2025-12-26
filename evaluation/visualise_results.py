#!/usr/bin/env python3
"""
Comprehensive Visualization Suite
- Separates different agent orderings
- Shows agent-level distributions
- Per-bias-type analysis
- Granular, detailed plots (not aggregated)

Usage:
    python visualize_comprehensive.py baseline.json chains.json
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def load_data(baseline_file, chain_file):
    """Load baseline and chain data"""
    with open(baseline_file) as f:
        baseline = json.load(f)
    
    with open(chain_file) as f:
        chains = json.load(f)
    
    return baseline, chains


def extract_baseline_detailed(data):
    """Extract baseline with full detail"""
    agent_scores = defaultdict(list)
    bias_type_scores = defaultdict(list)
    agent_bias_scores = defaultdict(lambda: defaultdict(list))  # agent -> bias_type -> scores
    
    for agent_name, results in data['results'].items():
        for result in results:
            for bias_eval in result['bias_evaluations']:
                score = bias_eval['score']
                bias_type = bias_eval['bias_type']
                
                agent_scores[agent_name].append(score)
                bias_type_scores[bias_type].append(score)
                agent_bias_scores[agent_name][bias_type].append(score)
    
    return agent_scores, bias_type_scores, agent_bias_scores


def extract_chain_detailed(data):
    """Extract chain data with full granularity"""
    orderings = {}
    
    for result in data['results']:
        ordering_key = tuple(result['agent_sequence'])
        ordering_name = ' → '.join(result['agent_sequence'])
        
        if ordering_key not in orderings:
            orderings[ordering_key] = {
                'name': ordering_name,
                'all_scores': [],
                'agent_scores': defaultdict(list),
                'turn_scores': defaultdict(list),
                'bias_type_scores': defaultdict(list),
                'agent_bias_scores': defaultdict(lambda: defaultdict(list)),
                'turn_bias_scores': defaultdict(lambda: defaultdict(list)),
            }
        
        for turn_eval in result['turn_bias_evaluations']:
            turn_num = turn_eval['turn_number']
            agent_name = turn_eval['agent_name']
            
            for bias_eval in turn_eval['bias_evaluations']:
                score = bias_eval['score']
                bias_type = bias_eval['bias_type']
                
                orderings[ordering_key]['all_scores'].append(score)
                orderings[ordering_key]['agent_scores'][agent_name].append(score)
                orderings[ordering_key]['turn_scores'][turn_num].append(score)
                orderings[ordering_key]['bias_type_scores'][bias_type].append(score)
                orderings[ordering_key]['agent_bias_scores'][agent_name][bias_type].append(score)
                orderings[ordering_key]['turn_bias_scores'][turn_num][bias_type].append(score)
    
    return orderings


def plot_per_bias_type_distributions(baseline_bias, chain_orderings, output_dir):
    """Plot 1: Separate distribution for EACH bias type"""
    all_bias_types = sorted(set(baseline_bias.keys()))
    n_bias = len(all_bias_types)
    
    fig, axes = plt.subplots(n_bias, 1, figsize=(14, 4*n_bias))
    if n_bias == 1:
        axes = [axes]
    
    for idx, bias_type in enumerate(all_bias_types):
        ax = axes[idx]
        
        # Baseline
        baseline_data = baseline_bias.get(bias_type, [])
        
        # Each ordering
        ordering_keys = list(chain_orderings.keys())
        
        data_to_plot = [baseline_data]
        labels = ['Baseline']
        colors = ['steelblue']
        
        for ord_idx, ordering_key in enumerate(ordering_keys):
            chain_data = chain_orderings[ordering_key]['bias_type_scores'].get(bias_type, [])
            data_to_plot.append(chain_data)
            labels.append(f"Chain Ord{ord_idx+1}")
            colors.append(['coral', 'mediumpurple'][ord_idx])
        
        # Violin plot
        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                              widths=0.7, showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Bias Score', fontsize=11)
        ax.set_title(f'{bias_type.replace("_", " ").title()} - Distribution', 
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-0.5, 10.5)
        
        # Statistics
        stats_text = ""
        for data, label in zip(data_to_plot, labels):
            if len(data) > 0:
                stats_text += f"{label}: μ={np.mean(data):.2f}, σ={np.std(data):.2f}, n={len(data)}\n"
        
        ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_bias_type_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: per_bias_type_distributions.png")
    plt.close()


def plot_agent_bias_heatmap(baseline_agent_bias, chain_orderings, output_dir):
    """Plot 2: Heatmap of Agent × Bias Type for each condition"""
    ordering_keys = list(chain_orderings.keys())
    n_orderings = len(ordering_keys)
    
    fig, axes = plt.subplots(1, n_orderings + 1, figsize=(6*(n_orderings+1), 8))
    
    all_agents = sorted(set(baseline_agent_bias.keys()))
    all_bias_types = sorted(set(bt for agent_data in baseline_agent_bias.values() 
                                for bt in agent_data.keys()))
    
    # Baseline heatmap
    ax = axes[0]
    baseline_matrix = np.zeros((len(all_agents), len(all_bias_types)))
    for i, agent in enumerate(all_agents):
        for j, bias_type in enumerate(all_bias_types):
            scores = baseline_agent_bias[agent].get(bias_type, [])
            baseline_matrix[i, j] = np.mean(scores) if scores else 0
    
    im = ax.imshow(baseline_matrix, cmap='RdYlGn_r', vmin=0, vmax=10, aspect='auto')
    ax.set_xticks(range(len(all_bias_types)))
    ax.set_xticklabels([bt.replace('_', '\n') for bt in all_bias_types], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(all_agents)))
    ax.set_yticklabels(all_agents, fontsize=10)
    ax.set_title('Baseline\n(Individual Agents)', fontsize=11, fontweight='bold')
    
    # Add values
    for i in range(len(all_agents)):
        for j in range(len(all_bias_types)):
            ax.text(j, i, f'{baseline_matrix[i, j]:.1f}', 
                   ha='center', va='center', fontsize=8)
    
    # Chain heatmaps
    for ord_idx, ordering_key in enumerate(ordering_keys):
        ax = axes[ord_idx + 1]
        chain_matrix = np.zeros((len(all_agents), len(all_bias_types)))
        
        for i, agent in enumerate(all_agents):
            for j, bias_type in enumerate(all_bias_types):
                scores = chain_orderings[ordering_key]['agent_bias_scores'][agent].get(bias_type, [])
                chain_matrix[i, j] = np.mean(scores) if scores else 0
        
        im = ax.imshow(chain_matrix, cmap='RdYlGn_r', vmin=0, vmax=10, aspect='auto')
        ax.set_xticks(range(len(all_bias_types)))
        ax.set_xticklabels([bt.replace('_', '\n') for bt in all_bias_types], 
                           rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(all_agents)))
        ax.set_yticklabels(all_agents, fontsize=10)
        ax.set_title(f'Chain Ordering {ord_idx+1}\n{chain_orderings[ordering_key]["name"][:40]}...', 
                     fontsize=11, fontweight='bold')
        
        # Add values
        for i in range(len(all_agents)):
            for j in range(len(all_bias_types)):
                ax.text(j, i, f'{chain_matrix[i, j]:.1f}', 
                       ha='center', va='center', fontsize=8)
    
    # Colorbar - positioned below the plots instead of on top
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Make room at bottom
    fig.colorbar(im, ax=axes, orientation='horizontal', 
                 pad=0.15, fraction=0.03, label='Mean Bias Score',
                 shrink=0.8)
    
    plt.savefig(output_dir / 'agent_bias_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: agent_bias_heatmap.png")
    plt.close()


def plot_ordering_comparison_detailed(baseline_agents, chain_orderings, output_dir):
    """Plot 3: Side-by-side agent comparison for each ordering"""
    ordering_keys = list(chain_orderings.keys())
    n_orderings = len(ordering_keys)
    
    fig, axes = plt.subplots(1, n_orderings, figsize=(8*n_orderings, 6))
    if n_orderings == 1:
        axes = [axes]
    
    for ord_idx, ordering_key in enumerate(ordering_keys):
        ax = axes[ord_idx]
        agents = list(ordering_key)
        
        baseline_means = [np.mean(baseline_agents[a]) for a in agents]
        chain_means = [np.mean(chain_orderings[ordering_key]['agent_scores'][a]) for a in agents]
        
        x = np.arange(len(agents))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_means, width, label='Baseline', 
                       color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, chain_means, width, label='Chain', 
                       color=['coral', 'mediumpurple'][ord_idx], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, (b, c) in enumerate(zip(baseline_means, chain_means)):
            ax.text(i - width/2, b + 0.1, f'{b:.2f}', ha='center', fontsize=9)
            ax.text(i + width/2, c + 0.1, f'{c:.2f}', ha='center', fontsize=9)
            
            delta = c - b
            color = 'green' if delta < 0 else 'red'
            ax.text(i, max(b, c) + 0.5, f'Δ{delta:+.2f}', ha='center', 
                    fontsize=10, color=color, fontweight='bold')
        
        ax.set_xlabel('Agent', fontsize=12)
        ax.set_ylabel('Mean Bias Score', fontsize=12)
        ax.set_title(f'Ordering {ord_idx+1}: {" → ".join(agents)}', 
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ordering_detailed_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: ordering_detailed_comparison.png")
    plt.close()


def plot_agent_distributions_per_ordering(baseline_agents, chain_orderings, output_dir):
    """Plot 4: Full distributions for each agent in each ordering"""
    all_agents = sorted(set(baseline_agents.keys()))
    ordering_keys = list(chain_orderings.keys())
    
    fig, axes = plt.subplots(len(all_agents), 1, figsize=(14, 4*len(all_agents)))
    if len(all_agents) == 1:
        axes = [axes]
    
    for agent_idx, agent in enumerate(all_agents):
        ax = axes[agent_idx]
        
        data_to_plot = []
        labels = []
        colors_list = []
        
        # Baseline
        data_to_plot.append(baseline_agents[agent])
        labels.append('Baseline')
        colors_list.append('steelblue')
        
        # Each ordering
        for ord_idx, ordering_key in enumerate(ordering_keys):
            if agent in chain_orderings[ordering_key]['agent_scores']:
                data_to_plot.append(chain_orderings[ordering_key]['agent_scores'][agent])
                labels.append(f'Ord{ord_idx+1}')
                colors_list.append(['coral', 'mediumpurple'][ord_idx])
        
        # Violin plot instead of box plot
        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                              widths=0.7, showmeans=True, showmedians=True, showextrema=True)
        
        # Color violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Bias Score', fontsize=11)
        ax.set_title(f'{agent} - Full Distribution Comparison', 
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-0.5, 10.5)
        
        # Stats
        stats_text = ""
        for data, label in zip(data_to_plot, labels):
            stats_text += f"{label}: μ={np.mean(data):.2f}, med={np.median(data):.2f}, σ={np.std(data):.2f}, n={len(data)}\n"
        
        ax.text(0.98, 0.98, stats_text.strip(), transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'agent_full_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: agent_full_distributions.png")
    plt.close()


def plot_turn_bias_evolution(chain_orderings, output_dir):
    """Plot 5: Turn-by-turn evolution for EACH bias type, separated by ordering"""
    ordering_keys = list(chain_orderings.keys())
    all_bias_types = sorted(set(bt for ord_data in chain_orderings.values() 
                                for bt in ord_data['turn_bias_scores'][1].keys()))
    
    n_bias = len(all_bias_types)
    n_orderings = len(ordering_keys)
    
    fig, axes = plt.subplots(n_bias, n_orderings, figsize=(7*n_orderings, 4*n_bias))
    if n_bias == 1 and n_orderings == 1:
        axes = [[axes]]
    elif n_bias == 1:
        axes = [axes]
    elif n_orderings == 1:
        axes = [[ax] for ax in axes]
    
    colors = ['coral', 'mediumpurple']
    
    for bias_idx, bias_type in enumerate(all_bias_types):
        for ord_idx, ordering_key in enumerate(ordering_keys):
            ax = axes[bias_idx][ord_idx]
            
            turn_bias_data = chain_orderings[ordering_key]['turn_bias_scores']
            turns = sorted(turn_bias_data.keys())
            
            means = []
            stds = []
            for turn in turns:
                scores = turn_bias_data[turn].get(bias_type, [])
                if scores:
                    means.append(np.mean(scores))
                    stds.append(np.std(scores))
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.plot(turns, means, marker='o', linewidth=2, markersize=8,
                   color=colors[ord_idx], label=f'{bias_type}')
            ax.fill_between(turns,
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.3, color=colors[ord_idx])
            
            # Annotate
            for t, m in zip(turns, means):
                if m > 0:
                    ax.text(t, m + 0.2, f'{m:.1f}', ha='center', fontsize=8)
            
            ax.set_xlabel('Turn Number', fontsize=10)
            ax.set_ylabel('Mean Bias Score', fontsize=10)
            ax.set_title(f'{bias_type.replace("_", " ").title()}\nOrdering {ord_idx+1}', 
                        fontsize=11, fontweight='bold')
            ax.set_xticks(turns)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'turn_bias_evolution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: turn_bias_evolution.png")
    plt.close()


def plot_overall_comparison(baseline_scores, chain_orderings, output_dir):
    """Plot 6: Overall distribution comparison"""
    ordering_keys = list(chain_orderings.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bins = np.arange(0, 10.5, 0.5)
    
    ax.hist(baseline_scores, bins=bins, alpha=0.6, label='Baseline', 
            color='steelblue', density=True, edgecolor='black', linewidth=0.5)
    
    colors = ['coral', 'mediumpurple']
    for ord_idx, ordering_key in enumerate(ordering_keys):
        chain_scores = chain_orderings[ordering_key]['all_scores']
        ax.hist(chain_scores, bins=bins, alpha=0.6, 
                label=f'Chain Ord{ord_idx+1}', 
                color=colors[ord_idx], density=True, edgecolor='black', linewidth=0.5)
    
    # Medians
    ax.axvline(np.median(baseline_scores), color='steelblue', linestyle='--', 
               linewidth=2, label=f'Baseline Med: {np.median(baseline_scores):.2f}')
    
    for ord_idx, ordering_key in enumerate(ordering_keys):
        chain_scores = chain_orderings[ordering_key]['all_scores']
        ax.axvline(np.median(chain_scores), color=colors[ord_idx], linestyle='--', 
                   linewidth=2, label=f'Ord{ord_idx+1} Med: {np.median(chain_scores):.2f}')
    
    ax.set_xlabel('Bias Score', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Overall Distribution: Baseline vs Chain Orderings', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: overall_distribution.png")
    plt.close()


def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_comprehensive.py <baseline.json> <chain.json>")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    chain_file = sys.argv[2]
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VISUALIZATION SUITE")
    print("="*80)
    
    # Load
    print("\n[1/10] Loading data...")
    baseline_data, chain_data = load_data(baseline_file, chain_file)
    print(f"  ✓ Baseline: {baseline_data['metadata']['num_prompts_completed']} prompts")
    print(f"  ✓ Chains: {chain_data['metadata']['num_chains_completed']} chains")
    
    # Extract
    print("\n[2/10] Extracting detailed scores...")
    baseline_agents, baseline_bias, baseline_agent_bias = extract_baseline_detailed(baseline_data)
    chain_orderings = extract_chain_detailed(chain_data)
    
    all_baseline = [s for scores in baseline_agents.values() for s in scores]
    print(f"  ✓ Baseline: {len(all_baseline)} evaluations")
    print(f"  ✓ Chains: {len(chain_orderings)} orderings detected")
    
    for ord_key, ord_data in chain_orderings.items():
        print(f"    - {ord_data['name']}: {len(ord_data['all_scores'])} evaluations")
    
    # Output dir
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    print(f"\n[3/10] Output directory: {output_dir}/")
    
    # Generate plots
    print("\n[4/10] Per-bias-type distributions...")
    plot_per_bias_type_distributions(baseline_bias, chain_orderings, output_dir)
    
    print("\n[5/10] Agent × Bias heatmaps...")
    plot_agent_bias_heatmap(baseline_agent_bias, chain_orderings, output_dir)
    
    print("\n[6/10] Ordering comparison (detailed)...")
    plot_ordering_comparison_detailed(baseline_agents, chain_orderings, output_dir)
    
    print("\n[7/10] Agent full distributions...")
    plot_agent_distributions_per_ordering(baseline_agents, chain_orderings, output_dir)
    
    print("\n[8/10] Turn-by-turn bias evolution...")
    plot_turn_bias_evolution(chain_orderings, output_dir)
    
    print("\n[9/10] Overall distribution comparison...")
    plot_overall_comparison(all_baseline, chain_orderings, output_dir)
    
    # Summary
    print("\n[10/10] Summary statistics...")
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nBaseline:")
    print(f"  Mean: {np.mean(all_baseline):.2f}")
    print(f"  Median: {np.median(all_baseline):.2f}")
    print(f"  Std: {np.std(all_baseline):.2f}")
    
    for ord_idx, (ord_key, ord_data) in enumerate(chain_orderings.items()):
        chain_scores = ord_data['all_scores']
        print(f"\nOrdering {ord_idx+1}: {ord_data['name']}")
        print(f"  Mean: {np.mean(chain_scores):.2f} (Δ{np.mean(chain_scores) - np.mean(all_baseline):+.2f})")
        print(f"  Median: {np.median(chain_scores):.2f}")
        print(f"  Std: {np.std(chain_scores):.2f}")
    
    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGenerated plots:")
    print("  1. per_bias_type_distributions.png - Each bias type separately")
    print("  2. agent_bias_heatmap.png - Agent × Bias matrix")
    print("  3. ordering_detailed_comparison.png - Side-by-side orderings")
    print("  4. agent_full_distributions.png - Box plots per agent")
    print("  5. turn_bias_evolution.png - Turn evolution for each bias type")
    print("  6. overall_distribution.png - Overall comparison")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()