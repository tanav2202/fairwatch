#!/usr/bin/env python3
"""
Section 4.3: Parallel Mode Analysis
Compare parallel vs sequential approval rate variance across models
"""

import sys
sys.path.insert(0, 'plotting')

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from run_small_model_analysis import SmallModelDataLoader

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def calculate_parallel_approval_rate(data, model_type='large'):
    """Calculate approval rate for parallel mode"""
    
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        return None
    
    approvals = 0
    total = 0
    
    for result in results:
        if 'agent_outputs' in result:
            # Count how many agents approved
            agent_outputs = result['agent_outputs']
            
            for agent, output in agent_outputs.items():
                decision = output.get('approval_decision')
                if decision:
                    decision = decision.lower()
                    if decision in ['approve', 'deny']:
                        total += 1
                        if decision == 'approve':
                            approvals += 1
    
    if total == 0:
        return None
    
    return approvals / total

def calculate_sequential_variance(sequential_data_dir, model_type='large', loader=None):
    """Calculate variance in approval rates across sequential orderings"""
    
    approval_rates = []
    
    if model_type == 'large':
        # For 70B model
        for json_file in sorted(sequential_data_dir.glob("sequential_*_formatted.json")):
            with open(json_file) as f:
                data = json.load(f)
            
            results = data.get('results', [])
            approvals = 0
            total = 0
            
            for result in results:
                if 'decisions' in result:
                    for agent, decision_data in result['decisions'].items():
                        dec = decision_data.get('approval_decision', '').lower()
                        if dec in ['approve', 'deny']:
                            total += 1
                            if dec == 'approve':
                                approvals += 1
            
            if total > 0:
                approval_rates.append(approvals / total)
    
    else:
        # For small models (Llama 3.2, Mistral)
        for json_file in sorted(sequential_data_dir.glob("sequential_*.json")):
            data = loader.load_json_file(json_file)
            
            if not data:
                continue
            
            if isinstance(data, dict) and 'results' in data:
                results = data['results']
            elif isinstance(data, list):
                results = data
            else:
                continue
            
            approvals = 0
            total = 0
            
            for result in results:
                if 'conversation_history' in result:
                    for turn in result['conversation_history']:
                        if 'output' in turn and isinstance(turn['output'], dict):
                            dec = turn['output'].get('approval_decision')
                            if dec:
                                dec = dec.lower()
                                if dec in ['approve', 'deny']:
                                    total += 1
                                    if dec == 'approve':
                                        approvals += 1
            
            if total > 0:
                rate = approvals / total
                # Filter out Mistral 0% orderings
                if rate > 0 or not sequential_data_dir.name.startswith('mistral'):
                    approval_rates.append(rate)
    
    if not approval_rates:
        return None, None, None
    
    return np.mean(approval_rates), np.std(approval_rates), approval_rates

def process_llama_70b():
    """Process Llama 70B model"""
    
    print("\n" + "="*80)
    print("Processing Llama 70B")
    print("="*80 + "\n")
    
    # Parallel mode
    with open('70bresults/parallel/parallel_evaluations.json') as f:
        parallel_data = json.load(f)
    
    parallel_rate = calculate_parallel_approval_rate(parallel_data, 'large')
    print(f"Parallel approval rate: {parallel_rate*100:.2f}%")
    
    # Sequential mode variance
    seq_dir = Path("70bresults/sequential")
    seq_mean, seq_std, seq_rates = calculate_sequential_variance(seq_dir, 'large')
    
    print(f"Sequential mean: {seq_mean*100:.2f}%")
    print(f"Sequential std: {seq_std*100:.4f}%")
    print(f"Sequential range: {min(seq_rates)*100:.2f}% - {max(seq_rates)*100:.2f}%")
    
    return {
        'model': 'Llama 70B',
        'parallel_rate': parallel_rate,
        'sequential_mean': seq_mean,
        'sequential_std': seq_std,
        'sequential_rates': seq_rates,
        'sequential_variance': seq_std ** 2
    }

def process_small_model(model_name, model_display_name):
    """Process Llama 3.2 or Mistral model"""
    
    print("\n" + "="*80)
    print(f"Processing {model_display_name}")
    print("="*80 + "\n")
    
    loader = SmallModelDataLoader("outputs_simple", model_name)
    
    # Parallel mode
    parallel_file = Path(f"outputs_simple/{model_name}/parallel/parallel_synthesis_simple.json")
    parallel_data = loader.load_json_file(parallel_file)
    
    parallel_rate = calculate_parallel_approval_rate(parallel_data, 'small')
    print(f"Parallel approval rate: {parallel_rate*100:.2f}%")
    
    # Sequential mode variance
    seq_dir = Path(f"outputs_simple/{model_name}/sequential")
    seq_mean, seq_std, seq_rates = calculate_sequential_variance(seq_dir, 'small', loader)
    
    print(f"Sequential mean: {seq_mean*100:.2f}%")
    print(f"Sequential std: {seq_std*100:.4f}%")
    print(f"Sequential range: {min(seq_rates)*100:.2f}% - {max(seq_rates)*100:.2f}%")
    
    return {
        'model': model_display_name,
        'parallel_rate': parallel_rate,
        'sequential_mean': seq_mean,
        'sequential_std': seq_std,
        'sequential_rates': seq_rates,
        'sequential_variance': seq_std ** 2
    }

def create_comparison_plots(model_results, output_dir):
    """Create comparison plots for parallel vs sequential analysis"""
    
    # Plot 1: Parallel vs Sequential Mean with Error Bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = [r['model'] for r in model_results]
    parallel_rates = [r['parallel_rate'] * 100 for r in model_results]
    sequential_means = [r['sequential_mean'] * 100 for r in model_results]
    sequential_stds = [r['sequential_std'] * 100 for r in model_results]
    
    # Bar plot
    x = np.arange(len(models))
    width = 0.35
    
    ax = axes[0]
    bars1 = ax.bar(x - width/2, parallel_rates, width, label='Parallel', alpha=0.8, color='#2ecc71')
    bars2 = ax.bar(x + width/2, sequential_means, width, yerr=sequential_stds, 
                   label='Sequential (mean ± std)', alpha=0.8, color='#3498db', capsize=5)
    
    ax.set_ylabel('Approval Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Parallel vs Sequential Approval Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Sequential Variance Comparison
    ax = axes[1]
    variances = [r['sequential_std'] * 100 for r in model_results]
    bars = ax.bar(models, variances, alpha=0.8, color='#e74c3c')
    
    ax.set_ylabel('Standard Deviation (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Sequential Mode Approval Rate Variance', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / "parallel_vs_sequential_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    # Also save PDF
    output_file_pdf = output_dir / "parallel_vs_sequential_comparison.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_file_pdf}")
    
    plt.close()
    
    # Plot 3: Distribution of Sequential Approval Rates
    fig, axes = plt.subplots(1, len(model_results), figsize=(5 * len(model_results), 5))
    
    if len(model_results) == 1:
        axes = [axes]
    
    for idx, result in enumerate(model_results):
        ax = axes[idx]
        
        seq_rates_pct = [r * 100 for r in result['sequential_rates']]
        parallel_pct = result['parallel_rate'] * 100
        
        # Histogram
        ax.hist(seq_rates_pct, bins=15, alpha=0.7, color='#3498db', edgecolor='black')
        
        # Add parallel line
        ax.axvline(parallel_pct, color='#2ecc71', linestyle='--', linewidth=2, 
                  label=f'Parallel: {parallel_pct:.1f}%')
        
        # Add mean line
        ax.axvline(result['sequential_mean'] * 100, color='#e74c3c', linestyle='--', 
                  linewidth=2, label=f'Seq Mean: {result["sequential_mean"]*100:.1f}%')
        
        ax.set_xlabel('Approval Rate (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Orderings', fontsize=11, fontweight='bold')
        ax.set_title(f'{result["model"]}\nSequential Approval Rate Distribution', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "sequential_distribution_by_model.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    # Also save PDF
    output_file_pdf = output_dir / "sequential_distribution_by_model.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_file_pdf}")
    
    plt.close()

def main():
    """Run Section 4.3 analysis"""
    
    output_dir = Path("icml_analysis/section_4_3_parallel_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SECTION 4.3: PARALLEL MODE ANALYSIS")
    print("="*80)
    
    model_results = []
    
    # Process each model
    model_results.append(process_llama_70b())
    model_results.append(process_small_model("llama3.2", "Llama 3.2"))
    model_results.append(process_small_model("mistral:latest", "Mistral Latest"))
    
    # Create plots
    print("\n" + "="*80)
    print("Creating comparison plots...")
    print("="*80)
    
    create_comparison_plots(model_results, output_dir)
    
    # Save metrics
    metrics = {
        'models': [{
            'name': r['model'],
            'parallel_approval_rate': float(r['parallel_rate']),
            'sequential_mean_approval_rate': float(r['sequential_mean']),
            'sequential_std': float(r['sequential_std']),
            'sequential_variance': float(r['sequential_variance']),
            'sequential_range': {
                'min': float(min(r['sequential_rates'])),
                'max': float(max(r['sequential_rates']))
            },
            'num_sequential_orderings': len(r['sequential_rates'])
        } for r in model_results]
    }
    
    metrics_file = output_dir / "parallel_analysis_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved metrics: {metrics_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for result in model_results:
        print(f"\n{result['model']}:")
        print(f"  Parallel: {result['parallel_rate']*100:.2f}%")
        print(f"  Sequential: {result['sequential_mean']*100:.2f}% ± {result['sequential_std']*100:.2f}%")
        print(f"  Variance Reduction: {((result['sequential_std'] - 0) / result['sequential_std'] * 100 if result['sequential_std'] > 0 else 0):.1f}% "
              f"(parallel has no ordering variance)")
    
    print("\n" + "="*80)
    print("✓ SECTION 4.3 ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
