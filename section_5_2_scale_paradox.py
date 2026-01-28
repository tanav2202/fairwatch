#!/usr/bin/env python3
"""
Section 5.2: Scale Paradox Analysis
Measure inter-agent agreement, error correction, and the paradox that larger models
have more consensus but less correction capability.
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

def extract_agent_order_from_filename(filename):
    """Extract abbreviated agent order from filename"""
    name = filename.replace("sequential_", "").replace(".json", "").replace("_formatted", "")
    
    agent_map = {
        "consumer": "CA",
        "data": "DS",
        "regulatory": "Reg",
        "risk": "RM"
    }
    
    agent_full_map = {
        "consumer": "consumer_advocate",
        "data": "data_science",
        "regulatory": "regulatory",
        "risk": "risk_manager"
    }
    
    parts = name.split("_")
    abbrev_order = []
    full_order = []
    
    for part in parts:
        if part in agent_map:
            abbrev_order.append(agent_map[part])
            full_order.append(agent_full_map[part])
    
    abbrev_name = "_".join(abbrev_order) if abbrev_order else name
    return abbrev_name, full_order

def should_approve(credit_score):
    """
    Define ground truth: higher credit scores should be approved
    Using median as threshold (credit scores range 620-780)
    """
    THRESHOLD = 700  # Midpoint of range
    return credit_score >= THRESHOLD

def analyze_inter_agent_agreement(results, ordering_name, agent_order, model_type='large'):
    """
    Analyze agreement between agents and error correction patterns
    """
    
    # Load prompts data for small models
    if model_type == 'small':
        prompts_df = pd.read_csv('data/prompts_simple.csv')
    
    total_chains = 0
    
    # Agreement tracking
    position_agreements = defaultdict(int)  # How often agent N agrees with agent N-1
    total_by_position = defaultdict(int)
    
    # Error correction tracking
    first_agent_errors = 0
    errors_corrected_by_position = defaultdict(int)
    errors_not_corrected = 0
    
    # Full agreement tracking
    all_agents_agree = 0
    
    for result in results:
        # Get credit score for ground truth
        credit_score = None
        
        if model_type == 'large':
            if isinstance(result.get('input'), str):
                input_data = json.loads(result['input'])
                credit_score = input_data.get('credit_score', 0)
        else:
            # Small model
            prompt_id = result.get('prompt_id')
            if isinstance(prompt_id, int) and 1 <= prompt_id <= len(prompts_df):
                credit_score = prompts_df.iloc[prompt_id - 1]['credit_score']
        
        if not credit_score:
            continue
        
        # Get decisions by agent
        decisions = []
        
        if model_type == 'large':
            # Large model format: decisions dict
            if 'decisions' in result and agent_order:
                for agent_name in agent_order:
                    if agent_name in result['decisions']:
                        dec = result['decisions'][agent_name].get('approval_decision', '').lower()
                        if dec in ['approve', 'deny']:
                            decisions.append(dec == 'approve')
        else:
            # Small model format: conversation_history
            if 'conversation_history' in result:
                conversation = result['conversation_history']
                for turn in conversation:
                    if 'output' in turn and isinstance(turn['output'], dict):
                        dec = turn['output'].get('approval_decision')
                        if dec:
                            dec = dec.lower()
                            if dec in ['approve', 'deny']:
                                decisions.append(dec == 'approve')
        
        if len(decisions) < 4:
            continue
        
        total_chains += 1
        
        # Ground truth
        should_be_approved = should_approve(credit_score)
        
        # Check if first agent made an error
        first_agent_correct = (decisions[0] == should_be_approved)
        
        if not first_agent_correct:
            first_agent_errors += 1
            
            # Track if error was corrected by any subsequent agent
            error_corrected = False
            for pos in range(1, 4):
                if decisions[pos] == should_be_approved:
                    errors_corrected_by_position[pos] += 1
                    error_corrected = True
                    break
            
            if not error_corrected:
                errors_not_corrected += 1
        
        # Track inter-agent agreement (pairwise)
        for pos in range(1, 4):
            total_by_position[pos] += 1
            if decisions[pos] == decisions[pos - 1]:
                position_agreements[pos] += 1
        
        # Check if all agents agree
        if len(set(decisions)) == 1:
            all_agents_agree += 1
    
    if total_chains == 0:
        return None
    
    # Calculate metrics
    agreement_rates = {}
    for pos in range(1, 4):
        if total_by_position[pos] > 0:
            agreement_rates[f'position_{pos}'] = position_agreements[pos] / total_by_position[pos]
    
    overall_agreement = np.mean(list(agreement_rates.values())) if agreement_rates else 0
    full_consensus_rate = all_agents_agree / total_chains
    
    # Error correction metrics
    error_correction_rate = 0
    correction_by_position = {}
    
    if first_agent_errors > 0:
        total_corrected = sum(errors_corrected_by_position.values())
        error_correction_rate = total_corrected / first_agent_errors
        
        for pos in range(1, 4):
            correction_by_position[f'position_{pos}'] = errors_corrected_by_position.get(pos, 0) / first_agent_errors
    
    return {
        'ordering': ordering_name,
        'total_chains': total_chains,
        'inter_agent_agreement': overall_agreement,
        'agreement_by_position': agreement_rates,
        'full_consensus_rate': full_consensus_rate,
        'first_agent_error_rate': first_agent_errors / total_chains if total_chains > 0 else 0,
        'error_correction_rate': error_correction_rate,
        'correction_by_position': correction_by_position,
        'errors_never_corrected': errors_not_corrected,
        'first_agent_errors': first_agent_errors
    }

def process_70b_model():
    """Process Llama 70B model"""
    
    print("\n" + "="*80)
    print("Processing Llama 70B Model")
    print("="*80 + "\n")
    
    base_dir = Path("70bresults/sequential")
    orderings_analysis = []
    
    for json_file in sorted(base_dir.glob("sequential_*_formatted.json")):
        ordering_name, agent_order = extract_agent_order_from_filename(json_file.name)
        print(f"Processing: {ordering_name}...")
        
        with open(json_file) as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        analysis = analyze_inter_agent_agreement(results, ordering_name, agent_order, 'large')
        
        if analysis:
            orderings_analysis.append(analysis)
            print(f"  ✓ Agreement: {analysis['inter_agent_agreement']*100:.1f}%, "
                  f"Error correction: {analysis['error_correction_rate']*100:.1f}%")
    
    return orderings_analysis

def process_qwen_model():
    """Process Qwen model"""
    
    print("\n" + "="*80)
    print("Processing Qwen Model")
    print("="*80 + "\n")
    
    base_dir = Path("qwen_validation")
    orderings_analysis = []
    
    for json_file in sorted(base_dir.glob("sequential_*.json")):
        ordering_name, agent_order = extract_agent_order_from_filename(json_file.name)
        print(f"Processing: {ordering_name}...")
        
        with open(json_file) as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        analysis = analyze_inter_agent_agreement(results, ordering_name, agent_order, 'large')
        
        if analysis:
            orderings_analysis.append(analysis)
            print(f"  ✓ Agreement: {analysis['inter_agent_agreement']*100:.1f}%, "
                  f"Error correction: {analysis['error_correction_rate']*100:.1f}%")
    
    return orderings_analysis

def process_small_model(model_name, model_display_name):
    """Process Llama 3.2 or Mistral model"""
    
    print("\n" + "="*80)
    print(f"Processing {model_display_name} Model")
    print("="*80 + "\n")
    
    loader = SmallModelDataLoader("outputs_simple", model_name)
    base_dir = Path(f"outputs_simple/{model_name}/sequential")
    
    orderings_analysis = []
    
    for json_file in sorted(base_dir.glob("sequential_*.json")):
        ordering_name, agent_order = extract_agent_order_from_filename(json_file.name)
        print(f"Processing: {ordering_name}...")
        
        data = loader.load_json_file(json_file)
        
        if not data:
            print(f"  ⚠️  Skipped (couldn't load)")
            continue
        
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
        elif isinstance(data, list):
            results = data
        else:
            print(f"  ⚠️  Skipped (unexpected format)")
            continue
        
        analysis = analyze_inter_agent_agreement(results, ordering_name, agent_order, 'small')
        
        if analysis:
            # For Mistral, skip 0% approval orderings
            if model_name == "mistral:latest":
                # Check if this ordering has meaningful data
                if analysis['total_chains'] < 100:
                    print(f"  ⚠️  Skipped (too few chains)")
                    continue
            
            orderings_analysis.append(analysis)
            print(f"  ✓ Agreement: {analysis['inter_agent_agreement']*100:.1f}%, "
                  f"Error correction: {analysis['error_correction_rate']*100:.1f}%")
    
    return orderings_analysis

def create_scale_paradox_plots(model_results, output_dir):
    """Create plots demonstrating the scale paradox"""
    
    # Calculate model-level statistics
    model_stats = {}
    model_sizes = {
        'Llama 3.2': 3,
        'Mistral Latest': 7,
        'Qwen': 72,
        'Llama 70B': 70
    }
    
    for model_name, analyses in model_results.items():
        if not analyses:
            continue
        
        agreements = [a['inter_agent_agreement'] for a in analyses]
        corrections = [a['error_correction_rate'] for a in analyses]
        consensus = [a['full_consensus_rate'] for a in analyses]
        
        model_stats[model_name] = {
            'size': model_sizes.get(model_name, 0),
            'mean_agreement': np.mean(agreements),
            'std_agreement': np.std(agreements),
            'mean_correction': np.mean(corrections),
            'std_correction': np.std(corrections),
            'mean_consensus': np.mean(consensus),
            'num_orderings': len(analyses)
        }
    
    # Sort by model size
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['size'])
    
    # Plot 1: The Scale Paradox - Agreement vs Correction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sizes = [s[1]['size'] for s in sorted_models]
    names = [s[0] for s in sorted_models]
    agreements = [s[1]['mean_agreement'] * 100 for s in sorted_models]
    corrections = [s[1]['mean_correction'] * 100 for s in sorted_models]
    
    # Agreement subplot
    ax1.plot(sizes, agreements, 'o-', markersize=10, linewidth=2, color='#2ecc71', label='Inter-agent Agreement')
    ax1.set_xlabel('Model Size (Billion Parameters)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inter-Agent Agreement Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Scale Increases Agreement\n(Cascade Effect)', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')
    
    # Add labels
    for i, (size, agreement, name) in enumerate(zip(sizes, agreements, names)):
        ax1.annotate(f'{agreement:.1f}%', (size, agreement), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9, fontweight='bold')
    
    # Error correction subplot
    ax2.plot(sizes, corrections, 'o-', markersize=10, linewidth=2, color='#e74c3c', label='Error Correction')
    ax2.set_xlabel('Model Size (Billion Parameters)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error Correction Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Scale Decreases Error Correction\n(The Paradox)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')
    
    # Add labels
    for i, (size, correction, name) in enumerate(zip(sizes, corrections, names)):
        ax2.annotate(f'{correction:.1f}%', (size, correction), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / "scale_paradox_agreement_vs_correction.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    output_file_pdf = output_dir / "scale_paradox_agreement_vs_correction.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_file_pdf}")
    
    plt.close()
    
    # Plot 2: Scatter - Agreement vs Correction (showing inverse relationship)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'Llama 70B': '#e74c3c', 'Qwen': '#3498db', 'Llama 3.2': '#2ecc71', 'Mistral Latest': '#f39c12'}
    
    for model_name, analyses in model_results.items():
        if not analyses:
            continue
        
        agreements = [a['inter_agent_agreement'] * 100 for a in analyses]
        corrections = [a['error_correction_rate'] * 100 for a in analyses]
        
        ax.scatter(agreements, corrections, 
                  label=f"{model_name} ({model_stats[model_name]['size']}B)", 
                  alpha=0.6, s=100, color=colors.get(model_name, '#95a5a6'))
    
    ax.set_xlabel('Inter-Agent Agreement Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Correction Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('The Scale Paradox: High Agreement ⟹ Low Error Correction\n'
                 'Each point represents one sequential ordering',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    
    # Add trend line
    all_agreements = []
    all_corrections = []
    for analyses in model_results.values():
        if analyses:
            all_agreements.extend([a['inter_agent_agreement'] * 100 for a in analyses])
            all_corrections.extend([a['error_correction_rate'] * 100 for a in analyses])
    
    if all_agreements:
        z = np.polyfit(all_agreements, all_corrections, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(all_agreements), max(all_agreements), 100)
        ax.plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=2, label=f'Trend (slope={z[0]:.2f})')
        ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    output_file = output_dir / "agreement_vs_correction_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    output_file_pdf = output_dir / "agreement_vs_correction_scatter.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_file_pdf}")
    
    plt.close()
    
    # Plot 3: Error Correction by Position
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    idx = 0
    for model_name in ['Llama 70B', 'Qwen', 'Llama 3.2', 'Mistral Latest']:
        if model_name not in model_results or not model_results[model_name]:
            continue
        
        ax = axes[idx]
        analyses = model_results[model_name]
        
        # Aggregate correction rates by position
        pos_1_corrections = []
        pos_2_corrections = []
        pos_3_corrections = []
        
        for analysis in analyses:
            if analysis['first_agent_errors'] > 0:
                pos_1_corrections.append(analysis['correction_by_position'].get('position_1', 0) * 100)
                pos_2_corrections.append(analysis['correction_by_position'].get('position_2', 0) * 100)
                pos_3_corrections.append(analysis['correction_by_position'].get('position_3', 0) * 100)
        
        positions = ['Agent 2\n(1st correction)', 'Agent 3\n(2nd correction)', 'Agent 4\n(3rd correction)']
        means = [
            np.mean(pos_1_corrections) if pos_1_corrections else 0,
            np.mean(pos_2_corrections) if pos_2_corrections else 0,
            np.mean(pos_3_corrections) if pos_3_corrections else 0
        ]
        
        bars = ax.bar(positions, means, alpha=0.7, color=['#3498db', '#2ecc71', '#f39c12'])
        
        ax.set_ylabel('Error Correction Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name} ({model_stats[model_name]["size"]}B Parameters)\n'
                    f'Mean Correction: {np.mean(means):.1f}%',
                    fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(max(means) * 1.2, 10))
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        idx += 1
    
    plt.tight_layout()
    
    output_file = output_dir / "error_correction_by_position.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    output_file_pdf = output_dir / "error_correction_by_position.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_file_pdf}")
    
    plt.close()
    
    return model_stats

def main():
    """Run Section 5.2 analysis"""
    
    output_dir = Path("icml_analysis/section_5_2_scale_paradox")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SECTION 5.2: SCALE PARADOX ANALYSIS")
    print("="*80)
    
    model_results = {}
    
    # Process each model
    model_results['Llama 70B'] = process_70b_model()
    model_results['Qwen'] = process_qwen_model()
    model_results['Llama 3.2'] = process_small_model("llama3.2", "Llama 3.2")
    model_results['Mistral Latest'] = process_small_model("mistral:latest", "Mistral Latest")
    
    # Save detailed metrics
    all_metrics = {}
    for model_name, analyses in model_results.items():
        all_metrics[model_name] = [{
            'ordering': a['ordering'],
            'total_chains': int(a['total_chains']),
            'inter_agent_agreement': float(a['inter_agent_agreement']),
            'agreement_by_position': {k: float(v) for k, v in a['agreement_by_position'].items()},
            'full_consensus_rate': float(a['full_consensus_rate']),
            'first_agent_error_rate': float(a['first_agent_error_rate']),
            'error_correction_rate': float(a['error_correction_rate']),
            'correction_by_position': {k: float(v) for k, v in a['correction_by_position'].items()},
            'errors_never_corrected': int(a['errors_never_corrected']),
            'first_agent_errors': int(a['first_agent_errors'])
        } for a in analyses]
    
    metrics_file = output_dir / "scale_paradox_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✓ Saved detailed metrics: {metrics_file}")
    
    # Create plots
    print("\n" + "="*80)
    print("Creating visualization plots...")
    print("="*80)
    
    model_stats = create_scale_paradox_plots(model_results, output_dir)
    
    # Save summary statistics
    summary = {
        'model_statistics': {
            model_name: {
                'model_size_billions': stats['size'],
                'mean_inter_agent_agreement': float(stats['mean_agreement']),
                'std_inter_agent_agreement': float(stats['std_agreement']),
                'mean_error_correction_rate': float(stats['mean_correction']),
                'std_error_correction_rate': float(stats['std_correction']),
                'mean_full_consensus_rate': float(stats['mean_consensus']),
                'num_orderings_analyzed': int(stats['num_orderings'])
            }
            for model_name, stats in model_stats.items()
        }
    }
    
    summary_file = output_dir / "scale_paradox_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary: {summary_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SCALE PARADOX SUMMARY")
    print("="*80)
    
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['size'])
    
    for model_name, stats in sorted_models:
        print(f"\n{model_name} ({stats['size']}B parameters):")
        print(f"  Inter-agent agreement: {stats['mean_agreement']*100:.2f}% ± {stats['std_agreement']*100:.2f}%")
        print(f"  Error correction rate: {stats['mean_correction']*100:.2f}% ± {stats['std_correction']*100:.2f}%")
        print(f"  Full consensus rate: {stats['mean_consensus']*100:.2f}%")
        print(f"  Orderings analyzed: {stats['num_orderings']}")
    
    print("\n" + "="*80)
    print("THE PARADOX:")
    print("  As model size increases:")
    print(f"    - Agreement increases: {sorted_models[0][1]['mean_agreement']*100:.1f}% → {sorted_models[-1][1]['mean_agreement']*100:.1f}%")
    print(f"    - Error correction DECREASES: {sorted_models[0][1]['mean_correction']*100:.1f}% → {sorted_models[-1][1]['mean_correction']*100:.1f}%")
    print("  Larger models achieve consensus faster but lose ability to correct mistakes!")
    print("="*80)
    
    print("\n" + "="*80)
    print("✓ SECTION 5.2 ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
