#!/usr/bin/env python3
"""
Section 4.4: Functional Collapse Analysis
Calculate fairness metrics by ethnic group and credit score quartile
Find orderings where demographic parity is satisfied but credit score correlation is weak
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
from scipy.stats import spearmanr, chi2_contingency

from run_small_model_analysis import SmallModelDataLoader

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load demographic data from prompts CSV
PROMPTS_DATA = None
PROMPTS_DF = None

def load_prompts_data():
    """Load demographic data from prompts CSV"""
    global PROMPTS_DATA, PROMPTS_DF
    if PROMPTS_DF is None:
        PROMPTS_DF = pd.read_csv('data/prompts_simple.csv')
        # Also create dict indexed by the hex prompt_id
        PROMPTS_DATA = PROMPTS_DF.set_index('prompt_id').to_dict('index')
    return PROMPTS_DF, PROMPTS_DATA

def extract_agent_order_from_filename(filename):
    """Extract abbreviated agent order from filename"""
    name = filename.replace("sequential_", "").replace(".json", "").replace("_formatted", "")
    
    agent_map = {
        "consumer": "CA",
        "data": "DS",
        "regulatory": "Reg",
        "risk": "RM"
    }
    
    parts = name.split("_")
    order = []
    for part in parts:
        if part in agent_map:
            order.append(agent_map[part])
    
    return "_".join(order) if order else name

def get_final_decision(result, model_type='large'):
    """Extract final approval decision from a result"""
    
    if model_type == 'large':
        # 70B/Qwen format: decisions dict
        if 'decisions' in result:
            # Get last agent's decision (4th agent in ordering)
            # We need to determine the ordering, but for simplicity, look at all decisions
            decisions = result['decisions']
            # Get any decision (they should all be the same if cascade is working)
            for agent, dec in decisions.items():
                return dec.get('approval_decision', '').lower()
    else:
        # Small model format: conversation_history
        if 'conversation_history' in result:
            conversation = result['conversation_history']
            if len(conversation) >= 4:
                final = conversation[3]  # 4th agent
                if 'output' in final and isinstance(final['output'], dict):
                    decision = final['output'].get('approval_decision')
                    if decision:
                        return decision.lower()
    
    return None

def analyze_ordering_fairness(results, ordering_name, model_type='large'):
    """Analyze fairness metrics for a single ordering"""
    
    # Load prompts data for small models
    prompts_df, prompts_dict = load_prompts_data() if model_type == 'small' else (None, None)
    
    # Collect data by ethnicity and credit score
    ethnic_data = defaultdict(lambda: {'approved': 0, 'total': 0})
    credit_score_data = []
    
    for result in results:
        # Parse input
        input_data = None
        
        if isinstance(result.get('input'), str):
            input_data = json.loads(result['input'])
        elif isinstance(result.get('input'), dict):
            input_data = result['input']
        elif model_type == 'small' and 'prompt_id' in result and prompts_df is not None:
            # For small models, prompt_id is 1-indexed integer corresponding to row number
            prompt_id = result.get('prompt_id')
            if isinstance(prompt_id, int) and 1 <= prompt_id <= len(prompts_df):
                # Convert to 0-indexed for iloc
                input_data = prompts_df.iloc[prompt_id - 1].to_dict()
        
        if not input_data:
            continue
        
        ethnicity = input_data.get('ethnicity_signal', '')
        credit_score = input_data.get('credit_score', 0)
        
        # Get final decision
        final_decision = get_final_decision(result, model_type)
        
        if final_decision in ['approve', 'deny'] and ethnicity and credit_score:
            is_approved = (final_decision == 'approve')
            
            # Track by ethnicity
            ethnic_data[ethnicity]['total'] += 1
            if is_approved:
                ethnic_data[ethnicity]['approved'] += 1
            
            # Track for credit score correlation
            credit_score_data.append({
                'credit_score': credit_score,
                'approved': is_approved
            })
    
    if not credit_score_data:
        return None
    
    # Calculate demographic parity (max difference in approval rates across groups)
    approval_rates_by_ethnicity = {}
    for ethnicity, stats in ethnic_data.items():
        if stats['total'] > 0:
            approval_rates_by_ethnicity[ethnicity] = stats['approved'] / stats['total']
    
    if len(approval_rates_by_ethnicity) < 2:
        return None
    
    max_disparity = max(approval_rates_by_ethnicity.values()) - min(approval_rates_by_ethnicity.values())
    
    # Calculate credit score correlation
    df = pd.DataFrame(credit_score_data)
    
    # Calculate quartiles - handle cases with few unique values
    try:
        df['quartile'] = pd.qcut(df['credit_score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        # Approval rate by quartile
        quartile_approval = df.groupby('quartile')['approved'].mean().to_dict()
        
        # Chi-square test for independence
        contingency_table = pd.crosstab(df['quartile'], df['approved'])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2, chi2_p_value, dof, expected = chi2_contingency(contingency_table)
        else:
            chi2, chi2_p_value = 0, 1
    except (ValueError, KeyError):
        # If we can't create quartiles, just skip this metric
        quartile_approval = {}
        chi2, chi2_p_value = 0, 1
    
    # Spearman correlation between credit score and approval
    if len(df) > 1:
        corr, p_value = spearmanr(df['credit_score'], df['approved'])
    else:
        corr, p_value = 0, 1
    
    return {
        'ordering': ordering_name,
        'ethnic_approval_rates': approval_rates_by_ethnicity,
        'max_ethnic_disparity': max_disparity,
        'credit_score_correlation': corr,
        'credit_score_p_value': p_value,
        'quartile_approval_rates': quartile_approval,
        'chi2_statistic': chi2,
        'chi2_p_value': chi2_p_value,
        'total_samples': len(df),
        'ethnic_group_sizes': {k: v['total'] for k, v in ethnic_data.items()}
    }

def process_70b_model():
    """Process Llama 70B model"""
    
    print("\n" + "="*80)
    print("Processing Llama 70B Model")
    print("="*80 + "\n")
    
    base_dir = Path("70bresults/sequential")
    orderings_analysis = []
    
    for json_file in sorted(base_dir.glob("sequential_*_formatted.json")):
        ordering_name = extract_agent_order_from_filename(json_file.name)
        print(f"Processing: {ordering_name}...")
        
        with open(json_file) as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        analysis = analyze_ordering_fairness(results, ordering_name, 'large')
        
        if analysis:
            orderings_analysis.append(analysis)
            print(f"  ✓ Disparity: {analysis['max_ethnic_disparity']*100:.2f}%, "
                  f"Credit corr: {analysis['credit_score_correlation']:.3f}")
    
    return orderings_analysis

def process_qwen_model():
    """Process Qwen model"""
    
    print("\n" + "="*80)
    print("Processing Qwen Model")
    print("="*80 + "\n")
    
    base_dir = Path("qwen_validation")
    orderings_analysis = []
    
    for json_file in sorted(base_dir.glob("sequential_*.json")):
        ordering_name = extract_agent_order_from_filename(json_file.name)
        print(f"Processing: {ordering_name}...")
        
        with open(json_file) as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        analysis = analyze_ordering_fairness(results, ordering_name, 'large')
        
        if analysis:
            orderings_analysis.append(analysis)
            print(f"  ✓ Disparity: {analysis['max_ethnic_disparity']*100:.2f}%, "
                  f"Credit corr: {analysis['credit_score_correlation']:.3f}")
    
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
        ordering_name = extract_agent_order_from_filename(json_file.name)
        print(f"Processing: {ordering_name}...")
        
        data = loader.load_json_file(json_file)
        
        if not data:
            print(f"  ⚠️  Skipped (couldn't load)")
            continue
        
        # Handle dict with 'results' key
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
        elif isinstance(data, list):
            results = data
        else:
            print(f"  ⚠️  Skipped (unexpected format)")
            continue
        
        analysis = analyze_ordering_fairness(results, ordering_name, 'small')
        
        if analysis:
            # For Mistral, skip 0% approval orderings by checking overall approval rate
            if model_name == "mistral:latest":
                overall_approval_rate = sum(analysis['ethnic_approval_rates'].values()) / len(analysis['ethnic_approval_rates']) if analysis['ethnic_approval_rates'] else 0
                if overall_approval_rate == 0:
                    print(f"  ⚠️  Skipped (0% approval ordering)")
                    continue
            
            orderings_analysis.append(analysis)
            print(f"  ✓ Disparity: {analysis['max_ethnic_disparity']*100:.2f}%, "
                  f"Credit corr: {analysis['credit_score_correlation']:.3f}")
    
    return orderings_analysis

def create_functional_collapse_plots(model_results, output_dir):
    """Create plots showing functional collapse patterns"""
    
    # Plot 1: Demographic Disparity vs Credit Score Correlation (scatter)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'Llama 70B': '#e74c3c', 'Qwen': '#3498db', 'Llama 3.2': '#2ecc71', 'Mistral Latest': '#f39c12'}
    
    for model_name, analyses in model_results.items():
        if not analyses:
            continue
        
        disparities = [a['max_ethnic_disparity'] * 100 for a in analyses]
        correlations = [abs(a['credit_score_correlation']) for a in analyses]
        
        ax.scatter(correlations, disparities, 
                  label=model_name, alpha=0.7, s=100, 
                  color=colors.get(model_name, '#95a5a6'))
    
    ax.set_xlabel('|Credit Score Correlation| (Spearman ρ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max Ethnic Disparity (%)', fontsize=12, fontweight='bold')
    ax.set_title('Functional Collapse: Demographic Parity vs Credit Score Correlation\n'
                 'Each point represents one sequential ordering',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Add quadrant lines
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% disparity threshold')
    ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='Weak correlation threshold')
    
    plt.tight_layout()
    
    output_file = output_dir / "functional_collapse_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    output_file_pdf = output_dir / "functional_collapse_scatter.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_file_pdf}")
    
    plt.close()
    
    # Plot 2: Heatmaps of approval rates by ethnicity for each model
    n_models = len([m for m in model_results if model_results[m]])
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    
    if n_models == 1:
        axes = [axes]
    
    idx = 0
    for model_name in ['Llama 70B', 'Qwen', 'Llama 3.2', 'Mistral Latest']:
        if model_name not in model_results or not model_results[model_name]:
            continue
        
        analyses = model_results[model_name]
        
        # Create matrix of approval rates by ethnicity
        orderings = [a['ordering'] for a in analyses]
        ethnicities = ['Asian_Signal', 'Black_Signal', 'Hispanic_Signal', 'White_Signal']
        
        matrix = []
        for ethnicity in ethnicities:
            row = []
            for analysis in analyses:
                rate = analysis['ethnic_approval_rates'].get(ethnicity, 0) * 100
                row.append(rate)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        ax = axes[idx]
        sns.heatmap(matrix, annot=False, cmap='RdYlGn', 
                   xticklabels=[o[:8] + '...' if len(o) > 8 else o for o in orderings],
                   yticklabels=['Asian', 'Black', 'Hispanic', 'White'],
                   cbar_kws={'label': 'Approval Rate (%)'},
                   vmin=0, vmax=100, ax=ax)
        
        ax.set_title(f'{model_name}\nApproval Rates by Ethnicity', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Ordering', fontsize=10)
        ax.set_ylabel('Ethnic Group', fontsize=10)
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        
        idx += 1
    
    plt.tight_layout()
    
    output_file = output_dir / "ethnic_approval_heatmaps.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    output_file_pdf = output_dir / "ethnic_approval_heatmaps.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_file_pdf}")
    
    plt.close()
    
    # Plot 3: Distribution of credit score correlations by model
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_for_violin = []
    labels = []
    
    for model_name in ['Llama 70B', 'Qwen', 'Llama 3.2', 'Mistral Latest']:
        if model_name not in model_results or not model_results[model_name]:
            continue
        
        correlations = [a['credit_score_correlation'] for a in model_results[model_name]]
        data_for_violin.append(correlations)
        labels.append(model_name)
    
    parts = ax.violinplot(data_for_violin, positions=range(len(labels)), 
                          showmeans=True, showmedians=True)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Credit Score Correlation (Spearman ρ)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Credit Score Correlation Across Orderings', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    output_file = output_dir / "credit_correlation_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    output_file_pdf = output_dir / "credit_correlation_distribution.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_file_pdf}")
    
    plt.close()

def identify_collapse_orderings(model_results, output_dir):
    """Identify orderings with good demographic parity but weak credit correlation"""
    
    print("\n" + "="*80)
    print("IDENTIFYING FUNCTIONAL COLLAPSE ORDERINGS")
    print("="*80)
    
    # Thresholds
    DISPARITY_THRESHOLD = 0.05  # 5% max disparity for "fair"
    WEAK_CORRELATION_THRESHOLD = 0.3  # |ρ| < 0.3 for "weak"
    
    collapse_orderings = {}
    
    for model_name, analyses in model_results.items():
        if not analyses:
            continue
        
        collapsed = []
        
        for analysis in analyses:
            is_fair = analysis['max_ethnic_disparity'] <= DISPARITY_THRESHOLD
            is_weak_corr = abs(analysis['credit_score_correlation']) < WEAK_CORRELATION_THRESHOLD
            
            if is_fair and is_weak_corr:
                collapsed.append(analysis)
        
        collapse_orderings[model_name] = collapsed
        
        print(f"\n{model_name}:")
        print(f"  Total orderings: {len(analyses)}")
        print(f"  Functionally collapsed: {len(collapsed)}")
        
        if collapsed:
            print(f"  Collapsed orderings:")
            for c in collapsed:
                print(f"    - {c['ordering']}: disparity={c['max_ethnic_disparity']*100:.2f}%, "
                      f"corr={c['credit_score_correlation']:.3f}")
    
    # Save to JSON
    collapse_summary = {}
    for model_name, collapsed in collapse_orderings.items():
        collapse_summary[model_name] = {
            'total_orderings': len(model_results[model_name]),
            'collapsed_count': len(collapsed),
            'collapsed_orderings': [{
                'ordering': c['ordering'],
                'max_ethnic_disparity': float(c['max_ethnic_disparity']),
                'credit_score_correlation': float(c['credit_score_correlation']),
                'ethnic_approval_rates': {k: float(v) for k, v in c['ethnic_approval_rates'].items()},
                'quartile_approval_rates': {str(k): float(v) for k, v in c['quartile_approval_rates'].items()}
            } for c in collapsed]
        }
    
    output_file = output_dir / "functional_collapse_summary.json"
    with open(output_file, 'w') as f:
        json.dump(collapse_summary, f, indent=2)
    
    print(f"\n✓ Saved: {output_file}")
    
    return collapse_orderings

def main():
    """Run Section 4.4 analysis"""
    
    output_dir = Path("icml_analysis/section_4_4_functional_collapse")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SECTION 4.4: FUNCTIONAL COLLAPSE ANALYSIS")
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
            'max_ethnic_disparity': float(a['max_ethnic_disparity']),
            'credit_score_correlation': float(a['credit_score_correlation']),
            'credit_score_p_value': float(a['credit_score_p_value']),
            'ethnic_approval_rates': {k: float(v) for k, v in a['ethnic_approval_rates'].items()},
            'quartile_approval_rates': {str(k): float(v) for k, v in a['quartile_approval_rates'].items()},
            'total_samples': int(a['total_samples'])
        } for a in analyses]
    
    metrics_file = output_dir / "fairness_metrics_all_orderings.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✓ Saved detailed metrics: {metrics_file}")
    
    # Create plots
    print("\n" + "="*80)
    print("Creating visualization plots...")
    print("="*80)
    
    create_functional_collapse_plots(model_results, output_dir)
    
    # Identify collapse orderings
    identify_collapse_orderings(model_results, output_dir)
    
    print("\n" + "="*80)
    print("✓ SECTION 4.4 ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
