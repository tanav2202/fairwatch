#!/usr/bin/env python3
"""
Create first vs final decision transition plots for all sequential orderings.
One comprehensive plot per model showing all orderings.
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
import math

from run_small_model_analysis import SmallModelDataLoader

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def extract_agent_order_from_filename(filename):
    """Extract agent order from filename
    
    Returns tuple of (abbreviated_name, agent_list)
    """
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

def analyze_first_vs_final(records, ordering_name, agent_order=None):
    """Analyze first agent vs final decision transition
    
    Args:
        records: List of decision records
        ordering_name: Name of the ordering
        agent_order: List of agent names in order (for 70B/Qwen format)
    """
    
    first_decisions = []
    final_decisions = []
    
    for record in records:
        if 'conversation_history' in record:
            # Small model format (Llama 3.2, Mistral)
            conversation = record['conversation_history']
            
            if len(conversation) >= 4:
                # First agent decision
                first = conversation[0]
                if 'output' in first and isinstance(first['output'], dict):
                    first_dec = first['output'].get('approval_decision', '')
                    
                    # Final agent decision (last in chain)
                    final = conversation[3]  # 4th agent (index 3)
                    if 'output' in final and isinstance(final['output'], dict):
                        final_dec = final['output'].get('approval_decision', '')
                        
                        if first_dec and final_dec:
                            first_decisions.append(first_dec.lower())
                            final_decisions.append(final_dec.lower())
        
        elif 'decisions' in record and agent_order:
            # Large model format (70B, Qwen)
            decisions = record['decisions']
            
            if len(agent_order) >= 4:
                # Get first and last agent from ordering
                first_agent = agent_order[0]
                final_agent = agent_order[3]
                
                if first_agent in decisions and final_agent in decisions:
                    first_dec = decisions[first_agent].get('approval_decision', '')
                    final_dec = decisions[final_agent].get('approval_decision', '')
                    
                    if first_dec and final_dec:
                        first_decisions.append(first_dec.lower())
                        final_decisions.append(final_dec.lower())
    
    if not first_decisions:
        return None
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    
    # Map to binary
    first_binary = ['Approve' if d == 'approve' else 'Deny' for d in first_decisions]
    final_binary = ['Approve' if d == 'approve' else 'Deny' for d in final_decisions]
    
    cm = confusion_matrix(first_binary, final_binary, labels=['Approve', 'Deny'])
    
    # Calculate probabilities
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_prob = cm / row_sums
    
    # Calculate metrics
    total = len(first_decisions)
    agreement = sum(1 for f, fin in zip(first_binary, final_binary) if f == fin)
    agreement_rate = agreement / total if total > 0 else 0
    
    approve_to_deny = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
    deny_to_approve = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
    
    return {
        'ordering': ordering_name,
        'confusion_matrix': cm,
        'confusion_matrix_prob': cm_prob,
        'total': total,
        'agreement': agreement,
        'agreement_rate': agreement_rate,
        'approve_to_deny': approve_to_deny,
        'deny_to_approve': deny_to_approve,
        'approve_to_deny_rate': approve_to_deny / cm[0].sum() if cm[0].sum() > 0 else 0,
        'deny_to_approve_rate': deny_to_approve / cm[1].sum() if cm[1].sum() > 0 else 0
    }

def create_model_transition_grid(model_name, model_display_name, orderings_data, output_dir):
    """Create grid of transition matrices for all orderings in a model"""
    
    n_orderings = len(orderings_data)
    
    if n_orderings == 0:
        print(f"⚠️  No data for {model_display_name}")
        return
    
    # Calculate grid dimensions
    n_cols = 6 if n_orderings > 12 else 4
    n_rows = math.ceil(n_orderings / n_cols)
    
    # Create figure
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3.5))
    
    for idx, data in enumerate(orderings_data, 1):
        # Create subplot
        ax = plt.subplot(n_rows, n_cols, idx)
        
        # Get confusion matrix
        cm = data['confusion_matrix']
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                   xticklabels=['Approve', 'Deny'],
                   yticklabels=['Approve', 'Deny'],
                   cbar=False, ax=ax,
                   vmin=0, vmax=cm.max(),
                   linewidths=1, linecolor='white')
        
        # Title with ordering and agreement rate
        ordering_label = data['ordering']
        agreement_pct = data['agreement_rate'] * 100
        
        ax.set_title(f"{ordering_label}\nAgree: {agreement_pct:.1f}%",
                    fontsize=9, fontweight='bold', pad=5)
        ax.set_xlabel('Final Decision', fontsize=8)
        ax.set_ylabel('First Agent', fontsize=8)
        
        # Adjust tick labels
        ax.tick_params(labelsize=7)
    
    # Overall title
    overall_agreement = np.mean([d['agreement_rate'] for d in orderings_data]) * 100
    fig.suptitle(f'{model_display_name}: First vs Final Decision Transitions Across All Orderings\n'
                f'Average Agreement Rate: {overall_agreement:.2f}%',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    
    # Save
    filename = f"{model_name.replace(':', '_').replace('.', '_')}_first_vs_final_all_orderings.png"
    output_file = output_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")
    
    # Save metrics
    metrics = {
        'model': model_display_name,
        'total_orderings': n_orderings,
        'average_agreement_rate': float(overall_agreement / 100),
        'orderings': [{
            'ordering': d['ordering'],
            'total_chains': int(d['total']),
            'agreement_rate': float(d['agreement_rate']),
            'approve_to_deny_flip': int(d['approve_to_deny']),
            'deny_to_approve_flip': int(d['deny_to_approve']),
            'confusion_matrix': [[int(x) for x in row] for row in d['confusion_matrix']]
        } for d in orderings_data]
    }
    
    metrics_file = output_dir / f"{model_name.replace(':', '_').replace('.', '_')}_transition_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved metrics: {metrics_file}")

def process_70b_model(output_dir):
    """Process Llama 70B model"""
    
    print("\n" + "="*80)
    print("Processing Llama 70B Model")
    print("="*80 + "\n")
    
    base_dir = Path("70bresults/sequential")
    orderings_data = []
    
    for json_file in sorted(base_dir.glob("sequential_*_formatted.json")):
        print(f"Processing: {json_file.name}...")
        
        with open(json_file) as f:
            data = json.load(f)
        
        records = data.get('results', [])
        ordering_name, agent_order = extract_agent_order_from_filename(json_file.name)
        
        result = analyze_first_vs_final(records, ordering_name, agent_order)
        
        if result:
            orderings_data.append(result)
            print(f"  ✓ {result['total']} chains, {result['agreement_rate']*100:.1f}% agreement")
        else:
            print(f"  ⚠️  Skipped (no data)")
    
    create_model_transition_grid("llama70b", "Llama 70B", orderings_data, output_dir)

def process_qwen_model(output_dir):
    """Process Qwen model"""
    
    print("\n" + "="*80)
    print("Processing Qwen Model")
    print("="*80 + "\n")
    
    base_dir = Path("qwen_validation")
    orderings_data = []
    
    for json_file in sorted(base_dir.glob("sequential_*.json")):
        print(f"Processing: {json_file.name}...")
        
        with open(json_file) as f:
            data = json.load(f)
        
        records = data.get('results', [])
        ordering_name, agent_order = extract_agent_order_from_filename(json_file.name)
        
        result = analyze_first_vs_final(records, ordering_name, agent_order)
        
        if result:
            orderings_data.append(result)
            print(f"  ✓ {result['total']} chains, {result['agreement_rate']*100:.1f}% agreement")
        else:
            print(f"  ⚠️  Skipped (no data)")
    
    create_model_transition_grid("qwen", "Qwen", orderings_data, output_dir)

def process_small_model(model_name, model_display_name, output_dir):
    """Process Llama 3.2 or Mistral model"""
    
    print("\n" + "="*80)
    print(f"Processing {model_display_name} Model")
    print("="*80 + "\n")
    
    loader = SmallModelDataLoader("outputs_simple", model_name)
    base_dir = Path("outputs_simple") / model_name / "sequential"
    
    orderings_data = []
    
    for json_file in sorted(base_dir.glob("sequential_*.json")):
        print(f"Processing: {json_file.name}...")
        
        data = loader.load_json_file(json_file)
        
        if not data:
            print(f"  ⚠️  Skipped (couldn't load)")
            continue
        
        # Handle dict with 'results' key
        if isinstance(data, dict) and 'results' in data:
            records = data['results']
        elif isinstance(data, list):
            records = data
        else:
            print(f"  ⚠️  Skipped (unexpected format)")
            continue
        
        ordering_name, agent_order = extract_agent_order_from_filename(json_file.name)
        
        result = analyze_first_vs_final(records, ordering_name, agent_order)
        
        if result:
            # For Mistral, filter out 0% approval orderings
            if model_name == "mistral:latest" and result['total'] > 0:
                # Check if this is a 0% approval ordering
                total_approvals = result['confusion_matrix'][0].sum()  # Row 0 is approvals
                if total_approvals == 0:
                    print(f"  ⚠️  Skipped (0% approval ordering)")
                    continue
            
            orderings_data.append(result)
            print(f"  ✓ {result['total']} chains, {result['agreement_rate']*100:.1f}% agreement")
    
    create_model_transition_grid(model_name, model_display_name, orderings_data, output_dir)

def main():
    """Generate transition plots for all models"""
    
    output_dir = Path("icml_analysis/section_4_2_information_cascades")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GENERATING FIRST VS FINAL DECISION TRANSITIONS FOR ALL MODELS")
    print("="*80)
    
    # Process each model
    process_70b_model(output_dir)
    process_qwen_model(output_dir)
    process_small_model("llama3.2", "Llama 3.2", output_dir)
    process_small_model("mistral:latest", "Mistral Latest", output_dir)
    
    print("\n" + "="*80)
    print("✓ COMPLETE - All model transition plots generated")
    print("="*80)

if __name__ == "__main__":
    main()
