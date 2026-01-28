#!/usr/bin/env python3
"""
Create position×ordering heatmaps for Llama 3.2 and Mistral models.
Similar to the 70B and Qwen heatmaps but for small models.
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
    """Extract agent order from filename"""
    name = filename.replace("sequential_", "").replace(".json", "")
    
    # Map abbreviated names to full names
    agent_map = {
        "consumer": "Consumer_Advocate",
        "data": "Data_Science",
        "regulatory": "Regulatory",
        "risk": "Risk_Manager"
    }
    
    parts = name.split("_")
    order = []
    for part in parts:
        if part in agent_map:
            order.append(agent_map[part])
    
    return order

def create_position_ordering_heatmap(model_name, model_display_name, output_dir):
    """Create position×ordering heatmap for a small model"""
    
    print(f"\n{'='*80}")
    print(f"Creating position×ordering heatmap for {model_display_name}")
    print(f"{'='*80}\n")
    
    # Load data - handle colon in path
    base_dir = Path("outputs_simple") / model_name
    sequential_dir = base_dir / "sequential"
    
    if not sequential_dir.exists():
        print(f"❌ Sequential directory not found: {sequential_dir}")
        return
    
    # Create loader for LFS smudging
    loader = SmallModelDataLoader("outputs_simple", model_name)
    
    # Process all sequential orderings
    ordering_data = []
    
    json_files = sorted(sequential_dir.glob("sequential_*.json"))
    print(f"Found {len(json_files)} sequential ordering files\n")
    
    for json_file in json_files:
        print(f"Processing: {json_file.name}...")
        
        # Load the file
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
        
        # Extract agent order from filename
        agent_order = extract_agent_order_from_filename(json_file.name)
        if len(agent_order) != 4:
            print(f"  ⚠️  Skipped (invalid agent order: {agent_order})")
            continue
        
        # Calculate approval rates at each position
        position_approvals = defaultdict(list)
        
        for record in records:
            # Get conversation history with agent outputs
            if 'conversation_history' in record:
                conversation = record['conversation_history']
                
                for idx, turn_data in enumerate(conversation, 1):
                    if 'output' in turn_data and isinstance(turn_data['output'], dict):
                        output = turn_data['output']
                        decision = output.get('approval_decision', '')
                        
                        # Get turn number (position) - try 'turn' key first, then use index
                        turn = turn_data.get('turn', idx)
                        
                        if decision and 1 <= turn <= 4:
                            is_approved = decision.lower() == 'approve'
                            position_approvals[turn].append(is_approved)
        
        # Calculate rates
        if position_approvals:
            rates = {}
            for pos in [1, 2, 3, 4]:
                if pos in position_approvals and position_approvals[pos]:
                    rates[pos] = sum(position_approvals[pos]) / len(position_approvals[pos]) * 100
                else:
                    rates[pos] = 0.0
            
            ordering_data.append({
                'ordering': '_'.join(agent_order),
                'agent_order': agent_order,
                'pos1': rates.get(1, 0),
                'pos2': rates.get(2, 0),
                'pos3': rates.get(3, 0),
                'pos4': rates.get(4, 0),
                'total_chains': len(position_approvals.get(1, []))
            })
            
            print(f"  ✓ Processed {len(position_approvals.get(1, []))} chains")
            print(f"    Approval rates: {rates[1]:.1f}%, {rates[2]:.1f}%, {rates[3]:.1f}%, {rates[4]:.1f}%")
    
    if not ordering_data:
        print(f"\n❌ No data processed for {model_display_name}")
        return
    
    # Create DataFrame
    df = pd.DataFrame(ordering_data)
    
    # Calculate overall average
    overall_avg = df[['pos1', 'pos2', 'pos3', 'pos4']].mean().mean()
    
    print(f"\n{'='*80}")
    print(f"Creating heatmap...")
    print(f"  Orderings: {len(df)}")
    print(f"  Overall average: {overall_avg:.1f}%")
    print(f"{'='*80}\n")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Prepare data matrix
    matrix = df[['pos1', 'pos2', 'pos3', 'pos4']].values.T
    
    # Create abbreviated labels for orderings
    ordering_labels = []
    for _, row in df.iterrows():
        agents = row['agent_order']
        # Abbreviate: Consumer_Advocate -> CA, Data_Science -> DS, etc.
        abbrev = '_'.join([
            'CA' if a.startswith('Consumer') else
            'DS' if a.startswith('Data') else
            'Reg' if a.startswith('Regulatory') else 'RM'
            for a in agents
        ])
        ordering_labels.append(abbrev)
    
    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
               xticklabels=ordering_labels,
               yticklabels=['Position 1\n(First)', 'Position 2', 'Position 3', 'Position 4\n(Final)'],
               cbar_kws={'label': 'Approval Rate (%)'},
               vmin=0, vmax=100,
               ax=ax, linewidths=0.5, linecolor='white')
    
    # Formatting
    ax.set_xlabel('Agent Ordering', fontsize=13, fontweight='bold')
    ax.set_ylabel('Agent Position in Chain', fontsize=13, fontweight='bold')
    ax.set_title(f'{model_display_name}: Approval Rates by Agent Position Across All Orderings\n({overall_avg:.1f}% = Overall Average)',
                fontsize=15, fontweight='bold', pad=20)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'{model_name.replace(":", "_").replace(".", "_")}_position_ordering_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")
    
    # Also save the data
    data_file = output_dir / f'{model_name.replace(":", "_").replace(".", "_")}_position_ordering_data.csv'
    df.to_csv(data_file, index=False)
    print(f"✓ Saved data: {data_file}")

def main():
    """Generate heatmaps for both small models"""
    
    output_dir = Path("icml_analysis/section_4_1_ordering_instability")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GENERATING POSITION×ORDERING HEATMAPS FOR SMALL MODELS")
    print("="*80)
    
    # Process Llama 3.2
    create_position_ordering_heatmap("llama3.2", "Llama 3.2 Model", output_dir)
    
    # Process Mistral
    create_position_ordering_heatmap("mistral:latest", "Mistral Latest Model", output_dir)
    
    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
