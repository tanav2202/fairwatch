#!/usr/bin/env python3
"""
Incremental Chain Runner
Saves results after each prompt to prevent data loss on long runs
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ollama_client import OllamaClient
from evaluation.bias_detector import BiasType
from agents.farmer_agent import FarmerAgent
from agents.advocacy_agent import AdvocacyAgent
from agents.science_agent import ScienceAgent
from agents.media_agent import MediaAgent
from agents.policy_agent import PolicyAgent
from chain_runner import ChainRunner
from agentic_systems.conversation_chain import ConversationChain

import experiment_config as config


def get_agent_instances(client):
    """Get agent instances by name"""
    return {
        'FarmerAgent': FarmerAgent(client),
        'AdvocacyAgent': AdvocacyAgent(client),
        'ScienceAgent': ScienceAgent(client),
        'MediaAgent': MediaAgent(client),
        'PolicyAgent': PolicyAgent(client),
    }


def run_with_incremental_saves(csv_file, sample_size=None):
    """Run chains with incremental saves after each prompt"""
    
    print("\n" + "="*80)
    print("INCREMENTAL CHAIN RUNNER")
    print("="*80)
    print(f"Config: Sample size = {sample_size or 'All prompts'}")
    print(f"Orderings: {len(config.AGENT_ORDERINGS)}")
    print("="*80 + "\n")
    
    # Initialize
    client = OllamaClient(model=config.MODEL_NAME, timeout=config.TIMEOUT)
    
    if not client.health_check():
        print("❌ Ollama not available")
        return
    
    agent_instances = get_agent_instances(client)
    
    # Convert config orderings to agent lists
    agent_orderings = []
    for ordering in config.AGENT_ORDERINGS:
        agents = [agent_instances[name] for name in ordering]
        agent_orderings.append(agents)
    
    bias_types = [
        BiasType.ECONOMIC_FRAMING,
        BiasType.CULTURAL_INSENSITIVITY,
        BiasType.SOURCE_BIAS,
        BiasType.OVERCAUTIOUS_FRAMING,
        BiasType.AGGREGATION_DISTORTION,
        BiasType.STANCE_BIAS,
        BiasType.EMOTIONAL_MANIPULATION,
        BiasType.REPRESENTATION_BIAS,
    ]
    
    # Load prompts
    prompts = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'prompt' in row and row['prompt'].strip():
                prompts.append(row['prompt'].strip())
    
    if sample_size and sample_size < len(prompts):
        import random
        random.seed(config.RANDOM_SEED)
        prompts = random.sample(prompts, sample_size)
    
    print(f"Processing {len(prompts)} prompts × {len(agent_orderings)} orderings\n")
    
    # Setup output
    output_dir = Path(config.CHAIN_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output_dir / f"chains_incremental_{run_id}.json"
    csv_file_out = output_dir / f"chains_progress_{run_id}.csv"
    
    # Initialize results
    all_results = []
    
    # CSV progress tracking
    with open(csv_file_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt_idx', 'ordering_idx', 'prompt', 'status', 'timestamp'])
    
    # Process each prompt × ordering
    start_time = time.time()
    runner = ChainRunner(client, output_dir=str(output_dir))
    
    total_runs = len(prompts) * len(agent_orderings)
    current_run = 0
    
    for prompt_idx, prompt in enumerate(prompts):
        for ordering_idx, agents in enumerate(agent_orderings):
            current_run += 1
            elapsed = time.time() - start_time
            eta = (elapsed / current_run) * (total_runs - current_run) if current_run > 0 else 0
            
            ordering_names = ' → '.join([a.name for a in agents])
            
            print(f"\n{'='*80}")
            print(f"RUN {current_run}/{total_runs}")
            print(f"Prompt {prompt_idx+1}/{len(prompts)} | Ordering {ordering_idx+1}/{len(agent_orderings)}")
            print(f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
            print(f"{'='*80}")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Chain: {ordering_names}")
            print(f"{'-'*80}\n")
            
            try:
                # Create chain with these agents
                chain = ConversationChain(agents)
                
                result = runner.run_chain_on_prompt(
                    chain=chain,
                    prompt=prompt,
                    prompt_index=prompt_idx,
                    bias_types=bias_types,
                    run_id=f"{run_id}_ord{ordering_idx}"
                )
                
                all_results.append(result)
                
                print(f"\n  ✓ Chain complete")
                
                # Log progress to CSV
                with open(csv_file_out, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        prompt_idx,
                        ordering_idx,
                        prompt[:50],
                        'complete',
                        datetime.now().isoformat()
                    ])
                
            except Exception as e:
                print(f"\n  ✗ FAILED: {e}")
                
                # Log failure
                with open(csv_file_out, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        prompt_idx,
                        ordering_idx,
                        prompt[:50],
                        f'failed: {str(e)}',
                        datetime.now().isoformat()
                    ])
            
            # Save JSON after each chain
            print(f"  Saving checkpoint...", end=' ', flush=True)
            
            # Extract metadata
            agent_sequences = list(set(tuple(r.agent_sequence) for r in all_results))
            unique_prompts = list(set(r.prompt for r in all_results))
            
            data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "run_id": run_id,
                    "num_chains_completed": len(all_results),
                    "num_chains_total": total_runs,
                    "num_prompts": len(prompts),
                    "num_orderings": len(agent_orderings),
                    "progress": f"{len(all_results)/total_runs*100:.1f}%",
                    "agent_sequences": [list(seq) for seq in agent_sequences],
                    "prompts": unique_prompts,
                },
                "results": [r.to_dict() for r in all_results],
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✓")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Prompts: {len(prompts)}")
    print(f"Orderings: {len(agent_orderings)}")
    print(f"Total chains: {len(all_results)}")
    print(f"\nOutputs:")
    print(f"  JSON:  {json_file}")
    print(f"  CSV:   {csv_file_out}")
    print(f"{'='*80}\n")
    
    return str(json_file)


def main():
    import sys
    
    # Get sample size from config or command line
    sample_size = config.PROMPT_SAMPLE_SIZE
    
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
            print(f"Using command line sample size: {sample_size}")
        except ValueError:
            print(f"Invalid sample size, using config: {sample_size}")
    
    run_with_incremental_saves(
        csv_file=config.PROMPT_CSV,
        sample_size=sample_size
    )


if __name__ == '__main__':
    main()