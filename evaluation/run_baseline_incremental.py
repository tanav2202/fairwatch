#!/usr/bin/env python3
"""
Incremental Baseline Runner
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
from individual_agent_runner import IndividualAgentRunner

import experiment_config as config


def run_with_incremental_saves(csv_file, sample_size=None):
    """Run baseline with incremental saves after each prompt"""
    
    print("\n" + "="*80)
    print("INCREMENTAL BASELINE RUNNER")
    print("="*80)
    print(f"Config: Sample size = {sample_size or 'All prompts'}")
    print("="*80 + "\n")
    
    # Initialize
    client = OllamaClient(model=config.MODEL_NAME, timeout=config.TIMEOUT)
    
    if not client.health_check():
        print("❌ Ollama not available")
        return
    
    # Load agents
    agents = [
        FarmerAgent(client),
        AdvocacyAgent(client),
        ScienceAgent(client),
        MediaAgent(client),
        PolicyAgent(client),
    ]
    
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
    
    print(f"Processing {len(prompts)} prompts\n")
    
    # Setup output
    output_dir = Path(config.BASELINE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output_dir / f"baseline_incremental_{run_id}.json"
    csv_file_out = output_dir / f"baseline_progress_{run_id}.csv"
    
    # Initialize results structure
    all_results = {agent.name: [] for agent in agents}
    
    # CSV progress tracking
    with open(csv_file_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt_idx', 'prompt', 'agent', 'status', 'timestamp'])
    
    # Process each prompt
    start_time = time.time()
    runner = IndividualAgentRunner(client, output_dir=str(output_dir))
    
    for idx, prompt in enumerate(prompts):
        elapsed = time.time() - start_time
        eta = (elapsed / (idx + 1)) * (len(prompts) - idx - 1) if idx > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"PROMPT {idx+1}/{len(prompts)}")
        print(f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
        print(f"{'='*80}")
        print(f"Prompt: {prompt[:100]}...")
        print(f"{'-'*80}\n")
        
        # Run each agent on this prompt
        for agent in agents:
            agent_start = time.time()
            
            try:
                print(f"  Running {agent.name}...", end=' ', flush=True)
                
                result = runner.run_agent_on_prompts(
                    agent=agent,
                    prompts=[prompt],
                    bias_types=bias_types,
                    run_id=run_id
                )[0]  # Get first result
                
                all_results[agent.name].append(result)
                
                agent_time = time.time() - agent_start
                print(f"✓ ({agent_time:.1f}s)")
                
                # Log progress to CSV
                with open(csv_file_out, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        idx,
                        prompt[:50],
                        agent.name,
                        'complete',
                        datetime.now().isoformat()
                    ])
                
            except Exception as e:
                print(f"✗ FAILED: {e}")
                
                # Log failure
                with open(csv_file_out, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        idx,
                        prompt[:50],
                        agent.name,
                        f'failed: {str(e)}',
                        datetime.now().isoformat()
                    ])
        
        # Save JSON after each prompt
        print(f"\n  Saving checkpoint...", end=' ', flush=True)
        
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "num_agents": len(agents),
                "num_prompts_completed": idx + 1,
                "num_prompts_total": len(prompts),
                "progress": f"{(idx+1)/len(prompts)*100:.1f}%",
                "agents": list(all_results.keys()),
            },
            "results": {
                agent_name: [r.to_dict() for r in agent_results]
                for agent_name, agent_results in all_results.items()
            },
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
    print(f"Agents: {len(agents)}")
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
