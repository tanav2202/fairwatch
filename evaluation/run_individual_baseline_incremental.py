"""
Run Individual Baseline Evaluations - WITH INCREMENTAL SAVING
Saves after each prompt to prevent data loss
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.risk_manager_agent import RiskManagerAgent
from agents.regulatory_agent import RegulatoryAgent
from agents.data_science_agent import DataScienceAgent
from agents.consumer_advocate_agent import ConsumerAdvocateAgent
from utils.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def load_prompts(prompts_file: str) -> list:
    """Load prompts from CSV or JSON file"""
    import pandas as pd
    
    if prompts_file.endswith('.csv'):
        df = pd.read_csv(prompts_file)
        if 'prompt' in df.columns:
            return df['prompt'].tolist()
        elif 'application' in df.columns:
            return df['application'].tolist()
        else:
            return df.iloc[:, 0].tolist()
    elif prompts_file.endswith('.json'):
        with open(prompts_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [item if isinstance(item, str) else item.get('prompt', str(item)) for item in data]
            return [str(data)]
    else:
        raise ValueError("Prompts file must be .csv or .json")


def save_incremental(output_file: Path, metadata: dict, results: list):
    """
    Save results incrementally (overwrites file each time)
    
    Args:
        output_file: Path to output JSON file
        metadata: Metadata dict
        results: List of results so far
    """
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": metadata,
            "results": results
        }, f, indent=2)


def run_baseline(llm_model: str, prompts_file: str, output_dir: str):
    """
    Run baseline evaluations for all agents with incremental saving
    
    Args:
        llm_model: Name of LLM model
        prompts_file: Path to prompts file
        output_dir: Output directory for results
    """
    # Create output directory
    output_path = Path(output_dir) / llm_model
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    client = OllamaClient(model=llm_model)
    
    # Initialize agents
    agents = {
        "risk_manager": RiskManagerAgent(client),
        "regulatory": RegulatoryAgent(client),
        "data_science": DataScienceAgent(client),
        "consumer_advocate": ConsumerAdvocateAgent(client)
    }
    
    # Load prompts
    LOG.info(f"Loading prompts from {prompts_file}")
    prompts = load_prompts(prompts_file)
    LOG.info(f"Loaded {len(prompts)} prompts")
    
    # Run each agent
    for agent_name, agent in agents.items():
        LOG.info(f"\n{'='*60}")
        LOG.info(f"Running {agent_name} on {len(prompts)} prompts")
        LOG.info(f"{'='*60}\n")
        
        output_file = output_path / f"{agent_name}_baseline.json"
        
        # Check if partially complete (resume capability)
        existing_results = []
        start_idx = 0
        if output_file.exists():
            LOG.warning(f"Found existing results at {output_file}")
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
                existing_results = existing_data.get('results', [])
                start_idx = len(existing_results)
            LOG.warning(f"Resuming from prompt {start_idx + 1}/{len(prompts)}")
        
        results = existing_results.copy()
        
        # Metadata
        metadata = {
            "llm_model": llm_model,
            "agent_name": agent_name,
            "total_prompts": len(prompts),
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        for i in range(start_idx, len(prompts)):
            prompt = prompts[i]
            prompt_num = i + 1
            
            LOG.info(f"[{agent_name}] Evaluating prompt {prompt_num}/{len(prompts)}")
            
            try:
                output = agent.evaluate_loan_application(prompt)
                
                result = {
                    "prompt_id": prompt_num,
                    "prompt": prompt,
                    "agent_name": agent_name,
                    "output": output,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
                
                # Log summary
                decision = output.get('approval_decision', 'N/A')
                approval_type = output.get('approval_type', 'N/A')
                confidence = output.get('confidence_probability', 'N/A')
                LOG.info(f"  model Decision: {decision}, Type: {approval_type}, Confidence: {confidence}%")
                
            except Exception as e:
                LOG.error(f"Error on prompt {prompt_num}: {e}")
                results.append({
                    "prompt_id": prompt_num,
                    "prompt": prompt,
                    "agent_name": agent_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            # SAVE AFTER EVERY PROMPT (incremental)
            metadata["last_updated"] = datetime.now().isoformat()
            metadata["completed_prompts"] = len(results)
            save_incremental(output_file, metadata, results)
            
            LOG.info(f"  model Saved to {output_file} ({len(results)}/{len(prompts)} complete)\n")
        
        # Mark as complete
        metadata["completed_at"] = datetime.now().isoformat()
        metadata["status"] = "complete"
        save_incremental(output_file, metadata, results)
        
        LOG.info(f"\n model {agent_name} complete. Final save to {output_file}\n")
    
    LOG.info(f"\n{'='*60}")
    LOG.info(f"model All baseline evaluations complete!")
    LOG.info(f"Results saved to: {output_path}")
    LOG.info(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run individual agent baseline evaluations")
    parser.add_argument("--llm", required=True, help="LLM model name (e.g., llama-3.2)")
    parser.add_argument("--prompts", required=True, help="Path to prompts file (.csv or .json)")
    parser.add_argument("--output", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    run_baseline(
        llm_model=args.llm,
        prompts_file=args.prompts,
        output_dir=args.output
    )