"""
Run Sequential Multi-Agent Chains - WITH INCREMENTAL SAVING
Saves after each chain to prevent data loss
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
from agents.business_decision_agent import BusinessDecisionAgent
from agentic_systems.conversation_chain import ConversationChain, ChainMode
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
    """Save results incrementally"""
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": metadata,
            "results": results
        }, f, indent=2)


def run_sequential_chains(
    llm_model: str,
    prompts_file: str,
    ordering: str,
    output_dir: str,
    include_business: bool = True
):
    """
    Run sequential agent chains with incremental saving
    """
    # Create output directory
    output_path = Path(output_dir) / llm_model / "sequential"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    client = OllamaClient(model=llm_model)
    
    # Agent mapping
    agent_map = {
        "risk": RiskManagerAgent(client),
        "regulatory": RegulatoryAgent(client),
        "data": DataScienceAgent(client),
        "consumer": ConsumerAdvocateAgent(client)
    }
    
    # Parse ordering
    agent_order = ordering.split('_')
    agents = [agent_map[name] for name in agent_order]
    
    # Initialize business agent if needed
    business_agent = BusinessDecisionAgent(client) if include_business else None
    
    # Initialize chain
    chain = ConversationChain(agents, mode=ChainMode.SEQUENTIAL)
    
    # Load prompts
    LOG.info(f"Loading prompts from {prompts_file}")
    prompts = load_prompts(prompts_file)
    LOG.info(f"Loaded {len(prompts)} prompts")
    LOG.info(f"Agent ordering: {' → '.join([a.agent_name for a in agents])}")
    
    # Output file
    output_file = output_path / f"sequential_{ordering}.json"
    
    # Check for existing results (resume capability)
    existing_results = []
    start_idx = 0
    if output_file.exists():
        LOG.warning(f"Found existing results at {output_file}")
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            existing_results = existing_data.get('results', [])
            start_idx = len(existing_results)
        LOG.warning(f"Resuming from prompt {start_idx + 1}/{len(prompts)}")
    
    all_results = existing_results.copy()
    
    # Metadata
    metadata = {
        "llm_model": llm_model,
        "mode": "sequential",
        "ordering": ordering,
        "agent_sequence": [a.agent_name for a in agents],
        "include_business_decision": include_business,
        "total_prompts": len(prompts),
        "started_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }
    
    # Run chains
    for i in range(start_idx, len(prompts)):
        prompt = prompts[i]
        prompt_num = i + 1
        
        LOG.info(f"\n{'='*60}")
        LOG.info(f"Running chain {prompt_num}/{len(prompts)}")
        LOG.info(f"{'='*60}\n")
        
        try:
            if include_business:
                result = chain.run_with_synthesis(
                    initial_prompt=prompt,
                    business_agent=business_agent,
                    mode=ChainMode.SEQUENTIAL
                )
            else:
                result = chain.run_sequential(prompt)
            
            result["prompt_id"] = prompt_num
            result["ordering"] = ordering
            result["timestamp"] = datetime.now().isoformat()
            
            all_results.append(result)
            
            # Log summary
            if include_business:
                business_decision = result["business_decision"]
                LOG.info(f"\nmodel Chain {prompt_num} complete")
                LOG.info(f"Final Business Decision: {business_decision.get('approval_decision')}")
                LOG.info(f"Approval Type: {business_decision.get('approval_type')}")
                LOG.info(f"Primary Influence: {business_decision.get('agent_influence', {}).get('primary_influence')}")
            else:
                final_agent = result["final_output"]
                LOG.info(f"\nmodel Chain {prompt_num} complete")
                LOG.info(f"Final Decision: {final_agent.get('approval_decision')}")
            
        except Exception as e:
            LOG.error(f"Error on chain {prompt_num}: {e}")
            all_results.append({
                "prompt_id": prompt_num,
                "prompt": prompt,
                "ordering": ordering,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        # SAVE INCREMENTALLY (every 50 iterations or at the end)
        if (i + 1) % 50 == 0 or (i + 1) == len(prompts):
            metadata["last_updated"] = datetime.now().isoformat()
            metadata["completed_chains"] = len(all_results)
            save_incremental(output_file, metadata, all_results)
            
            LOG.info(f"  💾 Saved to {output_file} ({len(all_results)}/{len(prompts)} complete)\n")
        
        chain.reset()
    
    # Mark as complete
    metadata["completed_at"] = datetime.now().isoformat()
    metadata["status"] = "complete"
    save_incremental(output_file, metadata, all_results)
    
    LOG.info(f"\n{'='*60}")
    LOG.info(f"model All sequential chains complete!")
    LOG.info(f"Results saved to: {output_file}")
    LOG.info(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sequential multi-agent chains")
    parser.add_argument("--llm", required=True, help="LLM model name")
    parser.add_argument("--prompts", required=True, help="Path to prompts file")
    parser.add_argument("--ordering", required=True, 
                       help="Agent ordering (e.g., risk_regulatory_data_consumer)")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--no-business", action="store_true", 
                       help="Skip business decision synthesis")
    
    args = parser.parse_args()
    
    run_sequential_chains(
        llm_model=args.llm,
        prompts_file=args.prompts,
        ordering=args.ordering,
        output_dir=args.output,
        include_business=not args.no_business
    )