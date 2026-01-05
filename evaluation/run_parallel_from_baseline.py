"""
Run Parallel Synthesis from Baseline Results
Reuses individual agent evaluations, runs only business synthesis
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.business_decision_agent import BusinessDecisionAgent
from utils.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def load_baseline_results(baseline_dir: Path) -> dict:
    """
    Load all 4 agent baseline results from JSON files
    
    Args:
        baseline_dir: Directory containing baseline JSON files
        
    Returns:
        Dict mapping agent names to their results
    """
    agent_files = {
        'risk_manager': 'risk_manager_baseline.json',
        'regulatory': 'regulatory_baseline.json',
        'data_science': 'data_science_baseline.json',
        'consumer_advocate': 'consumer_advocate_baseline.json'
    }
    
    baseline_results = {}
    
    for agent_name, filename in agent_files.items():
        filepath = baseline_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Baseline file not found: {filepath}")
        
        LOG.info(f"Loading baseline results: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
            baseline_results[agent_name] = data['results']
            LOG.info(f"  Loaded {len(data['results'])} results from {agent_name}")
    
    # Verify all agents have same number of results
    result_counts = {name: len(results) for name, results in baseline_results.items()}
    if len(set(result_counts.values())) > 1:
        raise ValueError(f"Mismatched result counts: {result_counts}")
    
    LOG.info(f"\nAll agents have {list(result_counts.values())[0]} results ✓")
    
    return baseline_results


def save_incremental(output_file: Path, metadata: dict, results: list):
    """Save results incrementally"""
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": metadata,
            "results": results
        }, f, indent=2)


def run_parallel_synthesis(
    llm_model: str,
    baseline_dir: str,
    output_dir: str,
    template_name: str = "simple"
):
    """
    Run parallel synthesis using baseline results
    
    Args:
        llm_model: LLM model name for business agent
        baseline_dir: Directory containing baseline results
        output_dir: Output directory for synthesis results
        template_name: Template name (simple/complex) for metadata
    """
    # Setup paths
    baseline_path = Path(baseline_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load baseline results
    LOG.info(f"\n{'='*60}")
    LOG.info(f"Loading baseline results from {baseline_path}")
    LOG.info(f"{'='*60}\n")
    
    baseline_results = load_baseline_results(baseline_path)
    num_prompts = len(baseline_results['risk_manager'])
    
    # Initialize business agent
    LOG.info(f"Initializing Business Decision Agent (model: {llm_model})")
    client = OllamaClient(model=llm_model)
    business_agent = BusinessDecisionAgent(client)
    
    # Output file
    output_file = output_path / f"parallel_synthesis_{template_name}.json"
    
    # Check for existing results (resume capability)
    existing_results = []
    start_idx = 0
    if output_file.exists():
        LOG.warning(f"Found existing results at {output_file}")
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            existing_results = existing_data.get('results', [])
            start_idx = len(existing_results)
        LOG.warning(f"Resuming from prompt {start_idx + 1}/{num_prompts}\n")
    
    all_results = existing_results.copy()
    
    # Metadata
    metadata = {
        "llm_model": llm_model,
        "mode": "parallel_synthesis",
        "template": template_name,
        "baseline_dir": str(baseline_path),
        "agents": ["risk_manager", "regulatory", "data_science", "consumer_advocate"],
        "total_prompts": num_prompts,
        "started_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }
    
    # Run synthesis
    LOG.info(f"\n{'='*60}")
    LOG.info(f"Running parallel synthesis on {num_prompts} prompts")
    LOG.info(f"Template: {template_name}")
    LOG.info(f"{'='*60}\n")
    
    for i in range(start_idx, num_prompts):
        prompt_num = i + 1
        
        LOG.info(f"Synthesizing prompt {prompt_num}/{num_prompts}")
        
        try:
            # Gather all 4 agent outputs for this prompt
            agent_recommendations = [
                baseline_results['risk_manager'][i]['output'],
                baseline_results['regulatory'][i]['output'],
                baseline_results['data_science'][i]['output'],
                baseline_results['consumer_advocate'][i]['output']
            ]
            
            # Get original prompt
            original_prompt = baseline_results['risk_manager'][i]['prompt']
            
            # Run business synthesis
            business_output = business_agent.synthesize_decision(
                application_data=original_prompt,
                agent_recommendations=agent_recommendations
            )
            
            # Store result
            result = {
                "prompt_id": prompt_num,
                "prompt": original_prompt,
                "mode": "parallel",
                "agent_outputs": {
                    "risk_manager": baseline_results['risk_manager'][i]['output'],
                    "regulatory": baseline_results['regulatory'][i]['output'],
                    "data_science": baseline_results['data_science'][i]['output'],
                    "consumer_advocate": baseline_results['consumer_advocate'][i]['output']
                },
                "business_decision": business_output,
                "timestamp": datetime.now().isoformat()
            }
            
            all_results.append(result)
            
            # Log summary
            decision = business_output.get('approval_decision', 'N/A')
            approval_type = business_output.get('approval_type', 'N/A')
            primary = business_output.get('agent_influence', {}).get('primary_influence', 'N/A')
            
            LOG.info(f"  Business Decision: {decision}")
            LOG.info(f"  Approval Type: {approval_type}")
            LOG.info(f"  Primary Influence: {primary}")
            
        except Exception as e:
            LOG.error(f"Error on prompt {prompt_num}: {e}")
            all_results.append({
                "prompt_id": prompt_num,
                "prompt": baseline_results['risk_manager'][i]['prompt'],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        # SAVE AFTER EVERY SYNTHESIS (incremental)
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["completed_prompts"] = len(all_results)
        save_incremental(output_file, metadata, all_results)
        
        LOG.info(f"  💾 Saved to {output_file} ({len(all_results)}/{num_prompts} complete)\n")
    
    # Mark as complete
    metadata["completed_at"] = datetime.now().isoformat()
    metadata["status"] = "complete"
    save_incremental(output_file, metadata, all_results)
    
    LOG.info(f"\n{'='*60}")
    LOG.info(f"✅ Parallel synthesis complete!")
    LOG.info(f"Results saved to: {output_file}")
    LOG.info(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run parallel synthesis from baseline results"
    )
    parser.add_argument(
        "--llm", 
        required=True, 
        help="LLM model name for business agent (e.g., llama3.2)"
    )
    parser.add_argument(
        "--baseline-dir", 
        required=True,
        help="Directory containing baseline JSON files (e.g., outputs_simple/llama3.2)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for synthesis results (e.g., outputs_simple/llama3.2/parallel)"
    )
    parser.add_argument(
        "--template",
        default="simple",
        choices=["simple", "complex"],
        help="Template name for metadata (default: simple)"
    )
    
    args = parser.parse_args()
    
    run_parallel_synthesis(
        llm_model=args.llm,
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        template_name=args.template
    )