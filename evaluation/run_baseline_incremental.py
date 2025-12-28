#!/usr/bin/env python3
"""
Incremental Baseline Runner
Saves results after each prompt to prevent data loss on long runs
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import experiment_config as config

from agents.risk_manager_agent import RiskManagerAgent
from agents.regulatory_agent import RegulatoryAgent
from agents.data_science_agent import DataScienceAgent
from agents.consumer_advocate_agent import ConsumerAdvocateAgent
from utils.ollama_client import OllamaClient


def run_with_incremental_saves(csv_file, sample_size=None):
    """Run baseline with incremental saves after each prompt"""

    print("\n" + "=" * 80)
    print("INCREMENTAL BASELINE RUNNER")
    print("=" * 80)
    print(f"Config: Sample size = {sample_size or 'All prompts'}")
    print("=" * 80 + "\n")

    # Initialize
    client = OllamaClient(model=config.MODEL_NAME, timeout=config.TIMEOUT)

    if not client.health_check():
        print("❌ Ollama not available")
        return

    # Load credit risk agents
    agents = [
        RiskManagerAgent(client),
        RegulatoryAgent(client),
        DataScienceAgent(client),
        ConsumerAdvocateAgent(client),
    ]

    # Load loan applications (prompts)
    applications = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "prompt" in row and row["prompt"].strip():
                applications.append(row["prompt"].strip())

    if sample_size and sample_size < len(applications):
        import random

        random.seed(config.RANDOM_SEED)
        applications = random.sample(applications, sample_size)

    print(f"Processing {len(applications)} loan applications\n")

    # Setup output
    output_dir = Path(config.BASELINE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output_dir / f"baseline_incremental_{run_id}.json"
    csv_file_out = output_dir / f"baseline_progress_{run_id}.csv"

    # Initialize results structure
    all_results = {agent.name: [] for agent in agents}

    # CSV progress tracking
    with open(csv_file_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["application_idx", "application", "agent", "status", "timestamp"]
        )

    # Process each loan application
    start_time = time.time()

    for idx, application in enumerate(applications):
        elapsed = time.time() - start_time
        eta = (elapsed / (idx + 1)) * (len(applications) - idx - 1) if idx > 0 else 0

        print(f"\n{'=' * 80}")
        print(f"APPLICATION {idx + 1}/{len(applications)}")
        print(f"Elapsed: {elapsed / 60:.1f}min | ETA: {eta / 60:.1f}min")
        print(f"{'=' * 80}")
        print(f"Application: {application[:100]}...")
        print(f"{'-' * 80}\n")

        # Run each agent on this application
        for agent in agents:
            agent_start = time.time()

            try:
                print(f"  Running {agent.name}...", end=" ", flush=True)

                # Get agent's loan decision
                result = agent.evaluate_loan_application(application)

                # Add metadata
                result["application_index"] = idx
                result["application_text"] = application
                result["timestamp"] = datetime.now().isoformat()

                all_results[agent.name].append(result)

                agent_time = time.time() - agent_start
                decision = result.get("loan_decision", "unknown")
                print(f"✓ ({agent_time:.1f}s) - {decision}")

                # Log progress to CSV
                with open(csv_file_out, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            idx,
                            application[:50],
                            agent.name,
                            "complete",
                            datetime.now().isoformat(),
                        ]
                    )

            except Exception as e:
                print(f"✗ FAILED: {e}")

                # Log failure
                with open(csv_file_out, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            idx,
                            application[:50],
                            agent.name,
                            f"failed: {str(e)}",
                            datetime.now().isoformat(),
                        ]
                    )

        # Save JSON after each application
        print(f"\n  Saving checkpoint...", end=" ", flush=True)

        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "num_agents": len(agents),
                "num_applications_completed": idx + 1,
                "num_applications_total": len(applications),
                "progress": f"{(idx + 1) / len(applications) * 100:.1f}%",
                "agents": list(all_results.keys()),
            },
            "results": all_results,
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓")

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Loan Applications: {len(applications)}")
    print(f"Agents: {len(agents)}")
    print(f"\nOutputs:")
    print(f"  JSON:  {json_file}")
    print(f"  CSV:   {csv_file_out}")
    print(f"{'=' * 80}\n")

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

    run_with_incremental_saves(csv_file=config.PROMPT_CSV, sample_size=sample_size)


if __name__ == "__main__":
    main()
