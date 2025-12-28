#!/usr/bin/env python3
"""
Greedy Agent Ordering Finder
Finds the best or worst agent ordering based on bias reduction/amplification

This module implements a greedy algorithm that:
1. Iteratively selects agents one at a time
2. At each step, evaluates all remaining agents
3. Chooses the agent that maximally reduces (or increases) bias
4. Returns the final optimal ordering

Usage:
    python greedy_ordering_finder.py --mode best --num-prompts 10
    python greedy_ordering_finder.py --mode worst --num-prompts 5
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ollama_client import OllamaClient
from evaluation.bias_detector import BiasType
from agents.farmer_agent import FarmerAgent
from agents.advocacy_agent import AdvocacyAgent
from agents.science_agent import ScienceAgent
from agents.media_agent import MediaAgent
from agents.policy_agent import PolicyAgent
from agentic_systems.conversation_chain import ConversationChain
from evaluation.chain_runner import ChainRunner

import evaluation.experiment_config as config


@dataclass
class OrderingEvaluation:
    """Result from evaluating a partial or complete agent ordering"""

    ordering: List[str]  # Agent names in order
    avg_bias_score: float  # Average bias score across all prompts and bias types
    num_prompts_tested: int
    bias_breakdown: Dict[str, float]  # bias_type -> avg score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GreedySearchResult:
    """Complete result from greedy search"""

    mode: str  # "best" or "worst"
    final_ordering: List[str]
    final_bias_score: float
    search_steps: List[Dict[str, Any]]  # Track decisions at each step
    num_prompts_tested: int
    total_evaluations: int
    timestamp: str
    output_file: str = ""  # Path to output JSON file

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GreedyOrderingFinder:
    """
    Finds optimal agent orderings using greedy algorithm

    The greedy approach:
    1. Start with empty ordering
    2. For each position:
       - Try all remaining agents
       - Run chains with current ordering + candidate agent
       - Evaluate average bias
       - Select agent that maximizes objective (min or max bias)
    3. Return final ordering
    """

    def __init__(
        self,
        client: OllamaClient,
        bias_types: List[BiasType],
        output_dir: str = "results/greedy_search",
    ):
        """
        Initialize greedy ordering finder

        Args:
            client: OllamaClient for running agents
            bias_types: List of bias types to evaluate
            output_dir: Directory for saving results
        """
        self.client = client
        self.bias_types = bias_types
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chain_runner = ChainRunner(client, output_dir=str(self.output_dir))

        # Available agents
        self.all_agents = {
            "FarmerAgent": FarmerAgent(client),
            "AdvocacyAgent": AdvocacyAgent(client),
            "ScienceAgent": ScienceAgent(client),
            "MediaAgent": MediaAgent(client),
            "PolicyAgent": PolicyAgent(client),
        }

        self.evaluation_count = 0  # Track total evaluations

    def _save_incremental_result(
        self,
        output_path: Path,
        mode: str,
        current_ordering: List[str],
        search_steps: List[Dict[str, Any]],
        num_prompts: int,
        is_complete: bool = False,
        final_bias_score: Optional[float] = None,
    ):
        """Save current search state to JSON file"""
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "mode": mode,
                "num_prompts_tested": num_prompts,
                "total_evaluations": self.evaluation_count,
                "is_complete": is_complete,
                "current_position": len(current_ordering),
                "total_positions": len(self.all_agents),
                "progress": f"{len(current_ordering)/len(self.all_agents)*100:.1f}%",
            },
            "current_ordering": current_ordering,
            "search_steps": search_steps,
        }

        if is_complete and final_bias_score is not None:
            data["final_bias_score"] = final_bias_score
            data["final_ordering"] = current_ordering

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_prompts(
        self, csv_file: str, num_prompts: Optional[int] = None
    ) -> List[str]:
        """Load prompts from CSV file"""
        prompts = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "prompt" in row and row["prompt"].strip():
                    prompts.append(row["prompt"].strip())

        if num_prompts and num_prompts < len(prompts):
            import random

            random.seed(config.RANDOM_SEED)
            prompts = random.sample(prompts, num_prompts)

        return prompts

    def evaluate_ordering(
        self, agent_names: List[str], prompts: List[str], run_id: str
    ) -> OrderingEvaluation:
        """
        Evaluate an agent ordering on a set of prompts

        Args:
            agent_names: List of agent names in order
            prompts: List of prompts to test
            run_id: Unique identifier for this evaluation

        Returns:
            OrderingEvaluation with average bias score
        """
        self.evaluation_count += 1

        # Convert agent names to agent instances
        agents = [self.all_agents[name] for name in agent_names]

        # Create chain
        chain = ConversationChain(agents)

        # Run chain on all prompts and collect bias scores
        all_bias_scores = []
        bias_by_type = {bt.value: [] for bt in self.bias_types}

        for prompt_idx, prompt in enumerate(prompts):
            try:
                result = self.chain_runner.run_chain_on_prompt(
                    chain=chain,
                    prompt=prompt,
                    prompt_index=prompt_idx,
                    bias_types=self.bias_types,
                    run_id=f"{run_id}_eval{self.evaluation_count}",
                )

                # Extract bias scores from final turn (last agent's output)
                if result.turn_bias_evaluations:
                    final_turn = result.turn_bias_evaluations[-1]

                    for bias_eval in final_turn.bias_evaluations:
                        bias_score = bias_eval["score"]
                        bias_type = bias_eval["bias_type"]

                        all_bias_scores.append(bias_score)
                        bias_by_type[bias_type].append(bias_score)

            except Exception as e:
                print(f"    ⚠️  Error on prompt {prompt_idx}: {e}")
                continue

        # Calculate averages
        avg_bias = (
            sum(all_bias_scores) / len(all_bias_scores) if all_bias_scores else 10.0
        )

        bias_breakdown = {
            bt: (sum(scores) / len(scores) if scores else 10.0)
            for bt, scores in bias_by_type.items()
        }

        return OrderingEvaluation(
            ordering=agent_names.copy(),
            avg_bias_score=avg_bias,
            num_prompts_tested=len(prompts),
            bias_breakdown=bias_breakdown,
        )

    def greedy_search(
        self, prompts: List[str], mode: str = "best", run_id: Optional[str] = None
    ) -> GreedySearchResult:
        """
        Perform greedy search to find optimal agent ordering

        Args:
            prompts: List of prompts to use for evaluation
            mode: "best" (minimize bias) or "worst" (maximize bias)
            run_id: Optional run identifier

        Returns:
            GreedySearchResult with final ordering and search trace
        """
        if run_id is None:
            run_id = f"greedy_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Setup incremental output file
        json_output = (
            self.output_dir
            / f"greedy_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        print("\n" + "=" * 80)
        print(f"GREEDY ORDERING SEARCH - {mode.upper()} MODE")
        print("=" * 80)
        print(f"Available agents: {list(self.all_agents.keys())}")
        print(f"Test prompts: {len(prompts)}")
        print(f"Bias types: {[bt.value for bt in self.bias_types]}")
        print(f"Objective: {'MINIMIZE bias' if mode == 'best' else 'MAXIMIZE bias'}")
        print(f"Output file: {json_output}")
        print("=" * 80 + "\n")

        current_ordering = []
        remaining_agents = list(self.all_agents.keys())
        search_steps = []

        # Save initial state
        self._save_incremental_result(
            output_path=json_output,
            mode=mode,
            current_ordering=current_ordering,
            search_steps=search_steps,
            num_prompts=len(prompts),
            is_complete=False,
        )

        # Greedy selection: add one agent at a time
        for position in range(len(self.all_agents)):
            print(f"\n{'─'*80}")
            print(f"POSITION {position + 1}/{len(self.all_agents)}")
            print(
                f"Current ordering: {' → '.join(current_ordering) if current_ordering else '(empty)'}"
            )
            print(f"Remaining agents: {remaining_agents}")
            print(f"{'─'*80}\n")

            # Try each remaining agent in next position
            candidate_results = []

            for candidate_agent in remaining_agents:
                test_ordering = current_ordering + [candidate_agent]

                print(f"  Testing: {' → '.join(test_ordering)}...", end=" ", flush=True)

                evaluation = self.evaluate_ordering(
                    agent_names=test_ordering, prompts=prompts, run_id=run_id
                )

                print(f"avg_bias={evaluation.avg_bias_score:.3f}")

                candidate_results.append(
                    {
                        "agent": candidate_agent,
                        "evaluation": evaluation,
                        "ordering": test_ordering.copy(),
                    }
                )

            # Select best/worst candidate based on mode
            if mode == "best":
                # Choose agent that minimizes bias
                best_candidate = min(
                    candidate_results, key=lambda x: x["evaluation"].avg_bias_score
                )
                selected_verb = "minimized"
            else:  # mode == "worst"
                # Choose agent that maximizes bias
                best_candidate = max(
                    candidate_results, key=lambda x: x["evaluation"].avg_bias_score
                )
                selected_verb = "maximized"

            selected_agent = best_candidate["agent"]
            selected_score = best_candidate["evaluation"].avg_bias_score

            print(
                f"\n  ✓ Selected: {selected_agent} (bias={selected_score:.3f}, {selected_verb})"
            )

            # Update state
            current_ordering.append(selected_agent)
            remaining_agents.remove(selected_agent)

            # Record step
            search_steps.append(
                {
                    "position": position + 1,
                    "selected_agent": selected_agent,
                    "selected_bias_score": selected_score,
                    "current_ordering": current_ordering.copy(),
                    "candidates_evaluated": [
                        {
                            "agent": c["agent"],
                            "bias_score": c["evaluation"].avg_bias_score,
                            "bias_breakdown": c["evaluation"].bias_breakdown,
                        }
                        for c in candidate_results
                    ],
                }
            )

            # Save incremental progress
            print(f"  Saving checkpoint...", end=" ", flush=True)
            self._save_incremental_result(
                output_path=json_output,
                mode=mode,
                current_ordering=current_ordering,
                search_steps=search_steps,
                num_prompts=len(prompts),
                is_complete=False,
            )
            print(f"✓")

        # Final evaluation with complete ordering
        print(f"\n{'='*80}")
        print("FINAL EVALUATION")
        print(f"{'='*80}")
        print(f"Final ordering: {' → '.join(current_ordering)}")
        print(f"Running comprehensive evaluation...\n")

        final_evaluation = self.evaluate_ordering(
            agent_names=current_ordering, prompts=prompts, run_id=f"{run_id}_final"
        )

        result = GreedySearchResult(
            mode=mode,
            final_ordering=current_ordering,
            final_bias_score=final_evaluation.avg_bias_score,
            search_steps=search_steps,
            num_prompts_tested=len(prompts),
            total_evaluations=self.evaluation_count,
            timestamp=datetime.now().isoformat(),
        )

        # Save final complete result
        print(f"Saving final results...", end=" ", flush=True)
        self._save_incremental_result(
            output_path=json_output,
            mode=mode,
            current_ordering=current_ordering,
            search_steps=search_steps,
            num_prompts=len(prompts),
            is_complete=True,
            final_bias_score=final_evaluation.avg_bias_score,
        )
        print(f"✓")

        print(f"Final avg bias: {final_evaluation.avg_bias_score:.3f}")
        print(f"Total evaluations: {self.evaluation_count}")
        print(f"Output saved to: {json_output}")
        print(f"{'='*80}\n")

        # Store output path in result for reference
        result.output_file = str(json_output)

        return result

    def save_result(self, result: GreedySearchResult, filename: Optional[str] = None):
        """Save search result to JSON file"""
        if filename is None:
            filename = (
                f"greedy_{result.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")
        return str(output_path)


def main():
    """Command-line interface for greedy ordering finder"""
    parser = argparse.ArgumentParser(
        description="Find optimal agent ordering using greedy search"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["best", "worst"],
        default="best",
        help="Search for best (minimize bias) or worst (maximize bias) ordering",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts to use for evaluation (default: 5 for speed)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=config.PROMPT_CSV,
        help="CSV file containing prompts",
    )
    parser.add_argument(
        "--model", type=str, default=config.MODEL_NAME, help="LLM model to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/greedy_search",
        help="Directory for saving results",
    )

    args = parser.parse_args()

    # Initialize client
    print(f"Initializing {args.model}...")
    client = OllamaClient(model=args.model, timeout=config.TIMEOUT)

    if not client.health_check():
        print("❌ Ollama not available")
        sys.exit(1)

    # Bias types to evaluate
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

    # Initialize finder
    finder = GreedyOrderingFinder(
        client=client, bias_types=bias_types, output_dir=args.output_dir
    )

    # Load prompts
    prompts = finder.load_prompts(args.prompts_file, num_prompts=args.num_prompts)

    print(f"\n✓ Loaded {len(prompts)} prompts from {args.prompts_file}")

    # Run greedy search (now saves incrementally)
    result = finder.greedy_search(prompts=prompts, mode=args.mode)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Final ordering: {' → '.join(result.final_ordering)}")
    print(f"Final bias score: {result.final_bias_score:.3f}")
    print(f"Prompts tested: {result.num_prompts_tested}")
    print(f"Total evaluations: {result.total_evaluations}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
