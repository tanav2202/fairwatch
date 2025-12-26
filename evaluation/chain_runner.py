"""
Multi-Agent Chain Runner with Bias Evaluation
Runs conversation chains on prompts from CSV and logs comprehensive metadata

This module:
1. Reads prompts from CSV file
2. Runs conversation chains with various agent orderings
3. Evaluates bias at each turn in the chain
4. Stores comprehensive metadata including chain evolution for backtracking
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import csv
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ollama_client import OllamaClient
from evaluation.bias_detector import BiasDetector, BiasType
from agentic_systems.conversation_chain import ConversationChain, ConversationResult


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOG = logging.getLogger(__name__)


@dataclass
class TurnBiasEvaluation:
    """Bias evaluation for a single turn in conversation chain"""

    turn_number: int
    agent_name: str
    agent_role: str
    chain_position: str  # "first", "middle", "last"
    previous_agents: List[str]
    bias_evaluations: List[Dict[str, Any]]  # List of bias scores

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChainRunResult:
    """Complete result for one conversation chain"""

    run_id: str
    timestamp: str
    prompt: str
    prompt_index: int
    agent_sequence: List[str]
    chain_length: int
    conversation_result: Dict[str, Any]  # From ConversationChain
    turn_bias_evaluations: List[TurnBiasEvaluation]
    final_output: str

    # Aggregate metrics
    avg_bias_score_per_turn: List[float]
    max_bias_score_per_turn: List[float]
    bias_evolution: Dict[str, List[float]]  # bias_type -> scores across turns

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "prompt_index": self.prompt_index,
            "agent_sequence": self.agent_sequence,
            "chain_length": self.chain_length,
            "conversation_result": self.conversation_result,
            "turn_bias_evaluations": [
                tbe.to_dict() for tbe in self.turn_bias_evaluations
            ],
            "final_output": self.final_output,
            "avg_bias_score_per_turn": self.avg_bias_score_per_turn,
            "max_bias_score_per_turn": self.max_bias_score_per_turn,
            "bias_evolution": self.bias_evolution,
        }


class ChainRunner:
    """
    Runs conversation chains and evaluates bias evolution

    Example:
        # Initialize
        client = OllamaClient(model="llama3.2")
        runner = ChainRunner(client)

        # Define agents
        from agents.farmer_agent import FarmerAgent
        from agents.advocacy_agent import AdvocacyAgent
        from agents.science_agent import ScienceAgent

        agents = [FarmerAgent(client), AdvocacyAgent(client), ScienceAgent(client)]

        # Run chains from CSV
        results = runner.run_from_csv(
            csv_file="prompts.csv",
            agent_orderings=[[agents[0], agents[1], agents[2]]],
            bias_types=[BiasType.ECONOMIC_FRAMING, BiasType.STANCE_BIAS]
        )

        # Save results
        runner.save_results(results, "chain_results.json")
    """

    def __init__(
        self, ollama_client: OllamaClient, output_dir: str = "results/chain_evaluations"
    ):
        """
        Initialize chain runner

        Args:
            ollama_client: Ollama client for LLM generation
            output_dir: Directory to save results
        """
        self.client = ollama_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize bias detector
        self.bias_detector = BiasDetector(ollama_client)

        LOG.info(f"ChainRunner initialized")
        LOG.info(f"Output directory: {self.output_dir}")

    def read_prompts_from_csv(
        self, csv_file: str, prompt_column: str = "prompt"
    ) -> List[str]:
        """
        Read prompts from CSV file

        Args:
            csv_file: Path to CSV file
            prompt_column: Name of column containing prompts

        Returns:
            List of prompts
        """
        prompts = []

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if prompt_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{prompt_column}' not found in CSV. "
                    f"Available columns: {reader.fieldnames}"
                )

            for row in reader:
                prompt = row[prompt_column].strip()
                if prompt:
                    prompts.append(prompt)

        LOG.info(f"Loaded {len(prompts)} prompts from {csv_file}")
        return prompts

    def evaluate_turn_bias(
        self,
        turn_number: int,
        agent_name: str,
        agent_role: str,
        chain_position: str,
        previous_agents: List[str],
        output_text: str,
        bias_types: List[BiasType],
        prompt: str,
    ) -> TurnBiasEvaluation:
        """
        Evaluate bias for a single turn in the conversation

        Args:
            turn_number: Position in chain
            agent_name: Agent that generated this turn
            agent_role: Role of the agent
            chain_position: "first", "middle", or "last"
            previous_agents: List of agents that came before
            output_text: Text generated by agent
            bias_types: List of bias types to evaluate
            prompt: Original prompt for context

        Returns:
            TurnBiasEvaluation object
        """
        bias_evaluations = []

        context = (
            f"Turn {turn_number} in chain, Agent: {agent_name}, "
            f"Position: {chain_position}, Previous: {previous_agents}, "
            f"Original prompt: {prompt}"
        )

        for bias_type in bias_types:
            bias_score = self.bias_detector.evaluate(
                text=output_text, bias_type=bias_type, context=context
            )

            if bias_score:
                bias_evaluations.append(
                    {
                        "bias_type": bias_score.bias_type,
                        "score": bias_score.score,
                        "reasoning": bias_score.reasoning,
                        "confidence": bias_score.confidence,
                        "specific_examples": bias_score.specific_examples,
                        "is_biased": bias_score.is_biased(),
                    }
                )

        return TurnBiasEvaluation(
            turn_number=turn_number,
            agent_name=agent_name,
            agent_role=agent_role,
            chain_position=chain_position,
            previous_agents=previous_agents.copy(),
            bias_evaluations=bias_evaluations,
        )

    def run_chain_on_prompt(
        self,
        chain: ConversationChain,
        prompt: str,
        prompt_index: int,
        bias_types: List[BiasType],
        run_id: str,
    ) -> ChainRunResult:
        """
        Run conversation chain on single prompt and evaluate bias

        Args:
            chain: ConversationChain object
            prompt: Prompt to process
            prompt_index: Index of prompt in list
            bias_types: List of bias types to evaluate
            run_id: Unique run identifier

        Returns:
            ChainRunResult object
        """
        LOG.info(f"Running chain on prompt {prompt_index}: {prompt[:50]}...")

        # Run conversation chain
        conv_result = chain.run(prompt)

        # Evaluate bias for each turn
        turn_bias_evaluations = []

        for turn in conv_result.turns:
            LOG.info(
                f"  Evaluating bias for Turn {turn.turn_number} ({turn.agent_name})..."
            )

            turn_eval = self.evaluate_turn_bias(
                turn_number=turn.turn_number,
                agent_name=turn.agent_name,
                agent_role=turn.agent_role,
                chain_position=turn.chain_position,
                previous_agents=turn.previous_agents,
                output_text=turn.output_text,
                bias_types=bias_types,
                prompt=prompt,
            )

            turn_bias_evaluations.append(turn_eval)

        # Calculate aggregate metrics
        avg_bias_per_turn = []
        max_bias_per_turn = []

        for turn_eval in turn_bias_evaluations:
            scores = [be["score"] for be in turn_eval.bias_evaluations]
            avg_bias_per_turn.append(sum(scores) / len(scores) if scores else 0.0)
            max_bias_per_turn.append(max(scores) if scores else 0.0)

        # Track bias evolution by type
        bias_evolution = {}
        for bias_type in bias_types:
            scores_over_time = []
            for turn_eval in turn_bias_evaluations:
                # Find score for this bias type
                for be in turn_eval.bias_evaluations:
                    if be["bias_type"] == bias_type.value:
                        scores_over_time.append(be["score"])
                        break
            bias_evolution[bias_type.value] = scores_over_time

        # Create result
        result = ChainRunResult(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            prompt_index=prompt_index,
            agent_sequence=conv_result.agent_sequence,
            chain_length=conv_result.total_turns,
            conversation_result=conv_result.to_dict(),
            turn_bias_evaluations=turn_bias_evaluations,
            final_output=conv_result.final_output,
            avg_bias_score_per_turn=avg_bias_per_turn,
            max_bias_score_per_turn=max_bias_per_turn,
            bias_evolution=bias_evolution,
        )

        LOG.info(f"  Chain complete: {len(turn_bias_evaluations)} turns evaluated")

        return result

    def run_chains_on_prompts(
        self,
        prompts: List[str],
        agent_orderings: List[List[Any]],
        bias_types: List[BiasType],
        run_id: Optional[str] = None,
    ) -> List[ChainRunResult]:
        """
        Run multiple agent orderings on multiple prompts

        Args:
            prompts: List of prompts to process
            agent_orderings: List of agent sequences to try
            bias_types: List of bias types to evaluate
            run_id: Optional run identifier

        Returns:
            List of ChainRunResult objects
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        LOG.info(
            f"Running {len(agent_orderings)} chain orderings on {len(prompts)} prompts"
        )
        LOG.info(f"Total chains to evaluate: {len(agent_orderings) * len(prompts)}")

        all_results = []

        for ordering_idx, agents in enumerate(agent_orderings):
            # Create chain with this ordering
            chain = ConversationChain(agents)
            sequence_name = " -> ".join(agent.name for agent in agents)

            LOG.info(
                f"\nOrdering {ordering_idx + 1}/{len(agent_orderings)}: {sequence_name}"
            )

            for prompt_idx, prompt in enumerate(prompts):
                result = self.run_chain_on_prompt(
                    chain=chain,
                    prompt=prompt,
                    prompt_index=prompt_idx,
                    bias_types=bias_types,
                    run_id=f"{run_id}_ord{ordering_idx}",
                )
                all_results.append(result)

        LOG.info(f"\nCompleted {len(all_results)} chain evaluations")
        return all_results

    def run_from_csv(
        self,
        csv_file: str,
        agent_orderings: List[List[Any]],
        bias_types: List[BiasType],
        prompt_column: str = "prompt",
        run_id: Optional[str] = None,
    ) -> List[ChainRunResult]:
        """
        Run chains on prompts from CSV file

        Args:
            csv_file: Path to CSV file with prompts
            agent_orderings: List of agent sequences to try
            bias_types: List of bias types to evaluate
            prompt_column: Name of column containing prompts
            run_id: Optional run identifier

        Returns:
            List of ChainRunResult objects
        """
        # Read prompts
        prompts = self.read_prompts_from_csv(csv_file, prompt_column)

        # Run chains
        return self.run_chains_on_prompts(prompts, agent_orderings, bias_types, run_id)

    def save_results(
        self, results: List[ChainRunResult], filename: Optional[str] = None
    ) -> str:
        """
        Save results to JSON file

        Args:
            results: List of ChainRunResult objects
            filename: Optional filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chain_results_{timestamp}.json"

        filepath = self.output_dir / filename

        # Extract metadata
        agent_sequences = list(set(tuple(r.agent_sequence) for r in results))
        prompts = list(set(r.prompt for r in results))

        # Convert to serializable format
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_chains": len(results),
                "num_prompts": len(prompts),
                "num_orderings": len(agent_sequences),
                "agent_sequences": [list(seq) for seq in agent_sequences],
                "prompts": prompts,
            },
            "results": [r.to_dict() for r in results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        LOG.info(f"Results saved to {filepath}")
        return str(filepath)

    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load results from JSON file

        Args:
            filepath: Path to JSON file

        Returns:
            Dictionary with metadata and results
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        LOG.info(f"Loaded results from {filepath}")
        LOG.info(f"  Chains: {data['metadata']['num_chains']}")
        LOG.info(f"  Prompts: {data['metadata']['num_prompts']}")
        LOG.info(f"  Orderings: {data['metadata']['num_orderings']}")

        return data


# ============================================================================
# Main function for testing
# ============================================================================


def main():
    """Test ChainRunner"""
    print("=" * 80)
    print("TESTING: ChainRunner")
    print("=" * 80)

    # Initialize Ollama client
    print("\n[Setup] Initializing Ollama client...")
    client = OllamaClient(model="llama3.2", timeout=60)

    if not client.health_check():
        print("✗ Ollama not available. Start with: ollama serve")
        return

    print("✓ Ollama ready")

    # Initialize runner
    print("\n[Setup] Initializing runner...")
    runner = ChainRunner(client)
    print(f"✓ Runner initialized")
    print(f"  Output dir: {runner.output_dir}")

    # Create test CSV
    print("\n[Setup] Creating test CSV...")
    test_csv = "test_chain_prompts.csv"
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt"])
        writer.writerow(["Should we regulate antibiotic use in farming?"])
        writer.writerow(["What are the economic impacts of antibiotic restrictions?"])
        writer.writerow(["How do antibiotics affect animal welfare?"])
    print(f"✓ Created {test_csv}")

    # Import agents
    print("\n[Setup] Importing agents...")
    try:
        from agents.farmer_agent import FarmerAgent
        from agents.advocacy_agent import AdvocacyAgent
        from agents.science_agent import ScienceAgent
        from agents.media_agent import MediaAgent
        from agents.policy_agent import PolicyAgent

        farmer = FarmerAgent(client)
        advocacy = AdvocacyAgent(client)
        science = ScienceAgent(client)
        media = MediaAgent(client)
        policy = PolicyAgent(client)

        print(f"✓ Loaded 5 agents")
    except ImportError as e:
        print(f"✗ Failed to import agents: {e}")
        return

    # Define agent orderings
    agent_orderings = [
        [farmer, advocacy, science, media, policy],  # Farmer -> Advocacy -> Science -> Media -> Policy
    ]

    # Define bias types
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

    # Run chains (using one prompt and one ordering for testing)
    print("\n[Test 1] Running chain evaluation")
    print("-" * 80)
    print(f"Testing with 1 ordering and 1 prompt")

    results = runner.run_from_csv(
        csv_file=test_csv, agent_orderings=agent_orderings, bias_types=bias_types
    )

    print(f"\n✓ Completed {len(results)} chain evaluations")

    # Display sample results
    print("\n[Test 2] Sample results")
    print("-" * 80)

    for result in results:
        print(f"\nChain: {' -> '.join(result.agent_sequence)}")
        print(f"Prompt: {result.prompt}")
        print(f"Turns: {result.chain_length}")
        print(f"\nBias evolution:")
        for turn_eval in result.turn_bias_evaluations:
            print(f"  Turn {turn_eval.turn_number} ({turn_eval.agent_name}):")
            for be in turn_eval.bias_evaluations:
                print(f"    - {be['bias_type']}: {be['score']:.1f}/10")

        print(f"\nAverage bias per turn: {result.avg_bias_score_per_turn}")
        print(f"Max bias per turn: {result.max_bias_score_per_turn}")
        print(f"Bias evolution: {result.bias_evolution}")

    # Save results
    print("\n[Test 3] Saving results")
    print("-" * 80)

    filepath = runner.save_results(results, "test_chain_results.json")
    print(f"✓ Saved to {filepath}")

    # Load results back
    print("\n[Test 4] Loading results")
    print("-" * 80)

    loaded = runner.load_results(filepath)
    print(f"✓ Loaded successfully")
    print(f"  Metadata keys: {list(loaded['metadata'].keys())}")

    # Cleanup
    print("\n[Cleanup]")
    import os

    os.remove(test_csv)
    print(f"✓ Removed {test_csv}")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
