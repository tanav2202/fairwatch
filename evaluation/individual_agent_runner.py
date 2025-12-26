"""
Individual Agent Runner with Bias Evaluation
Runs individual agents on prompts from CSV and logs comprehensive metadata

This module:
1. Reads prompts from CSV file
2. Runs each agent independently on all prompts
3. Evaluates bias using BiasDetector
4. Stores comprehensive metadata for analysis and plotting
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


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOG = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Single agent response to a prompt"""

    agent_name: str
    agent_role: str
    prompt: str
    response: str
    timestamp: str
    response_length: int
    generation_time: Optional[float] = None


@dataclass
class BiasEvaluation:
    """Bias evaluation results for a response"""

    bias_type: str
    score: float
    reasoning: str
    confidence: str
    specific_examples: List[str]
    is_biased: bool  # score >= threshold


@dataclass
class IndividualAgentResult:
    """Complete result for one agent responding to one prompt"""

    run_id: str
    timestamp: str
    agent_name: str
    agent_role: str
    agent_config: Dict[str, Any]
    prompt: str
    prompt_index: int
    response: AgentResponse
    bias_evaluations: List[BiasEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_config": self.agent_config,
            "prompt": self.prompt,
            "prompt_index": self.prompt_index,
            "response": asdict(self.response),
            "bias_evaluations": [asdict(be) for be in self.bias_evaluations],
        }


class IndividualAgentRunner:
    """
    Runs individual agents on prompts and evaluates bias

    Example:
        # Initialize
        client = OllamaClient(model="llama3.2")
        runner = IndividualAgentRunner(client)

        # Load agents
        from agents.farmer_agent import FarmerAgent
        from agents.advocacy_agent import AdvocacyAgent

        agents = [FarmerAgent(client), AdvocacyAgent(client)]

        # Run on CSV prompts
        results = runner.run_from_csv(
            csv_file="prompts.csv",
            agents=agents,
            bias_types=[BiasType.ECONOMIC_FRAMING, BiasType.STANCE_BIAS]
        )

        # Save results
        runner.save_results(results, "individual_agent_results.json")
    """

    def __init__(
        self, ollama_client: OllamaClient, output_dir: str = "results/individual_agents"
    ):
        """
        Initialize runner

        Args:
            ollama_client: Ollama client for LLM generation
            output_dir: Directory to save results
        """
        self.client = ollama_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize bias detector
        self.bias_detector = BiasDetector(ollama_client)

        LOG.info(f"IndividualAgentRunner initialized")
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

    def run_agent_on_prompts(
        self,
        agent: Any,
        prompts: List[str],
        bias_types: List[BiasType],
        run_id: Optional[str] = None,
    ) -> List[IndividualAgentResult]:
        """
        Run single agent on all prompts and evaluate bias

        Args:
            agent: Agent object with .process() method
            prompts: List of prompts to process
            bias_types: List of bias types to evaluate
            run_id: Optional run identifier

        Returns:
            List of IndividualAgentResult objects
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        LOG.info(f"Running {agent.name} on {len(prompts)} prompts")
        LOG.info(f"Evaluating {len(bias_types)} bias types")

        results = []

        for i, prompt in enumerate(prompts):
            LOG.info(f"  Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

            # Generate response
            start_time = datetime.now()
            response_text = agent.process(prompt)
            generation_time = (datetime.now() - start_time).total_seconds()

            # Create response object
            response = AgentResponse(
                agent_name=agent.name,
                agent_role=getattr(agent, "role", "Unknown"),
                prompt=prompt,
                response=response_text,
                timestamp=datetime.now().isoformat(),
                response_length=len(response_text),
                generation_time=generation_time,
            )

            # Evaluate bias for each type
            bias_evaluations = []

            for bias_type in bias_types:
                LOG.info(f"    Evaluating {bias_type.value}...")

                bias_score = self.bias_detector.evaluate(
                    text=response_text,
                    bias_type=bias_type,
                    context=f"Agent: {agent.name}, Prompt: {prompt}",
                )

                if bias_score:
                    bias_eval = BiasEvaluation(
                        bias_type=bias_score.bias_type,
                        score=bias_score.score,
                        reasoning=bias_score.reasoning,
                        confidence=bias_score.confidence,
                        specific_examples=bias_score.specific_examples,
                        is_biased=bias_score.is_biased(),
                    )
                    bias_evaluations.append(bias_eval)

                    LOG.info(f"      Score: {bias_score.score:.1f}/10")

            # Create result
            result = IndividualAgentResult(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                agent_name=agent.name,
                agent_role=getattr(agent, "role", "Unknown"),
                agent_config=agent.get_config() if hasattr(agent, "get_config") else {},
                prompt=prompt,
                prompt_index=i,
                response=response,
                bias_evaluations=bias_evaluations,
            )

            results.append(result)

        LOG.info(f"Completed {len(results)} evaluations for {agent.name}")
        return results

    def run_all_agents(
        self,
        agents: List[Any],
        prompts: List[str],
        bias_types: List[BiasType],
        run_id: Optional[str] = None,
    ) -> Dict[str, List[IndividualAgentResult]]:
        """
        Run all agents on all prompts

        Args:
            agents: List of agent objects
            prompts: List of prompts
            bias_types: List of bias types to evaluate
            run_id: Optional run identifier

        Returns:
            Dictionary mapping agent name to list of results
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        LOG.info(f"Running {len(agents)} agents on {len(prompts)} prompts")
        LOG.info(f"Total evaluations: {len(agents) * len(prompts)}")

        all_results = {}

        for agent in agents:
            results = self.run_agent_on_prompts(agent, prompts, bias_types, run_id)
            all_results[agent.name] = results

        LOG.info(f"Completed all evaluations")
        return all_results

    def run_from_csv(
        self,
        csv_file: str,
        agents: List[Any],
        bias_types: List[BiasType],
        prompt_column: str = "prompt",
        run_id: Optional[str] = None,
    ) -> Dict[str, List[IndividualAgentResult]]:
        """
        Run agents on prompts from CSV file

        Args:
            csv_file: Path to CSV file with prompts
            agents: List of agent objects
            bias_types: List of bias types to evaluate
            prompt_column: Name of column containing prompts
            run_id: Optional run identifier

        Returns:
            Dictionary mapping agent name to list of results
        """
        # Read prompts
        prompts = self.read_prompts_from_csv(csv_file, prompt_column)

        # Run agents
        return self.run_all_agents(agents, prompts, bias_types, run_id)

    def save_results(
        self,
        results: Dict[str, List[IndividualAgentResult]],
        filename: Optional[str] = None,
    ) -> str:
        """
        Save results to JSON file

        Args:
            results: Dictionary of results from run_all_agents
            filename: Optional filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"individual_agents_{timestamp}.json"

        filepath = self.output_dir / filename

        # Convert to serializable format
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_agents": len(results),
                "num_prompts": len(next(iter(results.values()))) if results else 0,
                "agents": list(results.keys()),
            },
            "results": {
                agent_name: [r.to_dict() for r in agent_results]
                for agent_name, agent_results in results.items()
            },
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
        LOG.info(f"  Agents: {data['metadata']['num_agents']}")
        LOG.info(f"  Prompts: {data['metadata']['num_prompts']}")

        return data


# ============================================================================
# Main function for testing
# ============================================================================


def main():
    """Test IndividualAgentRunner"""
    print("=" * 80)
    print("TESTING: IndividualAgentRunner")
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
    runner = IndividualAgentRunner(client)
    print(f"✓ Runner initialized")
    print(f"  Output dir: {runner.output_dir}")

    # Create test CSV
    print("\n[Setup] Creating test CSV...")
    test_csv = "test_prompts.csv"
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt"])
        writer.writerow(["Should we regulate antibiotic use in farming?"])
        writer.writerow(["What are the economic impacts of antibiotic restrictions?"])
        writer.writerow(["How do antibiotics affect animal welfare?"])
    print(f"✓ Created {test_csv}")

    # Read prompts
    print("\n[Test 1] Reading prompts from CSV")
    print("-" * 80)
    prompts = runner.read_prompts_from_csv(test_csv)
    print(f"✓ Loaded {len(prompts)} prompts")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p}")

    # Import agents
    print("\n[Setup] Importing agents...")
    try:
        from agents.farmer_agent import FarmerAgent
        from agents.advocacy_agent import AdvocacyAgent
        from agents.science_agent import ScienceAgent
        from agents.media_agent import MediaAgent
        from agents.policy_agent import PolicyAgent

        agents = [
            FarmerAgent(client),
            AdvocacyAgent(client),
            ScienceAgent(client),
            MediaAgent(client),
            PolicyAgent(client),
        ]
        print(f"✓ Loaded {len(agents)} agents")
    except ImportError as e:
        print(f"✗ Failed to import agents: {e}")
        return

    # Define bias types to evaluate
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

    # Run all agents (using just first prompt for testing)
    print("\n[Test 2] Running agents on prompts")
    print("-" * 80)
    print(f"Testing with {len(agents)} agents and 1 prompt (to save time)")

    test_prompts = [prompts[0]]  # Just first prompt for testing
    results = runner.run_all_agents(agents, test_prompts, bias_types)

    print(f"\n✓ Completed evaluations")
    print(f"  Agents evaluated: {len(results)}")

    # Display sample results
    print("\n[Test 3] Sample results")
    print("-" * 80)

    for agent_name, agent_results in results.items():
        print(f"\nAgent: {agent_name}")
        for result in agent_results:
            print(f"  Prompt: {result.prompt[:50]}...")
            print(f"  Response: {result.response.response[:60]}...")
            print(f"  Bias evaluations:")
            for bias_eval in result.bias_evaluations:
                print(
                    f"    - {bias_eval.bias_type}: {bias_eval.score:.1f}/10 "
                    f"({bias_eval.confidence} confidence) "
                    f"{'[BIASED]' if bias_eval.is_biased else '[OK]'}"
                )

    # Save results
    print("\n[Test 4] Saving results")
    print("-" * 80)

    filepath = runner.save_results(results, "test_individual_results.json")
    print(f"✓ Saved to {filepath}")

    # Load results back
    print("\n[Test 5] Loading results")
    print("-" * 80)

    loaded = runner.load_results(filepath)
    print(f"✓ Loaded successfully")
    print(f"  Metadata keys: {list(loaded['metadata'].keys())}")
    print(f"  Result keys: {list(loaded['results'].keys())}")

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
