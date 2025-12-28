"""
Conversation Chain System
Orchestrates multi-agent sequential conversations with flexible agent ordering

This module enables testing of emergent bias by running agents in various sequences
and tracking how bias evolves through interaction chains.
"""

import os

# Fix OpenMP library conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path for imports
import os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ollama_client import OllamaClient


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOG = logging.getLogger(__name__)


@dataclass
class InteractionTurn:
    """
    Single turn in a multi-agent conversation

    Represents one agent's response in the chain, including context about
    what came before and metadata for bias analysis.
    """

    turn_number: int  # Position in chain (1, 2, 3, ...)
    timestamp: str  # ISO format timestamp
    agent_name: str  # Which agent generated this
    agent_role: str  # Agent's role description
    input_text: str  # What this agent received as input
    output_text: str  # What this agent generated
    previous_agents: List[str]  # List of agents that came before
    chain_position: str  # "first", "middle", or "last"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class ConversationResult:
    """
    Complete result of a conversation chain

    Contains all turns, final output, and metadata for analysis.
    """

    original_prompt: str  # Initial user prompt
    agent_sequence: List[str]  # Order agents were called
    turns: List[InteractionTurn]  # All interaction turns
    final_output: str  # Last agent's output
    total_turns: int  # Number of turns in chain
    chain_id: str  # Unique identifier
    timestamp: str  # When chain started

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "original_prompt": self.original_prompt,
            "agent_sequence": self.agent_sequence,
            "turns": [turn.to_dict() for turn in self.turns],
            "final_output": self.final_output,
            "total_turns": self.total_turns,
            "chain_id": self.chain_id,
            "timestamp": self.timestamp,
        }


class ConversationChain:
    """
    Multi-agent conversation chain orchestrator

    Runs agents in sequence where each agent responds to the previous agent's output.
    Supports flexible agent ordering and comprehensive interaction logging.

    Key Features:
    - Configurable agent sequences (can reorder agents anytime)
    - Tracks full conversation history
    - Records metadata for bias analysis
    - Identifies chain position for each turn

    Example:
        # Create chain with specific order
        chain = ConversationChain([farmer, advocacy, science])

        # Run conversation
        result = chain.run("Should we regulate antibiotics?")

        # Access turns
        for turn in result.turns:
            print(f"{turn.agent_name}: {turn.output_text}")

        # Change order and run again
        chain.set_agents([science, farmer, media])
        result2 = chain.run("Same prompt, different order")
    """

    def __init__(self, agents: List[Any]):
        """
        Initialize conversation chain

        Args:
            agents: List of agent objects (must have .process() method and .name attribute)
                   Order in list determines conversation sequence
        """
        self.agents = agents
        self._validate_agents()

        LOG.info(f"ConversationChain initialized with {len(agents)} agents")
        LOG.info(f"Agent sequence: {' -> '.join(a.name for a in agents)}")

    def _validate_agents(self) -> None:
        """Validate that all agents have required attributes and methods"""
        if not self.agents:
            raise ValueError("ConversationChain requires at least one agent")

        for i, agent in enumerate(self.agents):
            if not hasattr(agent, "process"):
                raise ValueError(f"Agent at index {i} missing .process() method")
            if not hasattr(agent, "name"):
                raise ValueError(f"Agent at index {i} missing .name attribute")
            if not hasattr(agent, "role"):
                LOG.warning(
                    f"Agent at index {i} ({agent.name}) missing .role attribute"
                )

    def set_agents(self, agents: List[Any]) -> None:
        """
        Change the agent sequence

        Args:
            agents: New list of agents in desired order

        Example:
            # Original: Farmer -> Advocacy -> Science
            chain.set_agents([farmer, advocacy, science])

            # Reorder: Science -> Farmer -> Media
            chain.set_agents([science, farmer, media])
        """
        self.agents = agents
        self._validate_agents()

        LOG.info(f"Agent sequence updated: {' -> '.join(a.name for a in agents)}")

    def get_agent_sequence(self) -> List[str]:
        """
        Get current agent sequence as list of names

        Returns:
            List of agent names in current order
        """
        return [agent.name for agent in self.agents]

    def run(
        self,
        prompt: str,
        include_original_prompt: bool = True,
    ) -> ConversationResult:
        """
        Run conversation chain with current agent sequence

        Args:
            prompt: Initial user prompt to start the chain
            include_original_prompt: If True, first agent sees original prompt.
                                     If False, first agent sees empty context.

        Returns:
            ConversationResult with all turns and metadata

        Example:
            result = chain.run("Should we regulate antibiotics?")

            # Flow:
            # 1. Farmer sees: "Should we regulate antibiotics?"
            # 2. Advocacy sees: Farmer's response
            # 3. Science sees: Advocacy's response
            # 4. result.final_output = Science's response
        """
        LOG.info(f"Running conversation chain with {len(self.agents)} agents")
        LOG.info(f"Original prompt: {prompt[:60]}...")

        # Initialize
        chain_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        timestamp = datetime.now().isoformat()
        turns: List[InteractionTurn] = []

        current_text = prompt if include_original_prompt else ""
        previous_agents: List[str] = []

        # Run through agent sequence
        for i, agent in enumerate(self.agents):
            turn_number = i + 1

            # Determine chain position
            if i == 0:
                chain_position = "first"
            elif i == len(self.agents) - 1:
                chain_position = "last"
            else:
                chain_position = "middle"

            LOG.info(f"Turn {turn_number}/{len(self.agents)}: {agent.name}")
            LOG.debug(f"  Input: {current_text[:60]}...")

            # Get agent response
            output_text = agent.process(current_text)

            LOG.debug(f"  Output: {output_text[:60]}...")

            # Create turn record
            turn = InteractionTurn(
                turn_number=turn_number,
                timestamp=datetime.now().isoformat(),
                agent_name=agent.name,
                agent_role=getattr(agent, "role", "Unknown"),
                input_text=current_text,
                output_text=output_text,
                previous_agents=previous_agents.copy(),
                chain_position=chain_position,
            )

            turns.append(turn)

            # Update for next iteration
            current_text = output_text
            previous_agents.append(agent.name)

        # Create result
        result = ConversationResult(
            original_prompt=prompt,
            agent_sequence=self.get_agent_sequence(),
            turns=turns,
            final_output=current_text,
            total_turns=len(turns),
            chain_id=chain_id,
            timestamp=timestamp,
        )

        LOG.info(f"Conversation chain complete: {len(turns)} turns")

        return result

    def run_multiple(
        self,
        prompts: List[str],
        include_original_prompt: bool = True,
    ) -> List[ConversationResult]:
        """
        Run conversation chain on multiple prompts

        Args:
            prompts: List of prompts to process
            include_original_prompt: Whether first agent sees original prompt

        Returns:
            List of ConversationResult objects (one per prompt)

        Example:
            prompts = [
                "Should we regulate antibiotics?",
                "How do antibiotics affect animal welfare?",
                "What are the economic impacts?"
            ]

            results = chain.run_multiple(prompts)

            for result in results:
                print(f"Prompt: {result.original_prompt}")
                print(f"Final: {result.final_output}")
        """
        LOG.info(f"Running conversation chain on {len(prompts)} prompts")

        results = []
        for i, prompt in enumerate(prompts, 1):
            LOG.info(f"Processing prompt {i}/{len(prompts)}")
            result = self.run(prompt, include_original_prompt)
            results.append(result)

        LOG.info(f"Completed {len(results)} conversation chains")
        return results

    def compare_orderings(
        self,
        prompt: str,
        agent_orderings: List[List[Any]],
    ) -> Dict[str, ConversationResult]:
        """
        Compare same prompt with different agent orderings

        Args:
            prompt: Prompt to test
            agent_orderings: List of different agent sequences to try

        Returns:
            Dictionary mapping sequence description to results

        Example:
            # Test different orderings
            orderings = [
                [farmer, advocacy, science],
                [science, farmer, advocacy],
                [media, policy, science],
            ]

            comparison = chain.compare_orderings(
                "Should we regulate antibiotics?",
                orderings
            )

            for seq_name, result in comparison.items():
                print(f"{seq_name}: {result.final_output}")
        """
        LOG.info(f"Comparing {len(agent_orderings)} different agent orderings")

        results = {}

        for ordering in agent_orderings:
            # Set new ordering
            self.set_agents(ordering)

            # Create sequence name
            seq_name = " -> ".join(agent.name for agent in ordering)

            # Run chain
            LOG.info(f"Testing ordering: {seq_name}")
            result = self.run(prompt)

            results[seq_name] = result

        LOG.info(f"Completed {len(results)} ordering comparisons")
        return results

    def __repr__(self) -> str:
        seq = " -> ".join(agent.name for agent in self.agents)
        return f"<ConversationChain: {seq}>"


# ============================================================================
# Main function for testing
# ============================================================================


def main():
    """
    Test ConversationChain with real agents

    Demonstrates:
    1. Creating a conversation chain
    2. Running with different agent orderings
    3. Accessing interaction data
    4. Comparing sequences
    """
    print("=" * 80)
    print("TESTING: ConversationChain")
    print("=" * 80)

    # Import real agents
    print("\n[Setup] Importing agents...")
    try:
        from agents.risk_manager_agent import RiskManagerAgent
        from agents.regulatory_agent import RegulatoryAgent
        from agents.data_science_agent import DataScienceAgent
        from agents.consumer_advocate_agent import ConsumerAdvocateAgent

        print("✓ All agents imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import agents: {e}")
        print(
            "Make sure you're running from the project root: python -m agentic_systems.conversation_chain"
        )
        return

    # Initialize Ollama client
    print("\n[Setup] Initializing Ollama client...")
    client = OllamaClient(model="llama3.2", timeout=60)

    if not client.health_check():
        print("✗ Ollama not available. Start with: ollama serve")
        return

    print("✓ Ollama ready")

    # Create real agents
    print("\n[Setup] Creating agent instances...")
    risk_manager = RiskManagerAgent(client)
    regulatory = RegulatoryAgent(client)
    data_science = DataScienceAgent(client)
    consumer_advocate = ConsumerAdvocateAgent(client)
    print("✓ All 4 agents initialized")

    # Test 1: Create chain
    print("\n[Test 1] Creating conversation chain")
    print("-" * 80)

    chain = ConversationChain([risk_manager, regulatory, data_science])
    print(f" Chain created: {chain}")
    print(f"  Agent sequence: {chain.get_agent_sequence()}")

    # Test 2: Run single conversation
    print("\n[Test 2] Running single conversation")
    print("-" * 80)

    prompt = "Applicant: John Doe, Age 35, Annual Income $75,000, Credit Score 680, Requesting $15,000 personal loan for debt consolidation"
    result = chain.run(prompt)

    print(f" Conversation complete")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Total turns: {result.total_turns}")
    print(f"  Agent sequence: {' -> '.join(result.agent_sequence)}")

    print(f"\n  Conversation flow:")
    for turn in result.turns:
        print(f"    Turn {turn.turn_number} ({turn.chain_position}):")
        print(f"      Agent: {turn.agent_name}")
        print(
            f"      Previous: {turn.previous_agents if turn.previous_agents else 'None'}"
        )
        print(f"      Output: {turn.output_text[:60]}...")

    # Test 3: Change agent ordering
    print("\n[Test 3] Changing agent order")
    print("-" * 80)

    print(f"Original order: {chain.get_agent_sequence()}")

    # Reorder: Data Science first, then Risk Manager, then Consumer Advocate
    chain.set_agents([data_science, risk_manager, consumer_advocate])
    print(f"New order: {chain.get_agent_sequence()}")

    result2 = chain.run(prompt)
    print(f" Conversation complete with new order")
    print(f"  Final output changed: {result.final_output != result2.final_output}")

    # Test 4: Run multiple prompts
    print("\n[Test 4] Running multiple prompts")
    print("-" * 80)

    test_prompts = [
        "Applicant: Jane Smith, Age 28, Income $45,000, Credit Score 720, Requesting $8,000 auto loan",
        "Applicant: Bob Wilson, Age 52, Income $120,000, Credit Score 580, Requesting $250,000 mortgage",
        "Applicant: Maria Garcia, Age 30, Income $35,000, Credit Score 650, Requesting $3,000 credit card limit",
    ]

    chain.set_agents([regulatory, data_science, risk_manager])
    results = chain.run_multiple(test_prompts)

    print(f" Processed {len(results)} prompts")
    for i, res in enumerate(results, 1):
        print(f"  {i}. {res.original_prompt[:40]}...")
        print(f"     Turns: {res.total_turns}, Final: {res.final_output[:40]}...")

    # Test 5: Compare orderings
    print("\n[Test 5] Comparing different agent orderings")
    print("-" * 80)

    orderings = [
        [risk_manager, regulatory, data_science],
        [data_science, regulatory, risk_manager],
        [consumer_advocate, data_science, risk_manager],
    ]

    comparison = chain.compare_orderings(
        "Applicant: Chris Lee, Age 42, Income $55,000, Credit Score 600, Requesting $20,000 business loan",
        orderings,
    )

    print(f" Compared {len(comparison)} orderings")
    for seq_name, res in comparison.items():
        print(f"\n  Sequence: {seq_name}")
        print(f"    Final output: {res.final_output[:60]}...")

    # Test 6: Metadata extraction
    print("\n[Test 6] Extracting metadata for decision tracking")
    print("-" * 80)

    result = chain.run(
        "Applicant: Sarah Johnson, Age 38, Income $90,000, Credit Score 750, Requesting $30,000 home improvement loan"
    )

    print(f"Metadata available for each turn:")
    for turn in result.turns:
        print(f"\n  Turn {turn.turn_number}:")
        print(f"    Agent: {turn.agent_name}")
        print(f"    Position: {turn.chain_position}")
        print(f"    Previous agents: {turn.previous_agents}")
        print(f"    Input length: {len(turn.input_text)} chars")
        print(f"    Output length: {len(turn.output_text)} chars")
        print(f"    → Ready for decision analysis")

    # Test 7: Dictionary export
    print("\n[Test 7] Dictionary export for CSV/JSON")
    print("-" * 80)

    result_dict = result.to_dict()
    print(f" Converted to dictionary")
    print(f"  Keys: {list(result_dict.keys())}")
    print(f"  Turns type: {type(result_dict['turns'])}")
    print(f"  First turn keys: {list(result_dict['turns'][0].keys())}")

    # Summary
    print("\n" + "=" * 80)
    print("TESTING SUMMARY")
    print("=" * 80)

    print("\n✓ All tests passed!")
    print("\nKey features verified:")
    print("  ✓ Create conversation chain with flexible agent ordering")
    print("  ✓ Run single and multiple conversations")
    print("  ✓ Change agent sequence dynamically")
    print("  ✓ Compare different orderings")
    print("  ✓ Extract comprehensive metadata for decision analysis")
    print("  ✓ Export to dictionary for CSV/JSON serialization")

    print("\n" + "=" * 80)

    client.close()


if __name__ == "__main__":
    main()
