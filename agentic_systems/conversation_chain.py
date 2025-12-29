"""
Conversation Chain Orchestrator
Manages sequential and parallel multi-agent conversations
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

LOG = logging.getLogger(__name__)


class ChainMode(Enum):
    """Execution mode for agent chain"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class ConversationChain:
    """
    Orchestrates multi-agent conversations in sequential or parallel mode
    """
    
    def __init__(self, agents: List[Any], mode: ChainMode = ChainMode.SEQUENTIAL):
        """
        Initialize conversation chain
        
        Args:
            agents: List of agent instances
            mode: Sequential or parallel execution mode
        """
        self.agents = agents
        self.mode = mode
        self.conversation_history = []
    
    def run_sequential(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Run agents sequentially - each sees previous agents' outputs
        
        Args:
            initial_prompt: Initial loan application
            
        Returns:
            Dict with all agent outputs and final result
        """
        self.conversation_history = []
        current_input = initial_prompt
        
        all_outputs = []
        
        for i, agent in enumerate(self.agents):
            LOG.info(f"Running agent {i+1}/{len(self.agents)}: {agent.agent_name}")
            
            # Agent evaluates
            output = agent.evaluate_loan_application(current_input)
            
            # Store output
            turn_data = {
                "turn": i + 1,
                "agent_name": agent.agent_name,
                "input": current_input,
                "output": output
            }
            self.conversation_history.append(turn_data)
            all_outputs.append(output)
            
            # Prepare input for next agent (include context)
            current_input = self._build_next_input(initial_prompt, all_outputs)
        
        return {
            "mode": "sequential",
            "initial_prompt": initial_prompt,
            "conversation_history": self.conversation_history,
            "all_agent_outputs": all_outputs,
            "final_output": all_outputs[-1] if all_outputs else None
        }
    
    def run_parallel(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Run agents in parallel - all see only initial prompt
        
        Args:
            initial_prompt: Initial loan application
            
        Returns:
            Dict with all agent outputs
        """
        self.conversation_history = []
        all_outputs = []
        
        for i, agent in enumerate(self.agents):
            LOG.info(f"Running agent {i+1}/{len(self.agents)}: {agent.agent_name} (parallel)")
            
            # All agents see only initial prompt
            output = agent.evaluate_loan_application(initial_prompt)
            
            turn_data = {
                "turn": i + 1,
                "agent_name": agent.agent_name,
                "input": initial_prompt,
                "output": output
            }
            self.conversation_history.append(turn_data)
            all_outputs.append(output)
        
        return {
            "mode": "parallel",
            "initial_prompt": initial_prompt,
            "conversation_history": self.conversation_history,
            "all_agent_outputs": all_outputs
        }
    
    def run_with_synthesis(
        self, 
        initial_prompt: str, 
        business_agent: Any,
        mode: ChainMode = ChainMode.SEQUENTIAL
    ) -> Dict[str, Any]:
        """
        Run agent chain then synthesize with business decision agent
        
        Args:
            initial_prompt: Initial loan application
            business_agent: Business decision agent instance
            mode: Sequential or parallel
            
        Returns:
            Dict with all outputs plus business synthesis
        """
        # Run agent chain
        if mode == ChainMode.SEQUENTIAL:
            chain_result = self.run_sequential(initial_prompt)
        else:
            chain_result = self.run_parallel(initial_prompt)
        
        # Synthesize with business agent
        LOG.info("Running business decision synthesis")
        business_output = business_agent.synthesize_decision(
            application_data=initial_prompt,
            agent_recommendations=chain_result["all_agent_outputs"]
        )
        
        chain_result["business_decision"] = business_output
        return chain_result
    
    def _build_next_input(self, initial_prompt: str, previous_outputs: List[Dict]) -> str:
        """
        Build input for next agent including previous context
        
        Args:
            initial_prompt: Original application
            previous_outputs: List of previous agent outputs
            
        Returns:
            Formatted input string
        """
        context_parts = [f"ORIGINAL APPLICATION:\n{initial_prompt}\n"]
        
        context_parts.append("\nPREVIOUS AGENT EVALUATIONS:")
        for i, output in enumerate(previous_outputs, 1):
            agent_name = output.get('agent_name', f'Agent {i}')
            decision = output.get('approval_decision', 'N/A')
            approval_type = output.get('approval_type', 'N/A')
            rate = output.get('interest_rate', 'N/A')
            reasoning = output.get('reasoning', {}).get('approval_decision_reason', 'N/A')
            
            context_parts.append(f"""
Agent {i} ({agent_name}):
- Decision: {decision}
- Type: {approval_type}
- Rate: {rate}%
- Reasoning: {reasoning}
""")
        
        context_parts.append("\nNow provide YOUR evaluation based on the above information.")
        
        return "\n".join(context_parts)
    
    def get_turn_by_turn_evolution(self) -> List[Dict[str, Any]]:
        """
        Extract turn-by-turn evolution of recommendations
        
        Returns:
            List of dicts with turn-level data
        """
        evolution = []
        for turn in self.conversation_history:
            evolution.append({
                "turn": turn["turn"],
                "agent": turn["agent_name"],
                "decision": turn["output"].get("approval_decision"),
                "approval_type": turn["output"].get("approval_type"),
                "rate": turn["output"].get("interest_rate"),
                "confidence": turn["output"].get("confidence_probability")
            })
        return evolution
    
    def reset(self):
        """Clear conversation history"""
        self.conversation_history = []


class PairwiseComparison:
    """
    Handles pairwise comparisons for Elo rating
    """
    
    def __init__(self, agents: List[Any]):
        """
        Initialize pairwise comparison
        
        Args:
            agents: List of agent instances
        """
        self.agents = agents
    
    def compare_all_pairs(
        self, 
        applicants: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run pairwise comparisons for all applicant pairs
        
        Args:
            applicants: List of applicant profile strings
            
        Returns:
            Dict mapping agent names to their comparison results
        """
        results = {}
        
        for agent in self.agents:
            agent_comparisons = []
            
            # Compare all pairs
            for i in range(len(applicants)):
                for j in range(i + 1, len(applicants)):
                    LOG.info(f"{agent.agent_name}: Comparing applicant {i} vs {j}")
                    
                    comparison = agent.evaluate_pairwise_comparison(
                        applicant_a=applicants[i],
                        applicant_b=applicants[j]
                    )
                    
                    agent_comparisons.append({
                        "pair": (i, j),
                        "applicant_a_index": i,
                        "applicant_b_index": j,
                        "chosen": comparison.get("chosen_applicant"),
                        "confidence": comparison.get("confidence_probability"),
                        "reasoning": comparison.get("reasoning")
                    })
            
            results[agent.agent_name] = agent_comparisons
        
        return results
    
    def compare_single_pair(
        self,
        agent: Any,
        applicant_a: str,
        applicant_b: str
    ) -> Dict[str, Any]:
        """
        Compare single pair of applicants with one agent
        
        Args:
            agent: Agent instance
            applicant_a: First applicant profile
            applicant_b: Second applicant profile
            
        Returns:
            Comparison result
        """
        return agent.evaluate_pairwise_comparison(
            applicant_a=applicant_a,
            applicant_b=applicant_b
        )