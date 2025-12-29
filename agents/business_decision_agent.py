"""
Business Decision Agent
Synthesizes recommendations from all 4 agents into final business decision
Loads persona from prompts/business_decision_persona.txt
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from utils.ollama_client import OllamaClient, GenerationConfig

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class BusinessDecisionAgent(BaseAgent):
    """
    Business Decision Agent - synthesizes recommendations from all agents
    """

    def __init__(self, client: OllamaClient):
        """Initialize Business Decision Agent"""
        super().__init__(client, "business_decision")
        self.config = GenerationConfig(temperature=0.6, max_tokens=800)

    def synthesize_decision(
        self,
        application_data: str,
        agent_recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize final business decision from all agent recommendations
        
        Args:
            application_data: Original loan application
            agent_recommendations: List of 4 agent recommendation dicts
            
        Returns:
            Dict with final business decision and agent influence breakdown
        """
        # Load business synthesis template
        synthesis_template = self.templates.get('business_synthesis_template', {})
        
        # Format agent recommendations for prompt
        agent_summaries = []
        for i, rec in enumerate(agent_recommendations, 1):
            agent_summaries.append(f"""
AGENT {i}: {rec.get('agent_name', 'Unknown')}
- Decision: {rec.get('approval_decision', 'N/A')}
- Type: {rec.get('approval_type', 'N/A')}
- Rate: {rec.get('interest_rate', 'N/A')}%
- Confidence: {rec.get('confidence_probability', 0)}% ({rec.get('confidence_level', 'N/A')})
- Reasoning: {rec.get('reasoning', {}).get('approval_decision_reason', 'N/A')}
""")
        
        prompt = f"""You are the BUSINESS DECISION MAKER. Review all agent recommendations and make the final lending decision.

ORIGINAL APPLICATION:
{application_data}

AGENT RECOMMENDATIONS:
{"".join(agent_summaries)}

OUTPUT FORMAT (JSON only):
{json.dumps(synthesis_template.get('output_schema', {}), indent=2)}

SYNTHESIS INSTRUCTIONS:
{json.dumps(synthesis_template.get('synthesis_instructions', {}), indent=2)}

Provide your final business decision with clear agent influence weights that sum to 1.0."""

        result = self.client.generate(
            prompt=prompt,
            system_prompt=self._build_business_system_prompt(),
            config=self.config
        )

        if not result.success:
            LOG.error(f"Business Decision synthesis failed: {result.error_message}")
            return self._create_business_error_response(result.error_message)

        try:
            return self._parse_json_response(result.text)
        except Exception as e:
            LOG.error(f"Failed to parse Business Decision response: {e}")
            return self._create_business_error_response(str(e))

    def _build_business_system_prompt(self) -> str:
        """Build system prompt for business decision agent"""
        synthesis_template = self.templates.get('business_synthesis_template', {})
        
        return f"""{self.persona}

You are synthesizing recommendations from 4 agents:
1. Risk Manager (credit risk assessment)
2. Regulatory Compliance (legal compliance)
3. Data Science (statistical modeling)
4. Consumer Advocate (consumer protection)

Your job is to weigh their inputs and make the final BUSINESS decision that serves the lender's interests while being fair and compliant.

OUTPUT FORMAT:
{json.dumps(synthesis_template.get('output_schema', {}), indent=2)}

CRITICAL: Provide agent_influence weights that sum to 1.0 and explain which agent most influenced your decision."""

    def _create_business_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response for business decision"""
        return {
            "agent_name": "Business Decision",
            "loan_type": "personal_loan",
            "approval_decision": "deny",
            "approval_type": "DENIAL",
            "interest_rate": 15.0,
            "confidence_probability": 0,
            "confidence_level": "low",
            "reasoning": {
                "approval_decision_reason": f"Synthesis error: {error_message}",
                "approval_type_reason": "Unable to synthesize",
                "interest_rate_reason": "Default rate",
                "confidence_reason": "No confidence due to error"
            },
            "agent_influence": {
                "risk_manager_weight": 0.25,
                "regulatory_weight": 0.25,
                "data_science_weight": 0.25,
                "consumer_advocate_weight": 0.25,
                "primary_influence": "None",
                "influence_explanation": "Error prevented synthesis"
            }
        }

    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """Not used for business agent - use synthesize_decision instead"""
        raise NotImplementedError("Use synthesize_decision() for business agent")
