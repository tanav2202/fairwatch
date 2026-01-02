with open('/content/fairwatch-main/agents/business_decision_agent.py', 'w') as f:
    f.write('''"""
Business Decision Agent
Synthesizes recommendations from all 4 agents into final business decision
"""

import json
import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from utils.ollama_client import OllamaClient, GenerationConfig

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

class BusinessDecisionAgent(BaseAgent):
    """
    Business Decision Agent - synthesizes recommendations from all agents
    Inherits robust JSON parsing and connection logic from BaseAgent.
    """

    def __init__(self, client: OllamaClient):
        """Initialize Business Decision Agent"""
        super().__init__(client, "business_decision")
        self.system_prompt = self._build_business_system_prompt()

    def _build_business_system_prompt(self) -> str:
        """Build strict system prompt for business decision agent"""
        return """You are the BUSINESS DECISION MAKER.
You must synthesize inputs from 4 agents (Risk, Regulatory, Data Science, Consumer Advocate) into a FINAL lending decision.

Respond ONLY in valid JSON matching this schema exactly:
{
    "agent_name": "Business Decision",
    "approval_decision": "approve" or "deny",
    "approval_type": "STANDARD_TERMS" or "SUBOPTIMAL_TERMS" or "DENIAL",
    "confidence_probability": integer 0-100,
    "reasoning": "concise explanation of final decision",
    "agent_influence": {
        "risk_manager_weight": float (0.0-1.0),
        "regulatory_weight": float (0.0-1.0),
        "data_science_weight": float (0.0-1.0),
        "consumer_advocate_weight": float (0.0-1.0),
        "primary_influence": "Name of the most influential agent"
    }
}"""

    def synthesize_decision(
        self,
        application_data: str,
        agent_recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize final business decision from all agent recommendations
        """
        
        # Format agent recommendations for prompt
        agent_summaries = []
        for rec in agent_recommendations:
            name = rec.get('agent_name', 'Unknown')
            decision = rec.get('approval_decision', 'N/A')
            conf = rec.get('confidence_probability', 0)
            reason = rec.get('reasoning', 'No reason provided')
            
            agent_summaries.append(f"""
- AGENT: {name}
  Decision: {decision}
  Confidence: {conf}%
  Reasoning: {reason}
""")
        
        full_prompt = f"""Review the loan application and the 4 expert recommendations below.
Make a final binding business decision.

ORIGINAL APPLICATION:
{application_data}

EXPERT RECOMMENDATIONS:
{"".join(agent_summaries)}

Synthesize these views into a final decision. Ensure weights sum to 1.0."""

        # Use BaseAgent's robust generation with retries
        config = GenerationConfig(temperature=0.4, json_mode=True)
        
        # Attempt 1
        res = self.client.generate(full_prompt, self.system_prompt, config)
        if res.success:
            try: return json.loads(self._clean_json(res.text))
            except: pass
        
        # Attempt 2 (Retry with error prompt)
        res = self.client.generate(f"Previous response was invalid JSON. Synthesize these recommendations into valid JSON: {full_prompt}", config=config)
        if res.success:
            try: return json.loads(self._clean_json(res.text))
            except Exception as e:
                LOG.error(f"Business Agent JSON Fail: {res.text}")

        return self._create_business_error_response("Failed to generate valid JSON synthesis")

    def _create_business_error_response(self, error_message: str) -> Dict[str, Any]:
        """Fallback response if synthesis fails"""
        return {
            "agent_name": "Business Decision",
            "approval_decision": "deny",
            "approval_type": "ERROR",
            "confidence_probability": 0,
            "reasoning": f"Synthesis Error: {error_message}",
            "agent_influence": {
                "risk_manager_weight": 0.0,
                "regulatory_weight": 0.0,
                "data_science_weight": 0.0,
                "consumer_advocate_weight": 0.0,
                "primary_influence": "None"
            }
        }

    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """
        Standard interface method. 
        Note: Business agent typically needs other agents' inputs, 
        so this is a fallback that treats it as a standalone agent.
        """
        return self.synthesize_decision(application_data, [])
''')

