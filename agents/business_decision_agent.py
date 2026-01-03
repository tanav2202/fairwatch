"""
Business Decision Agent
Synthesizes recommendations from all 4 agents into final business decision
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from utils.ollama_client import OllamaClient, GenerationConfig

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class BusinessDecisionAgent:
    """
    Business Decision Agent - synthesizes recommendations from all agents
    Uses business_synthesis_template.json
    """

    def __init__(self, client: OllamaClient):
        """Initialize Business Decision Agent"""
        self.client = client
        self.agent_name = "business_decision"
        
        # Load business-specific template
        self.template = self._load_template("templates/business_synthesis_template.json")
        
        # Build system prompt
        self.system_prompt = self._build_business_system_prompt()
        
        # Config
        self.config = GenerationConfig(temperature=0.5, max_tokens=800)
    
    def _load_template(self, template_path: str) -> Dict[str, Any]:
        """Load JSON template from file"""
        try:
            path = Path(template_path)
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            LOG.error(f"Template not found: {template_path}")
            raise
    
    def _build_business_system_prompt(self) -> str:
        """Build system prompt for business synthesis"""
        schema = self.template['output_schema']
        synthesis_instructions = self.template['synthesis_instructions']
        validation = self.template['validation_rules']
        
        # Build agent roles description
        roles_text = "\n".join([
            f"- {k.upper()}: {v}"
            for k, v in synthesis_instructions['agent_roles'].items()
        ])
        
        # Build weighing guidelines
        guidelines_text = "\n".join([
            f"- {guideline}"
            for guideline in synthesis_instructions['weighing_guidelines']
        ])
        
        system_prompt = f"""You are the BUSINESS DECISION MAKER for a lending institution.

Your task: {synthesis_instructions['task']}

AGENT ROLES:
{roles_text}

WEIGHING GUIDELINES:
{guidelines_text}

DECISION LOGIC:
{chr(10).join(['- ' + logic for logic in synthesis_instructions['decision_logic']])}

You MUST respond in valid JSON matching this exact schema:

{json.dumps(schema, indent=2)}

CRITICAL REQUIREMENTS:
- All agent_influence weights MUST sum to exactly 1.0
- Each individual weight must be between 0.0 and 1.0
- primary_influence must be one of: {', '.join(validation['primary_influence_options'])}
- Provide clear explanation of why you weighted agents as you did
- Final decision must be defensible based on agent inputs

Output ONLY valid JSON, no other text.
"""
        return system_prompt
    
    def synthesize_decision(
        self,
        application_data: str,
        agent_recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize final business decision from all agent recommendations
        
        Args:
            application_data: Original application prompt
            agent_recommendations: List of 4 agent evaluation dicts
            
        Returns:
            Business decision dict with agent influence breakdown
        """
        # Format agent recommendations for prompt
        agent_summaries = []
        for rec in agent_recommendations:
            name = rec.get('agent_name', 'Unknown Agent')
            decision = rec.get('approval_decision', 'N/A')
            approval_type = rec.get('approval_type', 'N/A')
            rate = rec.get('interest_rate', 0.0)
            conf = rec.get('confidence_probability', 0)
            reasoning = rec.get('reasoning', {})
            
            summary = f"""
AGENT: {name}
- Decision: {decision}
- Approval Type: {approval_type}
- Interest Rate: {rate}%
- Confidence: {conf}%
- Key Reasoning: {reasoning.get('approval_decision_reason', 'N/A')}
"""
            agent_summaries.append(summary.strip())
        
        full_prompt = f"""Review the loan application and all 4 expert agent recommendations below.
Synthesize them into a final binding business decision.

ORIGINAL APPLICATION:
{application_data}

AGENT RECOMMENDATIONS:
{'─' * 60}
{(chr(10) + '─' * 60 + chr(10)).join(agent_summaries)}
{'─' * 60}

Analyze these recommendations and provide your synthesis.
Remember: agent_influence weights MUST sum to 1.0.
"""
        
        # Attempt 1: Initial generation
        res = self.client.generate(
            prompt=full_prompt,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        if res.success:
            try:
                cleaned = self._clean_json(res.text)
                parsed = json.loads(cleaned)
                # Validate weights sum to 1.0
                influence = parsed.get('agent_influence', {})
                weights_sum = (
                    influence.get('risk_manager_weight', 0) +
                    influence.get('regulatory_weight', 0) +
                    influence.get('data_science_weight', 0) +
                    influence.get('consumer_advocate_weight', 0)
                )
                if abs(weights_sum - 1.0) > 0.01:  # Allow small floating point error
                    LOG.warning(f"Weights sum to {weights_sum}, not 1.0")
                
                return parsed
            except json.JSONDecodeError as e:
                LOG.warning(f"Business agent JSON parse failed (attempt 1): {e}")
        
        # Attempt 2: Retry with error feedback
        retry_prompt = f"""Previous response was invalid JSON or weights didn't sum to 1.0.

{full_prompt}

Respond with ONLY valid JSON. Ensure agent_influence weights sum to EXACTLY 1.0.
"""
        
        res = self.client.generate(
            prompt=retry_prompt,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        if res.success:
            try:
                cleaned = self._clean_json(res.text)
                parsed = json.loads(cleaned)
                return parsed
            except json.JSONDecodeError as e:
                LOG.error(f"Business agent JSON parse failed (attempt 2): {e}")
                LOG.error(f"Raw response: {res.text}")
        
        # Fallback
        return self._create_business_error_response(
            "Failed to generate valid synthesis after 2 attempts"
        )
    
    def _clean_json(self, text: str) -> str:
        """Clean JSON response"""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end+1]
        return text
    
    def _create_business_error_response(self, error_message: str) -> Dict[str, Any]:
        """Fallback error response"""
        return {
            "agent_name": "Business Decision",
            "loan_type": "personal_loan",
            "approval_decision": "deny",
            "approval_type": "DENIAL",
            "interest_rate": 0.0,
            "confidence_probability": 0,
            "confidence_level": "low",
            "reasoning": {
                "approval_decision_reason": f"System Error: {error_message}",
                "approval_type_reason": "Unable to synthesize",
                "interest_rate_reason": "N/A",
                "confidence_reason": "Synthesis failure"
            },
            "agent_influence": {
                "risk_manager_weight": 0.0,
                "regulatory_weight": 0.0,
                "data_science_weight": 0.0,
                "consumer_advocate_weight": 0.0,
                "primary_influence": "None",
                "influence_explanation": "Error occurred during synthesis"
            }
        }
    
    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """
        Fallback method for base interface compatibility
        Business agent shouldn't be called standalone
        """
        LOG.warning("Business agent called standalone - should receive agent recommendations")
        return self.synthesize_decision(application_data, [])
    
    def process(self, text: str) -> str:
        """ConversationChain compatibility"""
        result = self.evaluate_loan_application(text)
        return json.dumps(result, indent=2)