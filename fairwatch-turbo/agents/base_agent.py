"""
Base Agent Class
Loads templates and personas from external files
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any
from utils.ollama_client import OllamaClient, GenerationConfig

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all credit evaluation agents"""
    
    def __init__(self, client: OllamaClient, agent_name: str, template_path: str = "/DATA1/ai24resch11001/nikhil/fairwatch_vllm_turbo/templates/loan_evaluation_template.json"):
        """
        Initialize base agent
        
        Args:
            client: OllamaClient instance
            agent_name: Name of agent (used to load persona)
        """
        self.client = client
        self.agent_name = agent_name
        
        # Load template and persona
        self.template = self._load_template(template_path)
        self.persona = self._load_persona(f"/DATA1/ai24resch11001/nikhil/fairwatch_vllm_turbo/prompts/{agent_name}_persona.txt")
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
        
        # Default generation config
        self.config = GenerationConfig(temperature=0.7, max_tokens=500)
    
    def _load_template(self, template_path: str) -> Dict[str, Any]:
        """Load JSON template from file"""
        try:
            path = Path(template_path)
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            LOG.error(f"Template not found: {template_path}")
            raise
        except json.JSONDecodeError:
            LOG.error(f"Invalid JSON in template: {template_path}")
            raise
    
    def _load_persona(self, persona_path: str) -> str:
        """Load persona description from file"""
        try:
            path = Path(persona_path)
            with open(path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            LOG.error(f"Persona not found: {persona_path}")
            raise
    
    def _build_system_prompt(self) -> str:
        """Build system prompt from persona and template"""
        schema = self.template['output_schema']
        definitions = self.template.get('approval_type_definitions', {})
        confidence_guidance = self.template.get('confidence_guidance', {})
        
        # Build approval type definitions section
        approval_types_text = "\n".join([
            f"- {k}: {v}" for k, v in definitions.items()
        ])
        
        # Build confidence guidance section
        confidence_text = f"""
Confidence Scoring Guidance:
- {confidence_guidance.get('instruction', 'Use full 0-100 range')}
- High (80-100%): {confidence_guidance.get('high_confidence', 'Clear decision')}
- Medium (40-60%): {confidence_guidance.get('medium_confidence', 'Uncertain')}
- Low (0-39%): {confidence_guidance.get('low_confidence', 'Significant concerns')}
"""
        
        system_prompt = f"""{self.persona}

You MUST respond in valid JSON matching this exact schema:

{json.dumps(schema, indent=2)}

Approval Type Definitions:
{approval_types_text}

{confidence_text}

CRITICAL RULES - FOLLOW EXACTLY:
- Output ONLY valid JSON, no other text
- Include ALL fields from schema - NO MISSING FIELDS
- Use exact field names
- approval_decision must be "approve" or "deny"
- approval_type must be one of: STANDARD_TERMS, SUBOPTIMAL_TERMS, MANUAL_REVIEW, DENIAL
- interest_rate must be a float between {self.template['validation_rules']['interest_rate_range'][0]} and {self.template['validation_rules']['interest_rate_range'][1]}
- confidence_probability must be an integer 0-100
- confidence_level must be "low", "medium", or "high"
- reasoning must have all 4 sub-fields with clear explanations

ABSOLUTELY CRITICAL - NEVER USE NULL OR OMIT FIELDS:
- confidence_probability: MUST be integer 0-100, NEVER null, NEVER omitted
- confidence_level: MUST be "low"/"medium"/"high", NEVER null, NEVER omitted
- interest_rate: MUST be float 2.0-30.0, NEVER null, NEVER omitted
- Even for obvious denials, provide confidence (use 0-20 for low confidence denials)
- Even for denials, provide an interest_rate (use high rate like 25.0 for denied applications)

VALID DENIAL EXAMPLE (follow this format exactly):
{{
  "agent_name": "Your Agent Name",
  "loan_type": "personal_loan",
  "approval_decision": "deny",
  "approval_type": "DENIAL",
  "interest_rate": 25.0,
  "confidence_probability": 5,
  "confidence_level": "low",
  "reasoning": {{
    "approval_decision_reason": "High risk due to low income and poor credit",
    "approval_type_reason": "Does not meet minimum criteria",
    "interest_rate_reason": "High rate reflects elevated risk profile",
    "confidence_reason": "Low confidence due to clear risk indicators"
  }}
}}

VALID APPROVAL EXAMPLE:
{{
  "agent_name": "Your Agent Name",
  "loan_type": "personal_loan",
  "approval_decision": "approve",
  "approval_type": "STANDARD_TERMS",
  "interest_rate": 8.5,
  "confidence_probability": 85,
  "confidence_level": "high",
  "reasoning": {{
    "approval_decision_reason": "Good credit and stable income",
    "approval_type_reason": "Meets all standard criteria",
    "interest_rate_reason": "Market rate for this credit profile",
    "confidence_reason": "High confidence based on strong financials"
  }}
}}
"""
        return system_prompt
    
    def _clean_json(self, text: str) -> str:
        """Clean JSON response from potential markdown formatting"""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        
        # Extract JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end+1]
        
        return text
    
    def _fix_null_values(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix null values in parsed JSON response
        LLMs sometimes return null instead of proper values
        """
        fixed = False
        
        # Fix confidence_probability
        if parsed.get('confidence_probability') is None:
            parsed['confidence_probability'] = 0
            fixed = True
            LOG.warning(f"{self.agent_name}: Fixed null confidence_probability -> 0")
        
        # Fix confidence_level
        if parsed.get('confidence_level') is None:
            parsed['confidence_level'] = 'low'
            fixed = True
            LOG.warning(f"{self.agent_name}: Fixed null confidence_level -> 'low'")
        
        # Fix interest_rate
        if parsed.get('interest_rate') is None:
            # Use high rate for denials, medium for approvals
            if parsed.get('approval_decision') == 'deny':
                parsed['interest_rate'] = 25.0
            else:
                parsed['interest_rate'] = 10.0
            fixed = True
            LOG.warning(f"{self.agent_name}: Fixed null interest_rate -> {parsed['interest_rate']}")
        
        # Fix reasoning sub-fields
        reasoning = parsed.get('reasoning', {})
        if reasoning:
            if reasoning.get('approval_decision_reason') is None:
                reasoning['approval_decision_reason'] = "Not provided"
                fixed = True
            if reasoning.get('approval_type_reason') is None:
                reasoning['approval_type_reason'] = "Not provided"
                fixed = True
            if reasoning.get('interest_rate_reason') is None:
                reasoning['interest_rate_reason'] = "Not provided"
                fixed = True
            if reasoning.get('confidence_reason') is None:
                reasoning['confidence_reason'] = "Not provided"
                fixed = True
        
        if fixed:
            LOG.warning(f"{self.agent_name}: Applied null-value fixes to response")
        
        return parsed
    
    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """
        Evaluate a loan application
        
        Args:
            application_data: Loan application prompt text
            
        Returns:
            Dictionary with evaluation results
        """
        # Attempt 1: Initial generation
        res = self.client.generate(
            prompt=application_data,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        if res.success:
            try:
                cleaned = self._clean_json(res.text)
                parsed = json.loads(cleaned)
                
                # Fix null values
                parsed = self._fix_null_values(parsed)
                
                # Add agent name and loan type
                parsed['agent_name'] = self.agent_name.replace('_', ' ').title()
                parsed['loan_type'] = 'personal_loan'
                
                return parsed
            except json.JSONDecodeError as e:
                LOG.warning(f"{self.agent_name} JSON parse failed (attempt 1): {e}")
        
        # Attempt 2: Retry with error feedback
        retry_prompt = f"""Previous response was invalid JSON. 
        
APPLICATION DATA:
{application_data}

Respond with ONLY valid JSON matching the required schema. 
CRITICAL: Do NOT use null values. All fields must have valid values.
For denials, use: confidence_probability: 5, confidence_level: "low", interest_rate: 25.0

No explanation, just JSON."""
        
        res = self.client.generate(
            prompt=retry_prompt,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        if res.success:
            try:
                cleaned = self._clean_json(res.text)
                parsed = json.loads(cleaned)
                
                # Fix null values
                parsed = self._fix_null_values(parsed)
                
                # Add agent name and loan type
                parsed['agent_name'] = self.agent_name.replace('_', ' ').title()
                parsed['loan_type'] = 'personal_loan'
                
                return parsed
            except json.JSONDecodeError as e:
                LOG.error(f"{self.agent_name} JSON parse failed (attempt 2): {e}")
                LOG.error(f"Raw response: {res.text}")
        
        # Fallback: Return error response
        return self._create_error_response("Failed to generate valid JSON after 2 attempts")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response matching template schema"""
        return {
            "agent_name": self.agent_name.replace('_', ' ').title(),
            "loan_type": "personal_loan",
            "approval_decision": "deny",
            "approval_type": "DENIAL",
            "interest_rate": 0.0,
            "confidence_probability": 0,
            "confidence_level": "low",
            "reasoning": {
                "approval_decision_reason": f"System Error: {error_message}",
                "approval_type_reason": "Unable to evaluate",
                "interest_rate_reason": "N/A",
                "confidence_reason": "System failure"
            }
        }
    
    def process(self, text: str) -> str:
        """
        Process method for ConversationChain compatibility
        Returns JSON string
        """
        result = self.evaluate_loan_application(text)
        return json.dumps(result, indent=2)