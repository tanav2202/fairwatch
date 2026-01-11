import json
import logging
import re
from pathlib import Path
from typing import Dict, Any
from utils.ollama_client import OllamaClient, GenerationConfig

# logging setup
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

class BaseAgent:
    # base class for all the agents in the system
    
    def __init__(self, client: OllamaClient, agent_name: str):
        # setup client and load persona/template
        self.client = client
        self.agent_name = agent_name
        
        # files are expected in templates/ and prompts/ relative to root
        self.template = self._load_template("templates/loan_evaluation_template.json")
        self.persona = self._load_persona(f"prompts/{agent_name}_persona.txt")
        
        # build the full system prompt for initialization
        self.system_prompt = self._build_system_prompt()
        
        # default settings for results
        self.config = GenerationConfig(temperature=0.7, max_tokens=500)
    
    def _load_template(self, template_path: str) -> Dict[str, Any]:
        # load json schema for output
        try:
            path = Path(template_path)
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            LOG.error(f"failed to load template: {e}")
            raise
    
    def _load_persona(self, persona_path: str) -> str:
        # load the persona text file
        try:
            path = Path(persona_path)
            with open(path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            LOG.error(f"failed to load persona: {e}")
            raise
    
    def _build_system_prompt(self) -> str:
        # put together the persona, schema, and rules into one big system prompt
        schema = self.template['output_schema']
        definitions = self.template.get('approval_type_definitions', {})
        confidence_guidance = self.template.get('confidence_guidance', {})
        
        approval_types_text = "\n".join([f"- {k}: {v}" for k, v in definitions.items()])
        
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

VALID DENIAL EXAMPLE:
{{
  "agent_name": "Your Agent Name",
  "loan_type": "personal_loan",
  "approval_decision": "deny",
  "approval_type": "DENIAL",
  "interest_rate": 25.0,
  "confidence_probability": 5,
  "confidence_level": "low",
  "reasoning": {{
    "approval_decision_reason": "reason here",
    "approval_type_reason": "reason here",
    "interest_rate_reason": "reason here",
    "confidence_reason": "reason here"
  }}
}}
"""
        return system_prompt
    
    def _clean_json(self, text: str) -> str:
        # strip markdown formatting if any
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        
        # find the actual brackets
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end+1]
        
        return text
    
    def _fix_null_values(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        # patch any nulls that leaked through
        fixed = False
        
        if parsed.get('confidence_probability') is None:
            parsed['confidence_probability'] = 0
            fixed = True
        
        if parsed.get('confidence_level') is None:
            parsed['confidence_level'] = 'low'
            fixed = True
        
        if parsed.get('interest_rate') is None:
            parsed['interest_rate'] = 25.0 if parsed.get('approval_decision') == 'deny' else 10.0
            fixed = True
        
        if fixed:
            LOG.warning(f"{self.agent_name}: fixed some nulls in output")
        
        return parsed
    
    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        # main evaluation logic with a retry if json fails
        res = self.client.generate(
            prompt=application_data,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        if res.success:
            try:
                cleaned = self._clean_json(res.text)
                parsed = json.loads(cleaned)
                parsed = self._fix_null_values(parsed)
                parsed['agent_name'] = self.agent_name.replace('_', ' ').title()
                parsed['loan_type'] = 'personal_loan'
                return parsed
            except:
                LOG.warning(f"{self.agent_name} failed first try, retrying...")
        
        # retry logic
        retry_prompt = f"Previous response was invalid JSON.\n\nAPPLICATION DATA:\n{application_data}\n\nRespond with ONLY valid JSON."
        
        res = self.client.generate(prompt=retry_prompt, system_prompt=self.system_prompt, config=self.config)
        
        if res.success:
            try:
                cleaned = self._clean_json(res.text)
                parsed = json.loads(cleaned)
                parsed = self._fix_null_values(parsed)
                parsed['agent_name'] = self.agent_name.replace('_', ' ').title()
                parsed['loan_type'] = 'personal_loan'
                return parsed
            except:
                LOG.error(f"{self.agent_name} failed completely.")
        
        return self._create_error_response("Parsing error")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        # fallback response if everything fails
        return {
            "agent_name": self.agent_name.replace('_', ' ').title(),
            "loan_type": "personal_loan",
            "approval_decision": "deny",
            "approval_type": "DENIAL",
            "interest_rate": 0.0,
            "confidence_probability": 0,
            "confidence_level": "low",
            "reasoning": {
                "approval_decision_reason": f"error: {error_message}",
                "approval_type_reason": "failed",
                "interest_rate_reason": "n/a",
                "confidence_reason": "n/a"
            }
        }
    
    def process(self, text: str) -> str:
        # helper for chain compatibility
        result = self.evaluate_loan_application(text)
        return json.dumps(result, indent=2)