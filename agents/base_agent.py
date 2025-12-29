"""
Base Agent Class with Template and Persona Loading
All agents inherit from this base class
"""

import os
import json
import logging
from typing import Dict, Any
from abc import ABC, abstractmethod

LOG = logging.getLogger(__name__)



class BaseAgent(ABC):
    """Base class for all credit evaluation agents"""

    def __init__(self, client, agent_name: str):
        """
        Initialize base agent
        
        Args:
            client: OllamaClient for LLM interactions
            agent_name: Name of the agent (e.g., "risk_manager", "regulatory")
        """
        self.client = client
        self.agent_name = agent_name
        
        # Load templates
        self.templates = self._load_templates()
        
        # Load persona
        self.persona = self._load_persona()
        
        # Build system prompt from persona
        self.system_prompt = self._build_system_prompt()

    def _load_templates(self) -> Dict[str, Any]:
        """Load all templates from templates/ folder"""
        templates_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "templates"
        )
        
        templates = {}
        template_files = [
            "loan_evaluation_template.json",
            "pairwise_comparison_template.json",
            "business_synthesis_template.json"
        ]
        
        for template_file in template_files:
            path = os.path.join(templates_dir, template_file)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    template_name = template_file.replace('.json', '')
                    templates[template_name] = json.load(f)
            else:
                LOG.warning(f"Template file not found: {path}")
        
        return templates

    def _load_persona(self) -> str:
        """Load agent persona from prompts/ folder"""
        prompts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "prompts"
        )
        
        # Persona filename (e.g., "risk_manager_persona.txt")
        persona_file = f"{self.agent_name}_persona.txt"
        path = os.path.join(prompts_dir, persona_file)
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()
        else:
            LOG.warning(f"Persona file not found: {path}")
            return f"AGENT: {self.agent_name}\nNo persona file found."

    def _build_system_prompt(self) -> str:
        """Build system prompt from persona and template"""
        loan_eval_template = self.templates.get('loan_evaluation_template', {})
        
        prompt = f"""{self.persona}

OUTPUT FORMAT:
You must respond with ONLY valid JSON matching this structure:
{json.dumps(loan_eval_template.get('output_schema', {}), indent=2)}

APPROVAL TYPE DEFINITIONS:
{json.dumps(loan_eval_template.get('approval_type_definitions', {}), indent=2)}

CONFIDENCE GUIDANCE:
{json.dumps(loan_eval_template.get('confidence_guidance', {}), indent=2)}

CRITICAL: Respond ONLY with valid JSON. No preamble, no explanation outside the JSON structure."""

        return prompt

    @abstractmethod
    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """Evaluate loan application - must be implemented by subclass"""
        pass

    def evaluate_pairwise_comparison(
        self, 
        applicant_a: str, 
        applicant_b: str
    ) -> Dict[str, Any]:
        """
        Compare two applicants for pairwise Elo rating
        
        Args:
            applicant_a: First applicant profile
            applicant_b: Second applicant profile
            
        Returns:
            Dict with chosen applicant and reasoning
        """
        pairwise_template = self.templates.get('pairwise_comparison_template', {})
        
        prompt = f"""Compare these two loan applicants and determine which is MORE CREDITWORTHY:

APPLICANT A:
{applicant_a}

APPLICANT B:
{applicant_b}

OUTPUT FORMAT (JSON only):
{json.dumps(pairwise_template.get('output_schema', {}), indent=2)}

INSTRUCTIONS:
{json.dumps(pairwise_template.get('comparison_instruction', {}), indent=2)}

Choose one applicant and provide your reasoning. Use the full confidence range."""

        result = self.client.generate(
            prompt=prompt,
            system_prompt=f"You are {self.agent_name}. Compare applicants from your perspective.",
            config=self.config
        )

        if not result.success:
            return self._create_error_response_pairwise(result.error_message)

        try:
            return self._parse_json_response(result.text)
        except Exception as e:
            return self._create_error_response_pairwise(str(e))

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response"""
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")

        json_str = text[start_idx:end_idx]
        return json.loads(json_str)

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response for loan evaluation"""
        return {
            "agent_name": self.agent_name,
            "loan_type": "personal_loan",
            "approval_decision": "deny",
            "approval_type": "DENIAL",
            "interest_rate": 15.0,
            "confidence_probability": 0,
            "confidence_level": "low",
            "reasoning": {
                "approval_decision_reason": f"Evaluation error: {error_message}",
                "approval_type_reason": "System error prevents evaluation",
                "interest_rate_reason": "Default rate due to error",
                "confidence_reason": "No confidence due to system error"
            }
        }

    def _create_error_response_pairwise(self, error_message: str) -> Dict[str, Any]:
        """Create error response for pairwise comparison"""
        return {
            "agent_name": self.agent_name,
            "chosen_applicant": "applicant_a",
            "confidence_probability": 0,
            "confidence_level": "low",
            "reasoning": {
                "comparison_rationale": f"Comparison error: {error_message}",
                "key_differentiators": "Unable to evaluate",
                "confidence_reason": "No confidence due to system error"
            }
        }

    def process(self, text: str) -> str:
        """
        Process wrapper for ConversationChain compatibility
        
        Args:
            text: Input text (loan application or previous agent's output)
            
        Returns:
            JSON string with loan decision
        """
        result = self.evaluate_loan_application(text)
        return json.dumps(result, indent=2)
