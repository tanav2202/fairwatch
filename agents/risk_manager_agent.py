"""
Risk Manager Agent
Loads persona from prompts/risk_manager_persona.txt
Loads templates from templates/ folder
"""

import os
import sys
import logging
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from utils.ollama_client import OllamaClient, GenerationConfig

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class RiskManagerAgent(BaseAgent):
    """Risk Manager Agent for credit risk assessment"""

    def __init__(self, client: OllamaClient):
        """Initialize Risk Manager Agent"""
        super().__init__(client, "risk_manager")
        self.config = GenerationConfig(temperature=0.7, max_tokens=500)

    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """
        Evaluate loan application from risk management perspective
        
        Args:
            application_data: Loan application details
            
        Returns:
            Dict with structured loan decision
        """
        prompt = f"""Evaluate this loan application from your perspective:

{application_data}

Provide your assessment in the required JSON format."""

        result = self.client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=self.config
        )

        if not result.success:
            LOG.error(f"Risk Manager evaluation failed: {result.error_message}")
            return self._create_error_response(result.error_message)

        try:
            return self._parse_json_response(result.text)
        except Exception as e:
            LOG.error(f"Failed to parse Risk Manager response: {e}")
            return self._create_error_response(str(e))
