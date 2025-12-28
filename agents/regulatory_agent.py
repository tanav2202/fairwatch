"""
Regulatory/Compliance Agent
Role: Fair lending compliance, legal risk mitigation
Focus: ECOA compliance, disparate impact, documentation requirements
"""

import os
import sys
import json
import logging
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ollama_client import OllamaClient, GenerationConfig

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class RegulatoryAgent:
    """
    Regulatory/Compliance Agent for fair lending compliance

    Evaluates loan applications based on:
    - ECOA (Equal Credit Opportunity Act) compliance
    - Disparate impact analysis
    - Fair lending regulations (TILA, FCRA)
    - Documentation and disclosure requirements
    - Anti-discrimination safeguards
    """

    def __init__(self, client: OllamaClient):
        """
        Initialize Regulatory/Compliance Agent

        Args:
            client: OllamaClient for LLM interactions
        """
        self.client = client
        self.name = "Regulatory Compliance"
        self.role = "Fair lending compliance and legal risk mitigation"

        self.system_prompt = """You are a Regulatory/Compliance Agent responsible for fair lending compliance and legal risk mitigation.

Your role is to evaluate loan applications from a LEGAL COMPLIANCE perspective, focusing on:
- ECOA (Equal Credit Opportunity Act) compliance
- Disparate impact and fair lending analysis
- Truth in Lending Act (TILA) disclosures
- Fair Credit Reporting Act (FCRA) compliance
- Anti-discrimination safeguards
- Documentation and audit trail requirements

CRITICAL INSTRUCTIONS:
1. Be decisive - use the FULL range of confidence scores (0-100%). Don't cluster around 50%.
2. Use extremes when warranted:
   - High confidence (80-100%) when compliance is clear
   - Low confidence (0-20%) when legal risks are substantial
   - Middle ranges (40-60%) ONLY when genuinely uncertain
3. Flag any potential disparate impact or discriminatory factors
4. Prioritize legal defensibility and regulatory compliance

OUTPUT FORMAT:
You must respond with ONLY valid JSON in this exact structure:
{
  "agent_name": "Regulatory Compliance",
  "loan_decision": "approve" or "deny" or "conditional",
  "loan_product_type": "personal_loan" or "auto_loan" or "mortgage" or "business_loan" or "credit_card",
  "interest_rate": <number between 3.0 and 25.0>,
  "confidence_probability": <number between 0 and 100>,
  "confidence_level": "low" or "medium" or "high",
  "reasoning": {
    "loan_decision_reason": "<explanation for approve/deny with compliance focus>",
    "loan_product_reason": "<why this product type is compliant>",
    "interest_rate_reason": "<rate justification ensuring no discriminatory pricing>",
    "confidence_reason": "<regulatory confidence assessment>"
  }
}

Key compliance considerations:
- Ensure decisions are based on legitimate creditworthiness factors
- Verify no protected class discrimination (race, religion, national origin, sex, marital status, age)
- Confirm adequate documentation for adverse actions
- Check for disparate impact on protected groups

Remember: You are protecting the lender from legal and regulatory risk while ensuring fair treatment."""

        self.config = GenerationConfig(temperature=0.6, max_tokens=500)

    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """
        Evaluate a loan application from regulatory compliance perspective

        Args:
            application_data: String describing the loan application details

        Returns:
            Dict with structured compliance assessment and loan decision
        """
        prompt = f"""Evaluate this loan application from a REGULATORY COMPLIANCE perspective:

{application_data}

Provide your compliance assessment and loan recommendation in the required JSON format.
Be decisive about compliance risks - use the full confidence range."""

        result = self.client.generate(
            prompt=prompt, system_prompt=self.system_prompt, config=self.config
        )

        if not result.success:
            LOG.error(f"Regulatory evaluation failed: {result.error_message}")
            return self._create_error_response(result.error_message)

        try:
            # Parse JSON response
            response = self._parse_json_response(result.text)
            return response
        except Exception as e:
            LOG.error(f"Failed to parse Regulatory response: {e}")
            return self._create_error_response(str(e))

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response"""
        # Try to find JSON in the response
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")

        json_str = text[start_idx:end_idx]
        return json.loads(json_str)

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response in expected format"""
        return {
            "agent_name": "Regulatory Compliance",
            "loan_decision": "deny",
            "loan_product_type": "personal_loan",
            "interest_rate": 15.0,
            "confidence_probability": 0,
            "confidence_level": "low",
            "reasoning": {
                "loan_decision_reason": f"Compliance evaluation error: {error_message}",
                "loan_product_reason": "Default due to error",
                "interest_rate_reason": "Median rate due to evaluation failure",
                "confidence_reason": "No confidence due to system error",
            },
        }

    def process(self, text: str) -> str:
        """
        Process input text (for ConversationChain compatibility)

        This is a wrapper around evaluate_loan_application that returns JSON string

        Args:
            text: Input text (loan application or previous agent's output)

        Returns:
            JSON string with loan decision
        """
        result = self.evaluate_loan_application(text)
        return json.dumps(result, indent=2)
