"""
Risk Manager Agent
Role: Credit risk assessment, portfolio protection
Focus: Default probability, repayment capacity, collateral evaluation
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


class RiskManagerAgent:
    """
    Risk Manager Agent for credit risk assessment

    Evaluates loan applications based on:
    - Default probability and credit risk metrics
    - Repayment capacity and debt-to-income ratios
    - Collateral value and coverage
    - Historical payment patterns and credit history
    """

    def __init__(self, client: OllamaClient):
        """
        Initialize Risk Manager Agent

        Args:
            client: OllamaClient for LLM interactions
        """
        self.client = client
        self.name = "Risk Manager"
        self.role = "Credit risk assessment and portfolio protection"

        self.system_prompt = """You are a Risk Manager Agent responsible for credit risk assessment and portfolio protection.

Your role is to evaluate loan applications from a RISK MANAGEMENT perspective, focusing on:
- Default probability and credit risk indicators
- Repayment capacity (income stability, debt-to-income ratio)
- Collateral value and coverage ratio
- Credit history and payment patterns
- Portfolio risk concentration

CRITICAL INSTRUCTIONS:
1. Be decisive - use the FULL range of confidence scores (0-100%). Don't cluster around 50%.
2. Use extremes when warranted:
   - High confidence (80-100%) when risk is clearly low or clearly high
   - Low confidence (0-20%) when applicant is clearly unsuitable
   - Middle ranges (40-60%) ONLY when genuinely uncertain
3. Your decision should be based on hard risk metrics, not compassion
4. Consider worst-case scenarios for portfolio protection

OUTPUT FORMAT:
You must respond with ONLY valid JSON in this exact structure:
{
  "agent_name": "Risk Manager",
  "loan_decision": "approve" or "deny",
  "loan_product_type": "personal_loan" or "auto_loan" or "mortgage" or "business_loan" or "credit_card",
  "interest_rate": <number between 3.0 and 25.0>,
  "confidence_probability": <number between 0 and 100>,
  "confidence_level": "low" or "medium" or "high",
  "reasoning": {
    "loan_decision_reason": "<explanation for approve/deny>",
    "loan_product_reason": "<why this specific product type>",
    "interest_rate_reason": "<justification for rate based on risk>",
    "confidence_reason": "<why this confidence level>"
  }
}

Base your interest rates on risk:
- Low risk (excellent credit): 3-8%
- Medium risk (fair credit): 8-15%
- High risk (poor credit): 15-25%

Remember: You are protecting the lender's portfolio. Be conservative but fair."""

        self.config = GenerationConfig(temperature=0.7, max_tokens=500)

    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """
        Evaluate a loan application from risk management perspective

        Args:
            application_data: String describing the loan application details

        Returns:
            Dict with structured loan decision and reasoning
        """
        prompt = f"""Evaluate this loan application from a RISK MANAGEMENT perspective:

{application_data}

Provide your risk assessment and loan recommendation in the required JSON format.
Remember to use the FULL confidence range - be bold with your assessment."""

        result = self.client.generate(
            prompt=prompt, system_prompt=self.system_prompt, config=self.config
        )

        if not result.success:
            LOG.error(f"Risk Manager evaluation failed: {result.error_message}")
            return self._create_error_response(result.error_message)

        try:
            # Parse JSON response
            response = self._parse_json_response(result.text)
            return response
        except Exception as e:
            LOG.error(f"Failed to parse Risk Manager response: {e}")
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
            "agent_name": "Risk Manager",
            "loan_decision": "deny",
            "loan_product_type": "personal_loan",
            "interest_rate": 25.0,
            "confidence_probability": 0,
            "confidence_level": "low",
            "reasoning": {
                "loan_decision_reason": f"Error in evaluation: {error_message}",
                "loan_product_reason": "Default due to error",
                "interest_rate_reason": "Maximum rate due to evaluation failure",
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
