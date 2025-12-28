"""
Consumer Advocate Agent
Role: Consumer protection, financial inclusion
Focus: Access to credit, transparency, fair treatment of borrowers
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


class ConsumerAdvocateAgent:
    """
    Consumer Advocate Agent for consumer protection and financial inclusion

    Evaluates loan applications based on:
    - Access to credit and financial inclusion
    - Fair treatment and transparency
    - Borrower ability to repay without hardship
    - Predatory lending risk assessment
    - Consumer rights and protections
    """

    def __init__(self, client: OllamaClient):
        """
        Initialize Consumer Advocate Agent

        Args:
            client: OllamaClient for LLM interactions
        """
        self.client = client
        self.name = "Consumer Advocate"
        self.role = "Consumer protection and financial inclusion"

        self.system_prompt = """You are a Consumer Advocate Agent responsible for consumer protection and financial inclusion.

Your role is to evaluate loan applications from a CONSUMER PROTECTION perspective, focusing on:
- Access to credit for underserved populations
- Fair treatment and transparent lending practices
- Borrower's ability to repay without financial hardship
- Prevention of predatory lending and debt traps
- Financial literacy and informed decision-making
- Long-term financial health of the borrower

CRITICAL INSTRUCTIONS:
1. Be decisive - use the FULL range of confidence scores (0-100%). Don't cluster around 50%.
2. Use extremes when warranted:
   - High confidence (80-100%) when loan clearly helps or harms consumer
   - Low confidence (0-20%) when loan appears predatory or unsuitable
   - Middle ranges (40-60%) ONLY when genuinely uncertain
3. Advocate for fair access while preventing harmful lending
4. Consider the borrower's full financial picture and well-being
5. Flag potential debt traps or predatory terms

OUTPUT FORMAT:
You must respond with ONLY valid JSON in this exact structure:
{
  "agent_name": "Consumer Advocate",
  "loan_decision": "approve" or "deny" or "approve_with_caution",
  "loan_product_type": "personal_loan" or "auto_loan" or "mortgage" or "business_loan" or "credit_card",
  "interest_rate": <number between 3.0 and 25.0>,
  "confidence_probability": <number between 0 and 100>,
  "confidence_level": "low" or "medium" or "high",
  "reasoning": {
    "loan_decision_reason": "<consumer protection perspective on decision>",
    "loan_product_reason": "<why this product serves consumer's best interest>",
    "interest_rate_reason": "<rate fairness and affordability assessment>",
    "confidence_reason": "<confidence in consumer welfare outcome>"
  }
}

Consumer-focused rate philosophy:
- Promote affordable rates that enable repayment
- Flag rates >20% as potentially predatory
- Consider total cost of credit, not just approval
- Balance access to credit with consumer protection

Remember: You advocate for consumers. Approve loans that help, deny those that harm, even if technically creditworthy."""

        self.config = GenerationConfig(temperature=0.7, max_tokens=500)

    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """
        Evaluate a loan application from consumer protection perspective

        Args:
            application_data: String describing the loan application details

        Returns:
            Dict with structured consumer-focused loan decision
        """
        prompt = f"""Evaluate this loan application from a CONSUMER ADVOCATE perspective:

{application_data}

Provide your consumer protection assessment and loan recommendation in the required JSON format.
Be a strong advocate - use the full confidence range when consumer welfare is clear."""

        result = self.client.generate(
            prompt=prompt, system_prompt=self.system_prompt, config=self.config
        )

        if not result.success:
            LOG.error(f"Consumer Advocate evaluation failed: {result.error_message}")
            return self._create_error_response(result.error_message)

        try:
            # Parse JSON response
            response = self._parse_json_response(result.text)
            return response
        except Exception as e:
            LOG.error(f"Failed to parse Consumer Advocate response: {e}")
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
            "agent_name": "Consumer Advocate",
            "loan_decision": "deny",
            "loan_product_type": "personal_loan",
            "interest_rate": 10.0,
            "confidence_probability": 0,
            "confidence_level": "low",
            "reasoning": {
                "loan_decision_reason": f"Consumer protection evaluation error: {error_message}",
                "loan_product_reason": "Default due to error",
                "interest_rate_reason": "Conservative rate due to evaluation failure",
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
