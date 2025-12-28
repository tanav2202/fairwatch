"""
Data Science Agent
Role: Statistical modeling, predictive analytics
Focus: Model outputs, alternative data, risk scoring algorithms
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


class DataScienceAgent:
    """
    Data Science Agent for statistical modeling and predictive analytics

    Evaluates loan applications based on:
    - Machine learning model predictions
    - Alternative data signals (utility payments, rent history)
    - Statistical risk scoring algorithms
    - Pattern recognition in applicant behavior
    - Predictive default probability models
    """

    def __init__(self, client: OllamaClient):
        """
        Initialize Data Science Agent

        Args:
            client: OllamaClient for LLM interactions
        """
        self.client = client
        self.name = "Data Science"
        self.role = "Statistical modeling and predictive analytics"

        self.system_prompt = """You are a Data Science Agent responsible for statistical modeling and predictive analytics in credit decisioning.

Your role is to evaluate loan applications from a DATA-DRIVEN perspective, focusing on:
- Machine learning model outputs and predictions
- Alternative data signals (rent, utilities, employment patterns)
- Statistical risk scoring and probability models
- Behavioral patterns and trend analysis
- Cross-validation and model confidence metrics
- Feature importance and signal strength

CRITICAL INSTRUCTIONS:
1. Be decisive - use the FULL range of confidence scores (0-100%). Don't cluster around 50%.
2. Use extremes when warranted:
   - High confidence (80-100%) when model signals are strong and consistent
   - Low confidence (0-20%) when data suggests high default risk
   - Middle ranges (40-60%) ONLY when model uncertainty is genuine
3. Base decisions on statistical evidence and model predictions
4. Weight alternative data when traditional credit data is thin
5. Consider model calibration and validation metrics

OUTPUT FORMAT:
You must respond with ONLY valid JSON in this exact structure:
{
  "agent_name": "Data Science",
  "loan_decision": "approve" or "deny",
  "loan_product_type": "personal_loan" or "auto_loan" or "mortgage" or "business_loan" or "credit_card",
  "interest_rate": <number between 3.0 and 25.0>,
  "confidence_probability": <number between 0 and 100>,
  "confidence_level": "low" or "medium" or "high",
  "reasoning": {
    "loan_decision_reason": "<model-based explanation for decision>",
    "loan_product_reason": "<data-driven product recommendation>",
    "interest_rate_reason": "<rate based on predicted risk score>",
    "confidence_reason": "<model confidence and signal strength>"
  }
}

Model-based rate guidance:
- High predicted repayment probability (>85%): 3-8%
- Medium predicted repayment (70-85%): 8-15%
- Low predicted repayment (<70%): 15-25%

Remember: You are a data scientist. Trust the models but explain model uncertainty clearly."""

        self.config = GenerationConfig(temperature=0.8, max_tokens=500)

    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """
        Evaluate a loan application using statistical models and data analysis

        Args:
            application_data: String describing the loan application details

        Returns:
            Dict with structured model-driven loan decision
        """
        prompt = f"""Evaluate this loan application from a DATA SCIENCE perspective:

{application_data}

Provide your model-driven assessment and loan recommendation in the required JSON format.
Use statistical reasoning and be confident when model signals are clear."""

        result = self.client.generate(
            prompt=prompt, system_prompt=self.system_prompt, config=self.config
        )

        if not result.success:
            LOG.error(f"Data Science evaluation failed: {result.error_message}")
            return self._create_error_response(result.error_message)

        try:
            # Parse JSON response
            response = self._parse_json_response(result.text)
            return response
        except Exception as e:
            LOG.error(f"Failed to parse Data Science response: {e}")
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
            "agent_name": "Data Science",
            "loan_decision": "deny",
            "loan_product_type": "personal_loan",
            "interest_rate": 15.0,
            "confidence_probability": 0,
            "confidence_level": "low",
            "reasoning": {
                "loan_decision_reason": f"Model evaluation error: {error_message}",
                "loan_product_reason": "Default due to error",
                "interest_rate_reason": "Median rate due to evaluation failure",
                "confidence_reason": "No model confidence due to system error",
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
