"""
Business Decision Agent - FINAL FIX for OllamaClient compatibility
"""

import json
from pathlib import Path


class BusinessDecisionAgent:
    """Synthesizes final decision from multiple agent recommendations"""
    
    def __init__(self, client):
        self.client = client
        self.agent_name = "business_decision"
        
        # Load persona
        persona_path = Path("prompts/business_decision_persona.txt")
        if persona_path.exists():
            with open(persona_path, 'r') as f:
                self.persona = f.read()
        else:
            self.persona = "You are a business decision maker synthesizing loan recommendations."
        
        # Load template
        template_path = Path("templates/business_synthesis_template.json")
        if template_path.exists():
            with open(template_path, 'r') as f:
                self.template = json.load(f)
        else:
            raise FileNotFoundError(f"Template not found: {template_path}")
    
    def _safe_float(self, value, default=0.0):
        """Safely convert to float"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value, default=0):
        """Safely convert to int"""
        if value is None:
            return default
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    def synthesize_decision(self, application_data: str, agent_recommendations: list) -> dict:
        """
        Synthesize final decision from agent recommendations
        
        Args:
            application_data: Original loan application text
            agent_recommendations: List of 4 agent outputs
            
        Returns:
            Business decision with agent influence weights
        """
        # Format agent recommendations
        formatted_recs = []
        agent_names = ['Risk Manager', 'Regulatory Compliance', 'Data Science', 'Consumer Advocate']
        
        for i, (name, rec) in enumerate(zip(agent_names, agent_recommendations)):
            decision = rec.get('approval_decision', 'N/A')
            app_type = rec.get('approval_type', 'N/A')
            rate = self._safe_float(rec.get('interest_rate'), 0.0)
            conf_prob = self._safe_int(rec.get('confidence_probability'), 0)
            conf_level = rec.get('confidence_level', 'N/A')
            reasoning = rec.get('reasoning', {}).get('approval_decision_reason', 'N/A')
            
            rec_text = f"""
Agent {i+1}: {name}
- Decision: {decision}
- Type: {app_type}
- Interest Rate: {rate}%
- Confidence: {conf_prob}% ({conf_level})
- Reasoning: {reasoning}
"""
            formatted_recs.append(rec_text.strip())
        
        recommendations_text = "\n\n".join(formatted_recs)
        
        # Build prompt with system instructions in the prompt itself
        system_instructions = "You are a business decision agent. Return ONLY valid JSON, no markdown, no other text."
        
        prompt = f"""{system_instructions}

{self.persona}

LOAN APPLICATION:
{application_data}

AGENT RECOMMENDATIONS:
{recommendations_text}

TASK:
Synthesize a final business decision considering all agent perspectives.

Return ONLY valid JSON matching this structure (no markdown, no backticks):
{{
  "agent_name": "Business Decision",
  "loan_type": "personal_loan",
  "approval_decision": "approve or deny",
  "approval_type": "STANDARD_TERMS or SUBOPTIMAL_TERMS or MANUAL_REVIEW or DENIAL",
  "interest_rate": 8.5,
  "confidence_probability": 75,
  "confidence_level": "high",
  "agent_influence": {{
    "risk_manager_weight": 0.25,
    "regulatory_weight": 0.25,
    "data_science_weight": 0.25,
    "consumer_advocate_weight": 0.25,
    "primary_influence": "Name of most influential agent"
  }},
  "reasoning": {{
    "synthesis_rationale": "Why this decision",
    "weight_justification": "Why these weights",
    "risk_assessment": "Final risk view"
  }}
}}

CRITICAL: Return ONLY the JSON object. Weights must sum to 1.0.
"""
        
        # Generate response using OllamaClient (no system_prompt parameter)
        result = self.client.generate(prompt=prompt)
        
        # Handle GenerationResult or string response
        if hasattr(result, 'text'):
            response = result.text
        else:
            response = result
        
        # Parse JSON
        try:
            business_decision = json.loads(response)
        except json.JSONDecodeError:
            # Extract JSON from markdown if present
            response_clean = response.strip()
            if "```json" in response_clean:
                json_text = response_clean.split("```json")[1].split("```")[0].strip()
            elif "```" in response_clean:
                json_text = response_clean.split("```")[1].split("```")[0].strip()
            else:
                # Try to find JSON object
                start = response_clean.find('{')
                end = response_clean.rfind('}') + 1
                if start >= 0 and end > start:
                    json_text = response_clean[start:end]
                else:
                    raise ValueError(f"Could not extract JSON from response: {response[:200]}")
            
            business_decision = json.loads(json_text)
        
        # Fix types
        business_decision['interest_rate'] = self._safe_float(business_decision.get('interest_rate'), 8.5)
        business_decision['confidence_probability'] = self._safe_int(business_decision.get('confidence_probability'), 50)
        
        # Fix and normalize weights
        agent_influence = business_decision.get('agent_influence', {})
        weights = [
            self._safe_float(agent_influence.get('risk_manager_weight', 0.25)),
            self._safe_float(agent_influence.get('regulatory_weight', 0.25)),
            self._safe_float(agent_influence.get('data_science_weight', 0.25)),
            self._safe_float(agent_influence.get('consumer_advocate_weight', 0.25))
        ]
        
        weight_sum = sum(weights)
        if weight_sum == 0:
            weights = [0.25, 0.25, 0.25, 0.25]
        elif abs(weight_sum - 1.0) > 0.01:
            weights = [w / weight_sum for w in weights]
        
        agent_influence['risk_manager_weight'] = weights[0]
        agent_influence['regulatory_weight'] = weights[1]
        agent_influence['data_science_weight'] = weights[2]
        agent_influence['consumer_advocate_weight'] = weights[3]
        
        business_decision['agent_influence'] = agent_influence
        
        return business_decision