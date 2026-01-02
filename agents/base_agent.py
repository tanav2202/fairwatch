
import json, logging, re
from abc import ABC, abstractmethod
from utils.ollama_client import GenerationConfig

LOG = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, client, agent_name: str):
        self.client = client
        self.agent_name = agent_name
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return f"""You are {self.agent_name}. Evaluate the loan application.
Respond ONLY in valid JSON matching this schema exactly:
{{
    "approval_decision": "approve" or "deny",
    "approval_type": "STANDARD_TERMS" or "SUBOPTIMAL_TERMS" or "DENIAL",
    "confidence_probability": integer 0-100,
    "reasoning": "concise explanation"
}}"""

    def _clean_json(self, text: str) -> str:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1: return text[start:end+1]
        return text

    def evaluate_loan_application(self, app_data: str):
        config = GenerationConfig(temperature=0.1, json_mode=True)
        # Attempt 1
        res = self.client.generate(app_data, self.system_prompt, config)
        if res.success:
            try: return json.loads(self._clean_json(res.text))
            except: pass
        
        # Attempt 2 (Retry)
        res = self.client.generate(f"Previous invalid JSON. Input: {app_data}. Fix JSON:", config=config)
        if res.success:
            try: return json.loads(self._clean_json(res.text))
            except Exception as e:
                LOG.error(f"JSON Fail: {res.text}")
        
        # Fallback
        return {"approval_decision": "deny", "approval_type": "ERROR", "confidence_probability": 0, "reasoning": "System Error"}

    @abstractmethod
    def evaluate_pairwise_comparison(self, a, b): pass
