
from agents.base_agent import BaseAgent
class RiskManagerAgent(BaseAgent):
    def __init__(self, client): super().__init__(client, "risk_manager")
    def evaluate_pairwise_comparison(self, a, b): return {}
