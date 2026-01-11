
from agents.base_agent import BaseAgent
class RegulatoryAgent(BaseAgent):
    def __init__(self, client): super().__init__(client, "regulatory")
    def evaluate_pairwise_comparison(self, a, b): return {}
