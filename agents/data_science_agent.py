
from agents.base_agent import BaseAgent
class DataScienceAgent(BaseAgent):
    def __init__(self, client): super().__init__(client, "data_science")
    def evaluate_pairwise_comparison(self, a, b): return {}
