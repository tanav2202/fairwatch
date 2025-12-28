"""
Experiment Configuration
Centralized parameters for baseline and chain experiments

To run different experiments, just edit the values below:
- Change PROMPT_SAMPLE_SIZE to control dataset size (None = all prompts)
- Add/remove AGENT_ORDERINGS to test different sequences
- Modify output directories for organizing results
"""

# Dataset Configuration
PROMPT_CSV = "loan_applications.csv"
PROMPT_SAMPLE_SIZE = 5  # Set to 20 for quick runs, None for full dataset
RANDOM_SEED = 42

# Agent Orderings to Test
# Format: List of agent class names in desired order
# Ordering A: Standard risk assessment flow (risk → compliance → analytics → consumer)
# Ordering B: Reverse order to test first-mover effects (consumer → analytics → compliance → risk)
AGENT_ORDERINGS = [
    [
        "RiskManagerAgent",
        "RegulatoryAgent",
        "DataScienceAgent",
        "ConsumerAdvocateAgent",
    ],  # Ordering A
    [
        "ConsumerAdvocateAgent",
        "DataScienceAgent",
        "RegulatoryAgent",
        "RiskManagerAgent",
    ],  # Ordering B (reverse)
]

# Output Configuration
BASELINE_OUTPUT_DIR = "results/individual_agents"
CHAIN_OUTPUT_DIR = "results/chain_evaluations"

# Model Configuration
MODEL_NAME = "llama3.2"
TIMEOUT = 60000
