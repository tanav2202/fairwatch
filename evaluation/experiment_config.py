"""
Experiment Configuration
Centralized parameters for baseline and chain experiments

To run different experiments, just edit the values below:
- Change PROMPT_SAMPLE_SIZE to control dataset size (None = all prompts)
- Add/remove AGENT_ORDERINGS to test different sequences
- Modify output directories for organizing results
"""

# Dataset Configuration
PROMPT_CSV = "test_prompts.csv"
PROMPT_SAMPLE_SIZE = 20  # Set to 20 for quick runs, None for full dataset
RANDOM_SEED = 42

# Agent Orderings to Test
# Format: List of agent class names in desired order
AGENT_ORDERINGS = [
    ['FarmerAgent', 'AdvocacyAgent', 'ScienceAgent', 'MediaAgent', 'PolicyAgent'],
    ['ScienceAgent', 'FarmerAgent', 'AdvocacyAgent', 'MediaAgent', 'PolicyAgent'],
]

# Output Configuration  
BASELINE_OUTPUT_DIR = "results/individual_agents"
CHAIN_OUTPUT_DIR = "results/chain_evaluations"

# Model Configuration
MODEL_NAME = "llama3.2"
TIMEOUT = 60000
