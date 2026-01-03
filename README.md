# FairWatch: Multi-Agent Bias Detection in Credit Decisioning

Research implementation for ACM FAccT 2025 paper: "When Agents Interact: Emergent Bias in Multi-Agent Credit Decisioning Systems"

## Overview

This project investigates whether multi-agent AI systems amplify or reduce bias in credit lending decisions. We evaluate 4 specialized agents (Risk Manager, Regulatory Compliance, Data Science, Consumer Advocate) across individual, sequential, and parallel architectures.

## Research Questions

**RQ1:** Do multi-agent AI systems amplify or reduce individual agent biases in credit decisions?

**RQ2:** Does agent ordering in sequential chains affect bias outcomes?

**RQ3:** Can architectural choices (sequential vs parallel) mitigate demographic disparities?

## Dataset

- **5,760 loan applications** per template (simple/complex)
- **13 ethnically-coded names** (6 ethnic groups, literature-validated)
- **Systematic variation:** 5 credit scores × 4 visa statuses × 3 income levels × 4 ages × 2 loan amounts
- **Templates:** Simple (baseline) and Complex (with misinformation traps)

## Architecture

### Agents
- `risk_manager` - Portfolio risk assessment
- `regulatory` - Compliance and fair lending
- `data_science` - Statistical risk modeling  
- `consumer_advocate` - Consumer protection
- `business_decision` - Synthesis agent

### Evaluation Modes
1. **Individual Baseline:** Each agent evaluates independently
2. **Sequential Chains:** 4 orderings testing anchoring effects (Tversky & Kahneman, 1974)
3. **Parallel + Synthesis:** Independent evaluation with business decision aggregation

## Quick Start

### Generate Dataset
```bash
python data/generate_dataset.py \
  --input data/input_list.py \
  --template prompts/simple_input.j2 \
  --output data/prompts_simple.csv
```

### Run Evaluations
```bash
# Individual baseline
python evaluation/run_individual_baseline_incremental.py \
  --llm llama3.2 \
  --prompts data/prompts_simple.csv \
  --output outputs_simple

# Sequential chains
python evaluation/run_sequential_chains_incremental.py \
  --llm llama3.2 \
  --prompts data/prompts_simple.csv \
  --ordering risk_regulatory_data_consumer \
  --output outputs_simple

# Parallel evaluation
python evaluation/run_parallel_chains_incremental.py \
  --llm llama3.2 \
  --prompts data/prompts_simple.csv \
  --output outputs_simple
```

## File Structure

```
fairwatch/
├── agents/               # Agent implementations
│   ├── base_agent.py
│   ├── risk_manager_agent.py
│   ├── regulatory_agent.py
│   ├── data_science_agent.py
│   ├── consumer_advocate_agent.py
│   └── business_decision_agent.py
├── prompts/              # Agent personas and templates
│   ├── *_persona.txt
│   ├── simple_input.j2
│   └── complex_input.j2
├── templates/            # JSON schemas
│   ├── loan_evaluation_template.json
│   └── business_synthesis_template.json
├── data/                 # Input lists and generated datasets
├── evaluation/           # Evaluation scripts
└── agentic_systems/      # Multi-agent orchestration
```

## Performance

- **Speed:** ~3.75 sec/evaluation (llama3.2 on M-series MacBook)
- **Individual baseline:** 24 hours (4 agents × 5,760 prompts)
- **Sequential chains:** 4 days (4 orderings)
- **Total runtime:** ~5 days for complete evaluation

## License

Research code for academic use.

**Status:** Active data collection • ETA completion: January 8, 2026