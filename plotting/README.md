# FairWatch Plotting Module

This directory contains comprehensive plotting scripts for analyzing multi-agent system results.

## Directory Structure

```
plotting/
├── data_loader.py              # Data loading and preprocessing utilities
├── plot_demographic_bias.py    # Ethnicity, age, visa status bias analysis
├── plot_agent_comparison.py    # Comparison across different agents
├── plot_sequential_analysis.py # Sequential chain ordering analysis
├── plot_mode_comparison.py     # Baseline vs Sequential vs Parallel comparison
├── plot_publication_figures.py # Publication-ready figures
├── plot_supplementary.py       # Additional detailed analyses
├── run_all_plots.py           # Master script to run all plots
└── outputs/                    # Generated plots (created after running)
    ├── demographic_bias/
    ├── agent_comparison/
    ├── sequential_analysis/
    ├── mode_comparison/
    ├── publication/
    └── supplementary/
```

## Quick Start

```bash
cd /path/to/fairwatch
python plotting/run_all_plots.py
```

Or run individual modules:

```bash
python plotting/plot_demographic_bias.py
python plotting/plot_agent_comparison.py
# etc.
```

## Requirements

```bash
pip install matplotlib seaborn pandas numpy scipy
```

## Generated Plots

### 1. Demographic Bias Analysis (`plot_demographic_bias.py`)

- `ethnicity_approval_rates.png` - Approval rates by ethnicity across modes
- `ethnicity_interest_rates.png` - Interest rate distributions by ethnicity
- `age_bias_analysis.png` - Age group impact on approvals and rates
- `visa_status_bias.png` - Visa status impact analysis
- `credit_score_analysis.png` - Credit score tier analysis
- `intersectional_bias.png` - Ethnicity × Credit score heatmaps
- `disparity_metrics.png` - Fairness disparity ratio metrics

### 2. Agent Comparison (`plot_agent_comparison.py`)

- `agent_approval_comparison.png` - Approval rates by agent
- `agent_interest_rate_comparison.png` - Interest rate distributions by agent
- `agent_confidence_levels.png` - Confidence level distributions
- `approval_type_distribution.png` - Approval type breakdown
- `agent_agreement_matrix.png` - Pairwise agent agreement
- `agent_bias_comparison.png` - Per-agent demographic bias
- `system_comparison.png` - Overall system metrics

### 3. Sequential Analysis (`plot_sequential_analysis.py`)

- `order_impact_approval.png` - Agent ordering effects on approval
- `decision_drift.png` - Decision evolution through chain
- `first_agent_influence.png` - First agent's influence on final decision
- `interest_rate_evolution.png` - Rate changes through chain
- `order_bias_interaction.png` - Ordering × ethnicity interaction
- `consensus_analysis.png` - Agent consensus patterns

### 4. Mode Comparison (`plot_mode_comparison.py`)

- `mode_comparison_overview.png` - Comprehensive 9-panel comparison
- `statistical_comparison.png` - Statistical significance tests
- `fairness_metrics_comparison.png` - DP ratio and fairness gaps
- `sample_size_distribution.png` - Data distribution analysis
- `parallel_vs_sequential_detail.png` - Detailed mode comparison

### 5. Publication Figures (`plot_publication_figures.py`)

- `figure_main_results.png/pdf` - Main paper figure (6 panels)
- `figure_sequential_ordering.png/pdf` - Sequential ordering analysis
- `figure_agent_agreement.png/pdf` - Agent agreement heatmap
- `figure_bias_heatmap.png/pdf` - Intersectional bias visualization
- `table_summary_stats.png/pdf/csv` - Summary statistics table

### 6. Supplementary (`plot_supplementary.py`)

- `income_analysis.png` - Income tier analysis
- `loan_amount_analysis.png` - Loan amount impact
- `reasoning_analysis.png` - Approval type distributions
- `name_based_analysis.png` - Analysis by applicant name
- `detailed_ethnicity_analysis.png` - Comprehensive ethnicity study
- `correlation_matrix.png` - Feature correlations

## Data Schema

The data loader expects JSON files with the following structure:

### Baseline Results

```json
{
  "results": [
    {
      "prompt_id": 123,
      "prompt": "...",
      "mode": "baseline",
      "agent": "consumer_advocate",
      "business_decision": {
        "approval_decision": "approve",
        "approval_type": "STANDARD_TERMS",
        "interest_rate": 7.5,
        "confidence_probability": 90,
        "confidence_level": "high"
      }
    }
  ]
}
```

### Sequential Results

```json
{
  "results": [
    {
      "chain_id": 1,
      "input": "{...}",
      "decisions": {
        "agent_name": {
          "approval_decision": "...",
          "interest_rate": "..."
        }
      }
    }
  ]
}
```

### Parallel Results

```json
{
  "results": [{
    "prompt_id": 123,
    "mode": "parallel",
    "agent_outputs": {...},
    "business_decision": {...}
  }]
}
```

## Customization

To add new plots:

1. Create a new plotting module (e.g., `plot_custom.py`)
2. Import `load_data` from `data_loader`
3. Define your plotting functions
4. Add a `main()` function
5. (Optional) Add to `run_all_plots.py`

Example:

```python
from data_loader import load_data
import matplotlib.pyplot as plt

def plot_custom_analysis(df, output_dir):
    # Your plotting code
    pass

def main():
    df, _ = load_data("70bresults")
    output_dir = Path("plotting/outputs/custom")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_custom_analysis(df, output_dir)

if __name__ == "__main__":
    main()
```
