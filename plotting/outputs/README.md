# FairWatch Plot Documentation

This directory contains visualization outputs from the FairWatch multi-agent loan evaluation system analysis.

## Dataset Summary

- **Total Records**: 576,000
- **Baseline Records**: 23,040 (4 agents × 5,760 samples each)
- **Sequential Records**: 552,960 (24 orderings × 5,760 chains × 4 agents)
- **Sequential Orderings**: 24 (all possible permutations)
- **Agents**: Consumer Advocate, Data Science, Regulatory, Risk Manager

---

## individual/

**Individual analysis for each baseline agent and sequential ordering.**

### individual/baselines/

Each agent has its own folder with:

- `overview_dashboard.png/pdf` - Comprehensive overview (approval rates, interest rates, demographics)
- `ethnicity_credit_heatmap.png/pdf` - Approval rates by ethnicity and credit score
- `interest_rate_demographics.png/pdf` - Interest rates by demographics
- `summary_stats.json` - Raw statistics

Agents: `consumer_advocate/`, `data_science/`, `regulatory/`, `risk_manager/`

### individual/sequential/

Each ordering has its own folder (24 total) with:

- `chain_overview.png/pdf` - Chain analysis dashboard
- `per_agent_analysis.png/pdf` - Per-agent breakdown within the chain
- `decision_flow.png/pdf` - Decision evolution through chain positions
- `summary_stats.json` - Raw statistics

Example folders: `consumer_advocate_data_science_regulatory_risk_manager/`, etc.

---

## demographic_bias/

Analyzes how demographic factors affect loan decisions.

| Plot                       | Description                                                                      |
| -------------------------- | -------------------------------------------------------------------------------- |
| `ethnicity_approval_rates` | Approval rates broken down by ethnicity signal across different experiment modes |
| `ethnicity_interest_rates` | Interest rate distributions by ethnicity                                         |
| `age_bias_analysis`        | How applicant age affects approval rates and terms                               |
| `visa_status_bias`         | Approval differences between visa statuses (US Citizen, H1B, etc.)               |
| `credit_score_analysis`    | Approval rates across credit score ranges                                        |
| `intersectional_bias`      | Combined effects of multiple demographic factors                                 |
| `disparity_metrics`        | Quantified bias metrics (demographic parity, equalized odds)                     |

---

## agent_comparison/

Compares behavior across different AI agents.

| Plot                             | Description                                               |
| -------------------------------- | --------------------------------------------------------- |
| `agent_approval_comparison`      | Overall approval rates by agent                           |
| `agent_interest_rate_comparison` | Interest rate distributions by agent                      |
| `agent_confidence_levels`        | Confidence score distributions by agent                   |
| `approval_type_distribution`     | Types of approvals (standard, conditional, etc.) by agent |
| `agent_agreement_matrix`         | How often agents agree/disagree with each other           |
| `agent_bias_comparison`          | Demographic bias levels per agent                         |
| `system_comparison`              | Baseline vs Sequential comparison                         |

---

## sequential_analysis/

Analyzes sequential chain behavior where agents evaluate in order.

| Plot                      | Description                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------------- |
| `order_impact_approval`   | How agent ordering affects final approval rates                                                 |
| `decision_drift`          | How decisions change as they pass through the chain                                             |
| `first_agent_influence`   | Impact of the first agent on final outcomes                                                     |
| `interest_rate_evolution` | How interest rates change through the chain                                                     |
| `order_bias_interaction`  | Heatmap showing bias levels for different orderings (top 5 most biased + bottom 5 least biased) |
| `consensus_analysis`      | Agreement patterns within sequential chains                                                     |

---

## mode_comparison/

Compares different evaluation modes.

| Plot                            | Description                                                         |
| ------------------------------- | ------------------------------------------------------------------- |
| `mode_comparison_overview`      | Side-by-side comparison of baseline, sequential, and parallel modes |
| `statistical_comparison`        | Statistical significance tests between modes                        |
| `fairness_metrics_comparison`   | Fairness metrics across modes                                       |
| `sample_size_distribution`      | Data distribution across modes                                      |
| `parallel_vs_sequential_detail` | Detailed comparison of parallel vs sequential                       |
| `statistical_tests.csv`         | Raw statistical test results                                        |

---

## fairness/

Dedicated fairness analysis plots.

| Plot                           | Description                                                    |
| ------------------------------ | -------------------------------------------------------------- |
| `fairness_dashboard`           | Overview of key fairness metrics                               |
| `fairness_by_ordering`         | How fairness varies by agent ordering                          |
| `protected_attribute_analysis` | Analysis of protected attributes (ethnicity, age, visa status) |

---

## publication/

Publication-ready figures (available in both PNG and PDF).

| Plot                         | Description                              |
| ---------------------------- | ---------------------------------------- |
| `figure_main_results`        | Main results figure for paper            |
| `figure_sequential_ordering` | Sequential ordering impact visualization |
| `figure_agent_agreement`     | Agent agreement patterns                 |
| `figure_bias_heatmap`        | Bias heatmap across conditions           |
| `table_summary_stats`        | Summary statistics table (also as CSV)   |

---

## supplementary/

Additional analysis plots.

| Plot                          | Description                               |
| ----------------------------- | ----------------------------------------- |
| `income_analysis`             | Approval rates by income level            |
| `loan_amount_analysis`        | How loan amount affects decisions         |
| `reasoning_analysis`          | Analysis of agent reasoning patterns      |
| `name_based_analysis`         | Whether applicant names affect decisions  |
| `detailed_ethnicity_analysis` | Granular ethnicity breakdown              |
| `correlation_matrix`          | Correlations between all numeric features |

---

## Regenerating Plots

```bash
cd /path/to/fairwatch

# Run all overall analysis plots
python plotting/run_all_plots.py

# Run individual analysis (per baseline + per sequential ordering)
python plotting/plot_individual_analysis.py
```

Individual plot scripts can also be run separately:

```bash
python plotting/plot_demographic_bias.py
python plotting/plot_agent_comparison.py
python plotting/plot_sequential_analysis.py
python plotting/plot_mode_comparison.py
python plotting/plot_fairness_analysis.py
python plotting/plot_publication_figures.py
python plotting/plot_supplementary.py
python plotting/plot_individual_analysis.py  # NEW: Individual analysis
```
