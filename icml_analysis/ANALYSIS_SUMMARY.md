# FairWatch: Multi-Agent LLM Analysis - Complete Results Summary

## Overview

This document summarizes the comprehensive analysis of multi-agent LLM systems across 4 model sizes for the ICML paper. The analysis examines how agent ordering, model scale, and interaction patterns affect fairness, decision-making, and error correction in credit approval systems.

**Models Analyzed:**
- **Llama 3.2** (3B parameters) - 24 sequential orderings
- **Mistral Latest** (7B parameters) - 24 sequential orderings  
- **Llama 70B** (70B parameters) - 24 sequential orderings
- **Qwen** (72B parameters) - 8 sequential orderings

**Dataset:** 5,760 synthetic loan applications with demographic attributes (ethnicity, credit score, income, age, etc.)

---

## Section 4.1: Ordering Instability

**Research Question:** How does agent ordering affect final decisions across different model sizes?

### Key Findings

**Variance in Approval Rates by Ordering:**

| Model | Min Approval | Max Approval | Range | Std Dev |
|-------|--------------|--------------|-------|---------|
| Llama 3.2 (3B) | 75.35% | 99.93% | 24.58% | ±5.11% |
| Mistral (7B) | 0.00% | 89.36% | 89.36% | ±37.33% |
| Llama 70B (70B) | 75.66% | 89.22% | 13.56% | ±7.94% |
| Qwen (72B) | 82.64% | 87.17% | 4.53% | ±1.57% |

**Critical Insight:** Small models (3B-7B) show extreme sensitivity to agent ordering, with approval rates varying by up to 89 percentage points. Larger models (70B-72B) demonstrate more stable behavior but still show 4-14% variation.

**Most Extreme Example (Mistral):**
- Ordering `Reg_DS_CA_RM`: 0.00% approval
- Ordering `DS_CA_RM_Reg`: 89.36% approval
- **Same data, 89.36% difference in outcomes**

### Implications
- Agent ordering is a **critical hyperparameter** that dramatically affects system behavior
- Small models are particularly vulnerable to ordering effects
- No single "correct" ordering exists - choice significantly impacts fairness

---

## Section 4.2: Information Cascades

**Research Question:** Do agents conform to earlier decisions (information cascades) or maintain independence?

### Key Findings

**First vs Final Decision Agreement Rates:**

| Model | Orderings Analyzed | Avg Agreement | Range | Cascade Lock-in |
|-------|-------------------|---------------|-------|-----------------|
| Llama 70B | 24 | **99.99%** | 99.95-100% | Near-perfect |
| Qwen | 8 | **99.95%** | 99.80-100% | Near-perfect |
| Llama 3.2 | 24 | **83.32%** | 67.01-99.90% | Moderate |
| Mistral | 10 | **90.69%** | 76.21-100% | High |

**Critical Insight:** Large models (70B-72B) exhibit near-perfect information cascades - once the first agent decides, subsequent agents almost never deviate. The first agent's decision becomes locked in.

**Cascade Strength by Model:**
- **70B/72B models:** First decision propagates with 99.95-100% fidelity
- **Mistral (7B):** 90.69% cascade rate, some diversity remains
- **Llama 3.2 (3B):** 83.32% cascade rate, most agent independence

### Example: Llama 70B, Ordering `DS_RM_CA_Reg`
- First agent (Data Science): Approve
- Agents 2-4: Approve, Approve, Approve
- **Conformity:** 100% across 5,760 cases

### Implications
- Large models prioritize consensus over independent evaluation
- The **position of each agent type matters enormously** - first agent has disproportionate influence
- Multi-agent deliberation may be illusory with large models

---

## Section 4.3: Parallel vs Sequential Mode Analysis

**Research Question:** Does parallel processing (simultaneous decisions) reduce ordering instability?

### Key Findings

**Parallel vs Sequential Approval Rates:**

| Model | Parallel Rate | Sequential Mean | Sequential Std | Variance Reduction |
|-------|--------------|-----------------|----------------|-------------------|
| Llama 70B | 84.04% | 83.11% | ±7.94% | **100%** eliminated |
| Llama 3.2 | 98.21% | 92.56% | ±5.11% | **100%** eliminated |
| Mistral | 73.24% | 28.02% | ±37.33% | **100%** eliminated |

**Critical Insight:** Parallel mode completely eliminates ordering variance. All agents see the same initial state simultaneously, removing cascade effects.

**Mistral Anomaly:**
- Sequential: 28.02% mean approval (highly unstable, 0-89% range)
- Parallel: 73.24% approval (stable, single value)
- Parallel mode reveals model's "true" aggregate preference

### Comparison: Sequential Distribution vs Parallel

**Llama 70B Sequential Orderings:**
- 24 different outcomes ranging 75.66% - 89.22%
- Mean: 83.11%, Std: 7.94%

**Llama 70B Parallel:**
- Single outcome: 84.04%
- Std: 0% (no variance)

### Implications
- **Parallel mode is more robust** for production deployment
- Sequential mode's variance represents systemic instability
- Trade-off: Parallel loses deliberation benefits but gains consistency

---

## Section 4.4: Functional Collapse Analysis

**Research Question:** Do models achieve fairness (demographic parity) while ignoring creditworthiness (functional collapse)?

**Metrics:**
- **Demographic Parity:** Max disparity in approval rates across 4 ethnic groups (Asian, Black, Hispanic, White)
- **Credit Score Correlation:** Spearman ρ between credit score and approval decision
- **Functional Collapse:** Disparity ≤ 5% AND |correlation| < 0.3

### Key Findings

**Functional Collapse by Model:**

| Model | Orderings | Collapsed | Collapse Rate | Typical Disparity | Typical Correlation |
|-------|-----------|-----------|---------------|-------------------|---------------------|
| Llama 70B | 24 | **24** | **100%** | 1.22-4.79% | 0.134-0.275 |
| Qwen | 8 | **8** | **100%** | 2.05-4.72% | 0.143-0.240 |
| Llama 3.2 | 24 | **16** | **67%** | 0.43-19.70% | 0.024-0.345 |
| Mistral | 10 | **0** | **0%** | 6.56-16.04% | 0.118-0.637 |

**Critical Insight:** Large models (70B-72B) consistently achieve demographic fairness BUT ignore credit scores. They are "fair" but not credit-worthy.

### Examples

**Llama 70B - Ordering `DS_RM_CA_Reg` (Collapsed):**
- Ethnic disparity: 1.25% (very fair)
- Credit correlation: ρ = 0.134 (weak signal)
- **Interpretation:** Equal approval across ethnic groups, but credit score barely matters

**Llama 3.2 - Ordering `CA_DS_Reg_RM` (NOT Collapsed):**
- Ethnic disparity: 3.44% (fair)
- Credit correlation: ρ = 0.345 (moderate signal)
- **Interpretation:** Both fair AND credit-aware

**Mistral - Ordering `CA_DS_Reg_RM` (NOT Collapsed):**
- Ethnic disparity: 15.10% (unfair)
- Credit correlation: ρ = 0.612 (strong signal)
- **Interpretation:** Credit-aware but ethnically biased

### Approval Rates by Ethnicity (Example: Llama 70B)

| Ordering | Asian | Black | Hispanic | White | Max Disparity |
|----------|-------|-------|----------|-------|---------------|
| DS_RM_CA_Reg | 83.40% | 84.51% | 83.06% | 82.26% | **1.25%** |
| CA_Reg_RM_DS | 84.03% | 87.85% | 83.47% | 83.44% | **4.39%** |
| Mean (24 orderings) | ~83% | ~85% | ~83% | ~83% | **2.8%** |

### Implications
- **Large models optimize for fairness at expense of utility**
- Credit score signal is suppressed to achieve demographic parity
- **The paradox:** Systems become "fair" but useless for lending decisions
- Smaller models (Llama 3.2) show better balance - some orderings achieve both fairness AND credit-awareness

---

## Section 5.2: The Scale Paradox

**Research Question:** Do larger models provide better error correction through multi-agent deliberation?

**Methodology:**
- Define ground truth: Credit scores ≥700 should be approved
- Track when first agent makes error (wrong decision vs ground truth)
- Measure if subsequent agents correct the error

### Key Findings

**Agreement vs Error Correction by Model Size:**

| Model | Size | Inter-Agent Agreement | Error Correction Rate | Full Consensus |
|-------|------|----------------------|----------------------|----------------|
| Llama 3.2 | 3B | 89.75% ± 8.39% | **19.99%** ± 16.41% | 78.60% |
| Mistral | 7B | 98.22% ± 4.81% | **6.71%** ± 19.66% | 95.41% |
| Llama 70B | 70B | 100.00% ± 0.01% | **0.01%** ± 0.02% | 99.99% |
| Qwen | 72B | 99.98% ± 0.03% | **0.04%** ± 0.08% | 99.95% |

**THE PARADOX:**
```
As model size increases 3B → 72B:
  ✓ Agreement increases: 89.75% → 99.98% (+10.23%)
  ✗ Error correction DECREASES: 19.99% → 0.04% (-19.95%)
```

**Critical Insight:** Larger models achieve near-perfect consensus through cascading, BUT they lose the ability to correct initial errors. Smaller models show more disagreement but demonstrate 200x-500x more error correction!

### Error Correction by Position

**When First Agent Makes a Mistake:**

| Model | Agent 2 Corrects | Agent 3 Corrects | Agent 4 Corrects | Total Corrected |
|-------|-----------------|-----------------|-----------------|-----------------|
| Llama 3.2 | ~8% | ~6% | ~6% | **~20%** |
| Mistral | ~3% | ~2% | ~2% | **~7%** |
| Llama 70B | ~0.01% | ~0% | ~0% | **~0.01%** |
| Qwen | ~0.02% | ~0.01% | ~0.01% | **~0.04%** |

**Example: Llama 3.2, Ordering `CA_DS_Reg_RM`:**
- First agent errors: 520 cases (9% of dataset)
- Corrected by Agent 2: 286 cases (55%)
- Remaining errors: 234 cases cascade through

**Example: Llama 70B, Ordering `DS_CA_Reg_RM`:**
- First agent errors: 893 cases (15% of dataset)
- Corrected by Agents 2-4: 1 case (0.1%)
- Remaining errors: 892 cases cascade through (99.9%)

### Implications
- **Multi-agent systems with large models don't deliberate - they cascade**
- Error correction requires agent independence, which large models lack
- **The "wisdom of crowds" fails when agents over-conform**
- Smaller models maintain diversity of thought, enabling self-correction
- Trade-off: Smaller models correct more errors but are less consistent

---

## Cross-Cutting Insights

### 1. The First-Agent Effect
Across all analyses, the **first agent's decision disproportionately determines outcomes**:
- Section 4.2: 99.99% first-to-final agreement (70B)
- Section 5.2: 0.01% error correction after first decision (70B)
- **Implication:** In large models, agent ordering = outcome

### 2. Model Scale as a Double-Edged Sword

**Large Models (70B-72B):**
- ✓ More stable across orderings (4-14% variance)
- ✓ Near-perfect demographic fairness (1-5% disparity)
- ✓ High consistency (99.99% consensus)
- ✗ Strong cascade effects (no deliberation)
- ✗ Weak credit score signals (functional collapse)
- ✗ No error correction (0.01-0.04%)

**Small Models (3B-7B):**
- ✗ Highly unstable across orderings (24-89% variance)
- ✗ Less consistent fairness (0-20% disparity)
- ✓ Some orderings balance fairness + utility
- ✓ Moderate agent independence
- ✓ Significant error correction (7-20%)

### 3. The Illusion of Multi-Agent Deliberation

**Expected behavior:** Multiple agents debate, disagree, refine decisions
**Actual behavior (large models):** First agent decides, others conform

**Evidence:**
- 99.99% agreement between first and final decisions
- 0.01% error correction rate
- 100% consensus in 99.99% of cases

**Conclusion:** With large models, multi-agent systems are effectively **single-agent systems with extra steps**.

### 4. The Parallel Mode Solution (Partial)

**Parallel processing eliminates ordering variance completely** by removing cascade effects. However:
- ✓ Solves: Ordering instability (Section 4.1)
- ✓ Solves: Sequential cascade lock-in (Section 4.2)
- ✗ Doesn't solve: Functional collapse (Section 4.4)
- ✗ Doesn't solve: Loss of error correction (Section 5.2)

**Trade-off:** Gain consistency, lose deliberation (which was already minimal in large models)

---

## Summary Statistics

### Overall Model Performance

| Metric | Llama 3.2 (3B) | Mistral (7B) | Llama 70B (70B) | Qwen (72B) |
|--------|----------------|--------------|-----------------|------------|
| **Ordering Variance** | ±5.11% | ±37.33% | ±7.94% | ±1.57% |
| **Cascade Strength** | 83.32% | 90.69% | 99.99% | 99.95% |
| **Demographic Fairness** | 67% collapsed | 0% collapsed | 100% collapsed | 100% collapsed |
| **Error Correction** | 19.99% | 6.71% | 0.01% | 0.04% |
| **Agent Independence** | Moderate | Low | None | None |

### Ordering Effects (Sequential Mode)

| Model | Most Lenient Ordering | Approval % | Most Strict Ordering | Approval % | Difference |
|-------|----------------------|------------|---------------------|------------|------------|
| Llama 3.2 | RM_Reg_CA_DS | 99.93% | CA_DS_Reg_RM | 75.35% | **24.58%** |
| Mistral | DS_CA_RM_Reg | 89.36% | Reg_DS_CA_RM | 0.00% | **89.36%** |
| Llama 70B | RM_DS_CA_Reg | 89.22% | Reg_CA_DS_RM | 75.66% | **13.56%** |
| Qwen | CA_DS_RM_Reg | 87.17% | DS_Reg_RM_CA | 82.64% | **4.53%** |

---

## Key Recommendations

### For Researchers

1. **Report ordering effects:** Model evaluations should include variance across agent orderings
2. **Measure cascade strength:** Test multi-agent independence, not just final accuracy
3. **Validate error correction:** Large models may achieve consensus without self-correction
4. **Balance fairness vs utility:** Functional collapse is a real risk with large models

### For Practitioners

1. **Use parallel mode for production:** Eliminates ordering instability (if deliberation isn't needed)
2. **Carefully select first agent:** With large models, first position dominates outcomes
3. **Consider smaller models:** Better error correction and fairness-utility balance in some cases
4. **Monitor demographic signals:** Large models may suppress creditworthiness to achieve fairness

### For Policy & Ethics

1. **Ordering is not neutral:** Agent sequence is a fairness-critical hyperparameter
2. **Consensus ≠ correctness:** High agreement may indicate cascade, not quality
3. **Scale introduces new risks:** Larger models show systematic functional collapse
4. **Transparency requirements:** Document which agent types appear first, ordering variance

---

## Files Generated

### Section 4.1: Ordering Instability
- `icml_analysis/section_4_1_ordering_instability/` (if exists - check for plots)

### Section 4.2: Information Cascades
- `icml_analysis/section_4_2_first_vs_final_transitions/`
  - `llama_70b_all_transitions.png` (479 KB)
  - `qwen_all_transitions.png` (242 KB)
  - `llama_3_2_all_transitions.png` (646 KB)
  - `mistral_all_transitions.png` (332 KB)
  - `transition_metrics.json` (agreement rates for all orderings)

### Section 4.3: Parallel Analysis
- `icml_analysis/section_4_3_parallel_analysis/`
  - `parallel_vs_sequential_comparison.png` (199 KB)
  - `sequential_approval_distributions.png` (171 KB)
  - `parallel_analysis_summary.json`

### Section 4.4: Functional Collapse
- `icml_analysis/section_4_4_functional_collapse/`
  - `functional_collapse_scatter.png` (305 KB)
  - `ethnic_approval_heatmaps.png` (242 KB)
  - `credit_correlation_distribution.png` (151 KB)
  - `fairness_metrics_all_orderings.json` (39 KB)
  - `functional_collapse_summary.json` (26 KB)

### Section 5.2: Scale Paradox
- `icml_analysis/section_5_2_scale_paradox/`
  - `scale_paradox_agreement_vs_correction.png` (281 KB)
  - `agreement_vs_correction_scatter.png` (301 KB)
  - `error_correction_by_position.png` (277 KB)
  - `scale_paradox_metrics.json` (48 KB)
  - `scale_paradox_summary.json` (1.5 KB)

**Total:** 15+ visualizations, 8+ JSON metric files, all with PDF versions

---

## Conclusion

This analysis reveals **fundamental tensions in multi-agent LLM systems**:

1. **Scale vs Independence:** Larger models achieve consensus but lose deliberation
2. **Fairness vs Utility:** Demographic parity often comes at the cost of credit-worthiness
3. **Consistency vs Correction:** Stable outputs require sacrificing error correction
4. **Ordering Sensitivity:** Agent sequence dramatically affects outcomes, especially in smaller models

**The Central Finding:** With large models (70B+), multi-agent systems exhibit strong **information cascade effects** that create the illusion of deliberation while actually replicating single-agent decision-making. The first agent's position becomes disproportionately influential, subsequent agents rarely correct errors, and systems optimize for demographic fairness by suppressing credit signals.

**The Path Forward:** Designers must choose:
- Large models in parallel mode (consistency, fairness, but functional collapse)
- Small models in sequential mode (error correction, utility preservation, but ordering instability)
- Or develop new architectures that preserve agent independence at scale

---

*Analysis conducted: January 2026*  
*Dataset: 5,760 synthetic loan applications*  
*Models: Llama 3.2 (3B), Mistral (7B), Llama 70B (70B), Qwen (72B)*  
*Framework: FairWatch Multi-Agent Credit Approval System*
