# FairWatch: Multi-Agent Bias Simulation

## Overview
This repository contains the **FairWatch High-Performance Inference Pipeline**, a research system designed to simulate and analyze bias in financial decision-making using multi-agent Large Language Model (LLM) architectures.

The system orchestrates a sequential loan approval process where four distinct agents evaluate applications in a chained manner:
1.  **Risk Manager**: Assesses financial viability and default probability.
2.  **Consumer Advocate**: Protects borrower interests and flags predatory terms.
3.  **Regulatory Compliance**: Ensures statutory adherence (e.g., ECOA, Fair Lending).
4.  **Data Science**: Provides data-driven risk scoring and alternative data analysis.

## Key Features
*   **Asynchronous Engine**: Built on `vLLM` and `asyncio`, capable of processing 2,000+ chains concurrently.
*   **Strict JSON Schema**: Enforces a deep-nested JSON structure with zero tolerance for null values or format errors.
*   **Legacy Compatibility**: Includes tools to convert high-speed "Turbo" outputs into the project's legacy analysis format.
*   **Automated Verification**: Integrated suite of scripts to verify data integrity and schema compliance.

## Repository Structure

```
fairwatch/
├── orchestrator.py            # Main async engine (vLLM client)
├── vllm_client.py             # Native asyncio adapter for vLLM
├── agents/                    # Persona definitions and prompt templates
├── data/                      # Input datasets (CSV)
├── scripts/                   # Utility suite
│   ├── convert_results.py     # Conversion to legacy format
│   ├── verify_results.py      # Strict schema & null verification
│   └── reprocess_failures.py  # Recovery tool for failed chains
└── docs/                      # Technical reports
```

## Installation

1.  **Clone the repository**
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **vLLM Setup**:
    Ensure you have a running vLLM server (e.g., Llama 3.2 3B).
    *   Default Port: `8000` (Configurable via `--port`)

## Usage

### Quick Start (Demo)
Run a small batch of 5 records to verify the pipeline:

```bash
python orchestrator.py \
  --ordering "regulatory,risk_manager,consumer_advocate,data_science" \
  --limit 5 \
  --concurrency 10 \
  --port 8005 \
  --output "demo_results.json"
```

### Full Simulation
To run the full dataset (e.g., 5760 records) with high concurrency:

```bash
python orchestrator.py \
  --ordering "regulatory,risk_manager,consumer_advocate,data_science" \
  --concurrency 1000 \
  --port 8005 \
  --output "full_run_results.json"
```

### Post-Processing & Verification
After generation, convert to the analytical legacy format and verify:

```bash
# Convert
python scripts/convert_results.py --input "full_run_results.json"

# Verify
python scripts/verify_results.py --file "full_run_results_legacy.json"
```

## License
MIT License - FairWatch Team
