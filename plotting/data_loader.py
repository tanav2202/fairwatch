"""
Data loader module for FairWatch results analysis.
Loads and preprocesses all result files from baseline, sequential, and parallel experiments.

Updated for new folder structure:
- 70bresults/baseline/
- 70bresults/sequential/
- 70bresults/parallel/
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


class FairWatchDataLoader:
    """Load and preprocess FairWatch experiment results."""

    def __init__(self, results_dir: str = "70bresults"):
        self.results_dir = Path(results_dir)

    def load_json_file(self, filepath: Path) -> Optional[Dict]:
        """Load a single JSON file."""
        try:
            file_size = filepath.stat().st_size
            if file_size > 150 * 1024 * 1024:  # Skip files > 150MB
                print(
                    f"  Warning: Skipping large file {filepath.name} ({file_size / 1024 / 1024:.1f} MB)"
                )
                return None
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")
            return None

    def extract_agent_order_from_filename(self, filename: str) -> List[str]:
        """Extract agent order from sequential filename."""
        name = filename.replace("sequential_", "").replace(".json", "")
        name = (
            name.replace("_legacy", "").replace("_final", "").replace("_formatted", "")
        )

        # Build ordered list by finding positions
        agents_found = []

        # Check for full names first (longer patterns first to avoid partial matches)
        full_patterns = [
            ("consumer_advocate", "Consumer Advocate"),
            ("data_science", "Data Science"),
            ("risk_manager", "Risk Manager"),
            ("regulatory", "Regulatory"),
        ]

        temp_name = name
        for pattern, display in full_patterns:
            idx = temp_name.find(pattern)
            if idx >= 0:
                agents_found.append((idx, display))
                # Replace with spaces to avoid re-matching
                temp_name = (
                    temp_name[:idx]
                    + " " * len(pattern)
                    + temp_name[idx + len(pattern) :]
                )

        # Check for short names in remaining text
        short_patterns = [
            ("consumer", "Consumer Advocate"),
            ("data", "Data Science"),
            ("risk", "Risk Manager"),
        ]

        for pattern, display in short_patterns:
            if display not in [a[1] for a in agents_found]:
                idx = temp_name.find(pattern)
                if idx >= 0:
                    agents_found.append((idx, display))

        # Sort by position and deduplicate
        agents_found.sort(key=lambda x: x[0])
        seen = set()
        result = []
        for _, agent in agents_found:
            if agent not in seen:
                seen.add(agent)
                result.append(agent)

        return result if result else ["Unknown"]

    def load_all_results(self) -> Dict:
        """Load all results from baseline/, sequential/, parallel/ directories."""
        all_data = {"baselines": [], "sequential": [], "parallel": [], "swarm": []}

        # Load baseline files
        baseline_dir = self.results_dir / "baseline"
        if baseline_dir.exists():
            print(f"Loading from {baseline_dir}...")
            for file in sorted(baseline_dir.glob("*.json")):
                data = self.load_json_file(file)
                if data is None:
                    continue

                filename = file.name

                if "swarm" in filename:
                    all_data["swarm"].append(
                        {"source": "baseline", "filename": filename, "data": data}
                    )
                else:
                    # Extract agent name: baseline_consumer_advocate_70b_formatted.json -> Consumer Advocate
                    agent = (
                        filename.replace("baseline_", "")
                        .replace("_70b_formatted.json", "")
                        .replace("_70b.json", "")
                    )
                    agent = agent.replace("_", " ").title()
                    all_data["baselines"].append(
                        {
                            "source": "baseline",
                            "filename": filename,
                            "agent": agent,
                            "data": data,
                        }
                    )
            print(
                f"  Loaded {len(all_data['baselines'])} baseline, {len(all_data['swarm'])} swarm files"
            )

        # Load sequential files
        sequential_dir = self.results_dir / "sequential"
        if sequential_dir.exists():
            print(f"Loading from {sequential_dir}...")
            for file in sorted(sequential_dir.glob("*.json")):
                data = self.load_json_file(file)
                if data is None:
                    continue

                order = self.extract_agent_order_from_filename(file.name)
                all_data["sequential"].append(
                    {
                        "source": "sequential",
                        "filename": file.name,
                        "agent_order": order,
                        "data": data,
                    }
                )
            print(f"  Loaded {len(all_data['sequential'])} sequential files")

        # Load parallel files
        parallel_dir = self.results_dir / "parallel"
        if parallel_dir.exists():
            print(f"Loading from {parallel_dir}...")
            for file in sorted(parallel_dir.glob("*.json")):
                data = self.load_json_file(file)
                if data is None:
                    continue

                all_data["parallel"].append(
                    {"source": "parallel", "filename": file.name, "data": data}
                )
            print(f"  Loaded {len(all_data['parallel'])} parallel files")

        return all_data

    def flatten_baseline_results(self, baseline_data: List[Dict]) -> pd.DataFrame:
        """Flatten baseline results into a DataFrame."""
        records = []

        for item in baseline_data:
            agent = item.get("agent", "unknown")
            source = item.get("source", "unknown")

            if "data" not in item or "results" not in item["data"]:
                continue

            for result in item["data"]["results"]:
                # Parse input to get demographic info
                input_str = result.get("input", "")
                base_info = self.parse_prompt(input_str)

                record = {
                    "source": source,
                    "agent": agent,
                    "chain_id": result.get("chain_id"),
                    "mode": "baseline",
                    **base_info,
                }

                # New format uses "decisions" dict
                decisions = result.get("decisions", {})
                if decisions:
                    # Get the agent's decision (there's only one for baseline)
                    agent_key = list(decisions.keys())[0] if decisions else None
                    if agent_key:
                        bd = decisions[agent_key]
                        record["approval_decision"] = bd.get("approval_decision")
                        record["approval_type"] = bd.get("approval_type")
                        record["interest_rate"] = bd.get("interest_rate")
                        record["confidence_probability"] = bd.get(
                            "confidence_probability"
                        )
                        record["confidence_level"] = bd.get("confidence_level")
                        records.append(record)
                else:
                    # Old format uses "business_decision"
                    bd = result.get("business_decision", {})
                    if bd:
                        record["approval_decision"] = bd.get("approval_decision")
                        record["approval_type"] = bd.get("approval_type")
                        record["interest_rate"] = bd.get("interest_rate")
                        record["confidence_probability"] = bd.get(
                            "confidence_probability"
                        )
                        record["confidence_level"] = bd.get("confidence_level")
                        records.append(record)

        return pd.DataFrame(records)

    def flatten_swarm_results(self, swarm_data: List[Dict]) -> pd.DataFrame:
        """Flatten swarm baseline results into a DataFrame."""
        records = []

        for item in swarm_data:
            source = item.get("source", "unknown")

            if "data" not in item or "results" not in item["data"]:
                continue

            for result in item["data"]["results"]:
                prompt = result.get("prompt", "")
                base_info = self.parse_prompt(prompt)

                agent_outputs = result.get("agent_outputs", {})

                # Skip entries with empty agent_outputs
                if not agent_outputs:
                    continue

                for agent_key, agent_data in agent_outputs.items():
                    if not agent_data:
                        continue
                    record = {
                        "source": source,
                        "prompt_id": result.get("prompt_id"),
                        "mode": "swarm",
                        "agent": agent_data.get(
                            "agent_name", agent_key.replace("_", " ").title()
                        ),
                        **base_info,
                        "approval_decision": agent_data.get("approval_decision"),
                        "approval_type": agent_data.get("approval_type"),
                        "interest_rate": agent_data.get("interest_rate"),
                        "confidence_probability": agent_data.get(
                            "confidence_probability"
                        ),
                        "confidence_level": agent_data.get("confidence_level"),
                    }
                    records.append(record)

        return pd.DataFrame(records)

    def flatten_sequential_results(self, sequential_data: List[Dict]) -> pd.DataFrame:
        """Flatten sequential results into a DataFrame."""
        records = []

        for item in sequential_data:
            order = item.get("agent_order", [])
            source = item.get("source", "unknown")
            filename = item.get("filename", "unknown")
            order_str = (
                "_".join([a.replace(" ", "_") for a in order])
                if order
                else filename.replace(".json", "")
            )

            if "data" not in item:
                continue

            # Handle both formats:
            # 1. New format: {"results": [...]}
            # 2. Legacy format: [...] (list directly)
            raw_data = item["data"]
            if isinstance(raw_data, list):
                # Legacy format - data is a list directly
                results_list = raw_data
            elif isinstance(raw_data, dict) and "results" in raw_data:
                # New format - data has "results" key
                results_list = raw_data["results"]
            else:
                continue

            for result in results_list:
                # Handle different input formats
                # Legacy: "initial_prompt" field with JSON string
                # New: "input" field with JSON string
                input_str = result.get("input") or result.get("initial_prompt", "{}")
                try:
                    input_data = json.loads(input_str)
                except:
                    input_data = {}

                base_record = {
                    "source": source,
                    "filename": filename,
                    "chain_id": result.get("chain_id"),
                    "agent_order": order_str,
                    "agent_order_list": order,
                    "name": input_data.get("name"),
                    "ethnicity_signal": input_data.get("ethnicity_signal"),
                    "credit_score": input_data.get("credit_score"),
                    "visa_status": input_data.get("visa_status"),
                    "income": input_data.get("income"),
                    "age": input_data.get("age"),
                    "loan_amount": input_data.get("loan_amount"),
                    "dti_ratio": input_data.get("dti_ratio"),
                }

                # Handle different decision formats
                # New format: {"decisions": {"consumer_advocate": {...}, ...}}
                # Legacy format: {"conversation_history": [...], "all_agent_outputs": [...]}
                decisions = result.get("decisions", {})

                if decisions:
                    # New format with decisions dict
                    for i, agent_name in enumerate(order):
                        agent_key = agent_name.lower().replace(" ", "_")
                        agent_data = decisions.get(agent_key, {})

                        if not agent_data:
                            continue

                        record = base_record.copy()
                        record["agent"] = agent_name
                        record["agent_position"] = i + 1
                        record["is_final_agent"] = i == len(order) - 1
                        record["approval_decision"] = agent_data.get(
                            "approval_decision"
                        )
                        record["approval_type"] = agent_data.get("approval_type")
                        record["interest_rate"] = agent_data.get("interest_rate")
                        record["confidence_probability"] = agent_data.get(
                            "confidence_probability"
                        )
                        record["confidence_level"] = agent_data.get("confidence_level")

                        records.append(record)
                else:
                    # Legacy format with conversation_history or all_agent_outputs
                    agent_outputs = result.get("all_agent_outputs", [])
                    if not agent_outputs:
                        # Try to extract from conversation_history
                        conv_history = result.get("conversation_history", [])
                        agent_outputs = [
                            turn.get("output", {})
                            for turn in conv_history
                            if turn.get("output")
                        ]

                    for i, agent_data in enumerate(agent_outputs):
                        if not agent_data:
                            continue

                        agent_name = agent_data.get("agent_name", "Unknown")
                        record = base_record.copy()
                        record["agent"] = agent_name
                        record["agent_position"] = i + 1
                        record["is_final_agent"] = i == len(agent_outputs) - 1
                        record["approval_decision"] = agent_data.get(
                            "approval_decision"
                        )
                        record["approval_type"] = agent_data.get("approval_type")
                        record["interest_rate"] = agent_data.get("interest_rate")
                        record["confidence_probability"] = agent_data.get(
                            "confidence_probability"
                        )
                        record["confidence_level"] = agent_data.get("confidence_level")

                        records.append(record)

        return pd.DataFrame(records)

    def flatten_parallel_results(self, parallel_data: List[Dict]) -> pd.DataFrame:
        """Flatten parallel results into a DataFrame."""
        records = []

        for item in parallel_data:
            source = item.get("source", "unknown")

            if "data" not in item or "results" not in item["data"]:
                continue

            for result in item["data"]["results"]:
                prompt = result.get("prompt", "")
                base_info = self.parse_prompt(prompt)

                base_record = {
                    "source": source,
                    "prompt_id": result.get("prompt_id"),
                    "mode": "parallel",
                    **base_info,
                }

                agent_outputs = result.get("agent_outputs", {})
                for agent_key, agent_data in agent_outputs.items():
                    if not agent_data:
                        continue
                    record = base_record.copy()
                    record["agent"] = agent_data.get(
                        "agent_name", agent_key.replace("_", " ").title()
                    )
                    record["approval_decision"] = agent_data.get("approval_decision")
                    record["approval_type"] = agent_data.get("approval_type")
                    record["interest_rate"] = agent_data.get("interest_rate")
                    record["confidence_probability"] = agent_data.get(
                        "confidence_probability"
                    )
                    record["confidence_level"] = agent_data.get("confidence_level")
                    records.append(record)

                bd = result.get("business_decision", {})
                if bd:
                    record = base_record.copy()
                    record["agent"] = "Business Decision"
                    record["approval_decision"] = bd.get("approval_decision")
                    record["approval_type"] = bd.get("approval_type")
                    record["interest_rate"] = bd.get("interest_rate")
                    record["confidence_probability"] = bd.get("confidence_probability")
                    record["confidence_level"] = bd.get("confidence_level")
                    records.append(record)

        return pd.DataFrame(records)

    def parse_prompt(self, prompt: str) -> Dict:
        """Extract applicant information from prompt text or JSON input."""
        info = {}

        # Try to parse as JSON first (new format)
        try:
            data = json.loads(prompt)
            if isinstance(data, dict):
                info["name"] = data.get("name")
                info["age"] = data.get("age")
                info["income"] = data.get("income")
                info["credit_score"] = data.get("credit_score")
                info["visa_status"] = data.get("visa_status")
                info["loan_amount"] = data.get("loan_amount")
                info["dti_ratio"] = data.get("dti_ratio")
                info["ethnicity_signal"] = data.get(
                    "ethnicity_signal", self.infer_ethnicity(data.get("name", ""))
                )
                return info
        except (json.JSONDecodeError, TypeError):
            pass  # Fall through to text parsing

        # Text parsing (old format)
        name_match = re.search(r"My name is ([^.]+)\.", prompt)
        info["name"] = name_match.group(1) if name_match else None

        age_match = re.search(r"I'm (\d+) years old", prompt)
        info["age"] = int(age_match.group(1)) if age_match else None

        income_match = re.search(r"\$([0-9,]+) per year", prompt)
        if income_match:
            info["income"] = int(income_match.group(1).replace(",", ""))
        else:
            info["income"] = None

        credit_match = re.search(r"credit score is (\d+)", prompt)
        info["credit_score"] = int(credit_match.group(1)) if credit_match else None

        visa_match = re.search(r"I am a ([^.]+)\.", prompt)
        info["visa_status"] = visa_match.group(1) if visa_match else None

        loan_match = re.search(r"\$([0-9,]+) personal loan", prompt)
        if loan_match:
            info["loan_amount"] = int(loan_match.group(1).replace(",", ""))
        else:
            info["loan_amount"] = None

        info["ethnicity_signal"] = self.infer_ethnicity(info.get("name", ""))

        return info

    def infer_ethnicity(self, name: str) -> str:
        """Infer ethnicity signal from name."""
        if not name:
            return "Unknown"

        ethnicity_map = {
            "Emily Anderson": "White_Signal",
            "Lakisha Washington": "Black_Signal",
            "Carlos Martinez": "Hispanic_Signal",
            "Lei Chen": "Asian_Signal",
            "Linh Nguyen": "Asian_Signal",
            "Jamal Williams": "Black_Signal",
            "Maria Garcia": "Hispanic_Signal",
            "John Smith": "White_Signal",
        }

        return ethnicity_map.get(name, "Unknown")

    def get_combined_df(self) -> pd.DataFrame:
        """Get a combined DataFrame with all results."""
        all_data = self.load_all_results()

        dfs = []

        if all_data["baselines"]:
            df_baseline = self.flatten_baseline_results(all_data["baselines"])
            df_baseline["experiment_type"] = "baseline"
            dfs.append(df_baseline)
            print(f"  Baseline records: {len(df_baseline)}")

        if all_data["swarm"]:
            df_swarm = self.flatten_swarm_results(all_data["swarm"])
            df_swarm["experiment_type"] = "baseline"  # Treat swarm as baseline
            dfs.append(df_swarm)
            print(f"  Swarm records: {len(df_swarm)}")

        if all_data["sequential"]:
            df_seq = self.flatten_sequential_results(all_data["sequential"])
            df_seq["experiment_type"] = "sequential"
            dfs.append(df_seq)
            print(f"  Sequential records: {len(df_seq)}")

        if all_data["parallel"]:
            df_par = self.flatten_parallel_results(all_data["parallel"])
            df_par["experiment_type"] = "parallel"
            dfs.append(df_par)
            print(f"  Parallel records: {len(df_par)}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"\nTotal combined records: {len(combined)}")
            return combined
        return pd.DataFrame()


def load_data(results_dir: str = "70bresults") -> Tuple[pd.DataFrame, Dict]:
    """Convenience function to load all data."""
    loader = FairWatchDataLoader(results_dir)
    all_data = loader.load_all_results()
    combined_df = loader.get_combined_df()
    return combined_df, all_data


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FairWatch Data Loader")
    print("=" * 60)

    df, raw = load_data("70bresults")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total records loaded: {len(df)}")
    print(f"\nExperiment types:\n{df['experiment_type'].value_counts()}")
    print(f"\nAgents:\n{df['agent'].value_counts()}")

    if "agent_order" in df.columns:
        seq_df = df[df["experiment_type"] == "sequential"]
        if len(seq_df) > 0:
            seq_orders = seq_df["agent_order"].dropna().unique()
            print(f"\nUnique sequential orderings: {len(seq_orders)}")
            for order in sorted(seq_orders)[:15]:
                print(f"  - {order}")
            if len(seq_orders) > 15:
                print(f"  ... and {len(seq_orders) - 15} more")
