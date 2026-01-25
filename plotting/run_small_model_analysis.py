"""
Run analysis for small model outputs (outputs_simple and outputs_complex).
Handles llama3.2 and mistral:latest models separately.
"""

import sys
import json
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class SmallModelDataLoader:
    """Load data from outputs_simple and outputs_complex folders."""

    def __init__(self, base_dir: str, model_name: str):
        """
        Args:
            base_dir: Either 'outputs_simple' or 'outputs_complex'
            model_name: Either 'llama3.2' or 'mistral:latest'
        """
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        self.model_dir = self.base_dir / model_name

    def load_json_file(self, filepath: Path) -> Optional[Dict]:
        """Load a single JSON file, handling Git LFS pointers."""
        try:
            # Check if it's a Git LFS pointer
            with open(filepath, "r") as f:
                first_line = f.readline()
                if first_line.startswith("version https://git-lfs"):
                    # Try to smudge the LFS file
                    print(f"  Smudging LFS file: {filepath.name}...")
                    import subprocess

                    try:
                        result = subprocess.run(
                            ["git", "lfs", "smudge"],
                            stdin=open(filepath, "rb"),
                            capture_output=True,
                            cwd=filepath.parent,
                            timeout=120,  # 2 minute timeout for large files
                        )
                        if result.returncode == 0 and result.stdout:
                            return json.loads(result.stdout.decode("utf-8"))
                        else:
                            print(
                                f"    Failed to smudge {filepath.name}: {result.stderr.decode()[:100]}"
                            )
                            return None
                    except subprocess.TimeoutExpired:
                        print(f"    Timeout smudging {filepath.name}")
                        return None
                    except Exception as e:
                        print(f"    Error smudging {filepath.name}: {e}")
                        return None
                f.seek(0)
                return json.load(f)
        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")
            return None

    def extract_agent_order_from_filename(self, filename: str) -> List[str]:
        """Extract agent order from sequential filename."""
        # sequential_consumer_data_regulatory_risk.json
        name = filename.replace("sequential_", "").replace(".json", "")

        agent_map = {
            "consumer": "Consumer Advocate",
            "data": "Data Science",
            "regulatory": "Regulatory",
            "risk": "Risk Manager",
        }

        # Split by underscore and map
        parts = name.split("_")
        order = []
        for part in parts:
            if part in agent_map:
                order.append(agent_map[part])

        return order if order else ["Unknown"]

    def parse_prompt(self, prompt: str) -> Dict:
        """Extract applicant information from prompt text or JSON input."""
        info = {}

        # Check if it starts with "Application Data:" (sequential format)
        if prompt.startswith("Application Data:"):
            try:
                json_str = prompt.replace("Application Data:", "").strip()
                data = json.loads(json_str)
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
            except:
                pass

        # Try to parse as JSON first
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
        except:
            pass

        # Text parsing
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

    def load_baseline_results(self) -> pd.DataFrame:
        """Load baseline results."""
        records = []
        baseline_dir = self.model_dir / "baseline"

        if not baseline_dir.exists():
            print(f"  No baseline directory found: {baseline_dir}")
            return pd.DataFrame()

        for file in sorted(baseline_dir.glob("*.json")):
            data = self.load_json_file(file)
            if data is None:
                continue

            # Extract agent name from filename: consumer_advocate_baseline.json
            agent = file.name.replace("_baseline.json", "").replace("_", " ").title()

            for result in data.get("results", []):
                prompt = result.get("prompt", "")
                base_info = self.parse_prompt(prompt)

                output = result.get("output", {})
                record = {
                    "agent": agent,
                    "prompt_id": result.get("prompt_id"),
                    "mode": "baseline",
                    **base_info,
                    "approval_decision": output.get("approval_decision"),
                    "approval_type": output.get("approval_type"),
                    "interest_rate": output.get("interest_rate"),
                    "confidence_probability": output.get("confidence_probability"),
                    "confidence_level": output.get("confidence_level"),
                }
                records.append(record)

        df = pd.DataFrame(records)
        if len(df) > 0:
            df["experiment_type"] = "baseline"
        return df

    def load_sequential_results(self) -> pd.DataFrame:
        """Load sequential results."""
        records = []
        sequential_dir = self.model_dir / "sequential"

        if not sequential_dir.exists():
            print(f"  No sequential directory found: {sequential_dir}")
            return pd.DataFrame()

        for file in sorted(sequential_dir.glob("*.json")):
            data = self.load_json_file(file)
            if data is None:
                continue

            order = self.extract_agent_order_from_filename(file.name)
            order_str = "_".join([a.replace(" ", "_") for a in order])

            for result in data.get("results", []):
                # Parse the initial prompt
                initial_prompt = result.get("initial_prompt", "")
                base_info = self.parse_prompt(initial_prompt)

                base_record = {
                    "filename": file.name,
                    "chain_id": result.get("prompt_id"),
                    "agent_order": order_str,
                    "agent_order_list": order,
                    **base_info,
                }

                # Get all agent outputs
                all_outputs = result.get("all_agent_outputs", [])
                if not all_outputs:
                    # Try conversation_history
                    conv_history = result.get("conversation_history", [])
                    all_outputs = [
                        turn.get("output", {})
                        for turn in conv_history
                        if turn.get("output")
                    ]

                for i, agent_data in enumerate(all_outputs):
                    if not agent_data:
                        continue

                    agent_name = agent_data.get("agent_name", "Unknown")
                    record = base_record.copy()
                    record["agent"] = agent_name
                    record["agent_position"] = i + 1
                    record["is_final_agent"] = i == len(all_outputs) - 1
                    record["approval_decision"] = agent_data.get("approval_decision")
                    record["approval_type"] = agent_data.get("approval_type")
                    record["interest_rate"] = agent_data.get("interest_rate")
                    record["confidence_probability"] = agent_data.get(
                        "confidence_probability"
                    )
                    record["confidence_level"] = agent_data.get("confidence_level")
                    records.append(record)

        df = pd.DataFrame(records)
        if len(df) > 0:
            df["experiment_type"] = "sequential"
        return df

    def load_parallel_results(self) -> pd.DataFrame:
        """Load parallel results."""
        records = []
        parallel_dir = self.model_dir / "parallel"

        if not parallel_dir.exists():
            print(f"  No parallel directory found: {parallel_dir}")
            return pd.DataFrame()

        for file in sorted(parallel_dir.glob("*.json")):
            data = self.load_json_file(file)
            if data is None:
                continue

            for result in data.get("results", []):
                prompt = result.get("prompt", "")
                base_info = self.parse_prompt(prompt)

                base_record = {
                    "prompt_id": result.get("prompt_id"),
                    "mode": "parallel",
                    **base_info,
                }

                # Get agent outputs
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

                # Get business decision
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

        df = pd.DataFrame(records)
        if len(df) > 0:
            df["experiment_type"] = "parallel"
        return df

    def get_combined_df(self) -> pd.DataFrame:
        """Get combined DataFrame with all results."""
        dfs = []

        print(f"\nLoading {self.model_name} from {self.base_dir}...")

        df_baseline = self.load_baseline_results()
        if len(df_baseline) > 0:
            dfs.append(df_baseline)
            print(f"  Baseline records: {len(df_baseline)}")

        df_seq = self.load_sequential_results()
        if len(df_seq) > 0:
            dfs.append(df_seq)
            print(f"  Sequential records: {len(df_seq)}")

        df_par = self.load_parallel_results()
        if len(df_par) > 0:
            dfs.append(df_par)
            print(f"  Parallel records: {len(df_par)}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"  Total records: {len(combined)}")
            return combined

        return pd.DataFrame()


def normalize_decision(decision):
    """Normalize approval decision strings."""
    if pd.isna(decision):
        return decision
    decision = str(decision).lower().strip()
    if decision in ["approve", "approval", "approval_with_caution"]:
        return "approve"
    if decision in ["deny", "denial", "denied"]:
        return "deny"
    if "manual" in decision or "review" in decision:
        return "manual_review"
    return decision


def create_baseline_analysis(df: pd.DataFrame, agent_name: str, output_dir: Path):
    """Create analysis plots for a single baseline agent."""
    df_agent = df[
        (df["experiment_type"] == "baseline") & (df["agent"] == agent_name)
    ].copy()

    if df_agent.empty:
        print(f"  No data for baseline {agent_name}")
        return

    agent_dir = output_dir / "baselines" / agent_name.lower().replace(" ", "_")
    agent_dir.mkdir(parents=True, exist_ok=True)

    df_agent["approval_norm"] = df_agent["approval_decision"].apply(normalize_decision)
    df_valid = df_agent[df_agent["approval_norm"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_norm"] == "approve").astype(int)
    df_approved = df_valid[df_valid["approval_norm"] == "approve"]

    if len(df_valid) == 0:
        print(f"  No valid approve/deny decisions for {agent_name}")
        return

    # Figure 1: Overview Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"{agent_name} - Baseline Analysis\n(n={len(df_valid):,})",
        fontsize=14,
        fontweight="bold",
    )

    # 1.1 Approval Rate by Ethnicity
    ax = axes[0, 0]
    if df_valid["ethnicity_signal"].notna().any():
        eth_rates = (
            df_valid.groupby("ethnicity_signal")["approved"].mean().sort_values()
        )
        colors = [
            (
                "#e74c3c"
                if r < eth_rates.mean() - 0.05
                else "#27ae60" if r > eth_rates.mean() + 0.05 else "#3498db"
            )
            for r in eth_rates
        ]
        bars = ax.barh(eth_rates.index, eth_rates.values, color=colors)
        ax.set_xlabel("Approval Rate")
        ax.set_title("Approval Rate by Ethnicity")
        ax.set_xlim(0, 1)
        for bar, rate in zip(bars, eth_rates.values):
            ax.text(
                rate + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}",
                va="center",
                fontsize=9,
            )

    # 1.2 Interest Rate Distribution
    ax = axes[0, 1]
    if len(df_approved) > 0 and df_approved["interest_rate"].notna().any():
        valid_ir = df_approved["interest_rate"].dropna()
        valid_ir = valid_ir[(valid_ir > 0) & (valid_ir < 50)]
        if len(valid_ir) > 0:
            sns.histplot(valid_ir, ax=ax, bins=20, color="#3498db", edgecolor="white")
            ax.axvline(
                valid_ir.mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {valid_ir.mean():.1f}%",
            )
            ax.legend()
    ax.set_xlabel("Interest Rate (%)")
    ax.set_title("Interest Rate Distribution")

    # 1.3 Approval Type Distribution
    ax = axes[0, 2]
    if df_valid["approval_type"].notna().any():
        type_counts = df_valid["approval_type"].value_counts()
        colors_type = {
            "STANDARD_TERMS": "#27ae60",
            "SUBOPTIMAL_TERMS": "#f39c12",
            "MANUAL_REVIEW": "#3498db",
            "DENIAL": "#e74c3c",
        }
        ax.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.1f%%",
            colors=[
                colors_type.get(str(t).upper(), "#95a5a6") for t in type_counts.index
            ],
        )
        ax.set_title("Approval Type Distribution")

    # 1.4 Approval by Credit Score
    ax = axes[1, 0]
    if df_valid["credit_score"].notna().any():
        df_valid["credit_tier"] = pd.cut(
            df_valid["credit_score"],
            bins=[0, 580, 670, 740, 800, 900],
            labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        )
        credit_rates = df_valid.groupby("credit_tier", observed=True)["approved"].mean()
        if len(credit_rates) > 0:
            ax.bar(
                credit_rates.index,
                credit_rates.values,
                color="#3498db",
                edgecolor="white",
            )
        ax.set_ylabel("Approval Rate")
        ax.set_title("Approval Rate by Credit Score")
        ax.set_ylim(0, 1)

    # 1.5 Approval by Age Group
    ax = axes[1, 1]
    if df_valid["age"].notna().any():
        df_valid["age_group"] = pd.cut(
            df_valid["age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "55+"],
        )
        age_rates = df_valid.groupby("age_group", observed=True)["approved"].mean()
        if len(age_rates) > 0:
            ax.bar(
                age_rates.index, age_rates.values, color="#9b59b6", edgecolor="white"
            )
        ax.set_ylabel("Approval Rate")
        ax.set_title("Approval Rate by Age Group")
        ax.set_ylim(0, 1)

    # 1.6 Confidence Distribution
    ax = axes[1, 2]
    if df_valid["confidence_level"].notna().any():
        conf_counts = df_valid["confidence_level"].value_counts()
        colors_conf = {"high": "#27ae60", "medium": "#f39c12", "low": "#e74c3c"}
        conf_order = ["high", "medium", "low"]
        conf_counts = conf_counts.reindex(
            [c for c in conf_order if c in conf_counts.index]
        )
        if len(conf_counts) > 0:
            ax.bar(
                conf_counts.index,
                conf_counts.values,
                color=[colors_conf.get(c, "#95a5a6") for c in conf_counts.index],
            )
        ax.set_ylabel("Count")
        ax.set_title("Confidence Level Distribution")

    plt.tight_layout()
    plt.savefig(agent_dir / "overview_dashboard.png", dpi=300, bbox_inches="tight")
    plt.savefig(agent_dir / "overview_dashboard.pdf", bbox_inches="tight")
    plt.close()

    # Figure 2: Ethnicity Bias Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    if (
        df_valid["ethnicity_signal"].notna().any()
        and df_valid["credit_score"].notna().any()
    ):
        df_valid["credit_tier"] = pd.cut(
            df_valid["credit_score"],
            bins=[0, 580, 670, 740, 800, 900],
            labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        )
        pivot = df_valid.pivot_table(
            values="approved",
            index="ethnicity_signal",
            columns="credit_tier",
            aggfunc="mean",
            observed=True,
        )
        if not pivot.empty:
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1%",
                cmap="RdYlGn",
                ax=ax,
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Approval Rate"},
            )
            ax.set_title(f"{agent_name}: Approval Rate by Ethnicity and Credit Score")

    plt.tight_layout()
    plt.savefig(
        agent_dir / "ethnicity_credit_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(agent_dir / "ethnicity_credit_heatmap.pdf", bbox_inches="tight")
    plt.close()

    # Figure 3: Interest Rate by Demographics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if len(df_approved) > 0:
        ax = axes[0]
        if (
            df_approved["ethnicity_signal"].notna().any()
            and df_approved["interest_rate"].notna().any()
        ):
            valid_rates = df_approved[df_approved["interest_rate"].between(0, 50)]
            if len(valid_rates) > 0:
                eth_ir = (
                    valid_rates.groupby("ethnicity_signal")["interest_rate"]
                    .agg(["mean", "std"])
                    .sort_values("mean")
                )
                ax.barh(
                    eth_ir.index,
                    eth_ir["mean"],
                    xerr=eth_ir["std"].fillna(0),
                    color="#3498db",
                    capsize=3,
                )
                ax.set_xlabel("Interest Rate (%)")
                ax.set_title("Average Interest Rate by Ethnicity")

        ax = axes[1]
        if (
            df_approved["credit_score"].notna().any()
            and df_approved["interest_rate"].notna().any()
        ):
            valid_rates = df_approved[df_approved["interest_rate"].between(0, 50)]
            if len(valid_rates) > 0:
                valid_rates["credit_tier"] = pd.cut(
                    valid_rates["credit_score"],
                    bins=[0, 580, 670, 740, 800, 900],
                    labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                )
                credit_ir = valid_rates.groupby("credit_tier", observed=True)[
                    "interest_rate"
                ].agg(["mean", "std"])
                if len(credit_ir) > 0:
                    ax.bar(
                        credit_ir.index,
                        credit_ir["mean"],
                        yerr=credit_ir["std"].fillna(0),
                        color="#9b59b6",
                        capsize=3,
                    )
                    ax.set_ylabel("Interest Rate (%)")
                    ax.set_title("Average Interest Rate by Credit Score")

    plt.tight_layout()
    plt.savefig(
        agent_dir / "interest_rate_demographics.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(agent_dir / "interest_rate_demographics.pdf", bbox_inches="tight")
    plt.close()

    # Save summary
    summary = {
        "agent": agent_name,
        "total_records": len(df_valid),
        "approval_rate": (
            float(df_valid["approved"].mean()) if len(df_valid) > 0 else None
        ),
        "avg_interest_rate": (
            float(df_approved["interest_rate"].mean())
            if len(df_approved) > 0 and df_approved["interest_rate"].notna().any()
            else None
        ),
    }
    with open(agent_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved {agent_name} baseline analysis")


def create_sequential_analysis(df: pd.DataFrame, ordering: str, output_dir: Path):
    """Create analysis plots for a single sequential ordering."""
    df_seq = df[
        (df["experiment_type"] == "sequential") & (df["agent_order"] == ordering)
    ].copy()

    if df_seq.empty:
        print(f"  No data for ordering {ordering}")
        return

    folder_name = ordering.lower().replace(" ", "_")
    seq_dir = output_dir / "sequential" / folder_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    df_seq["approval_norm"] = df_seq["approval_decision"].apply(normalize_decision)
    df_valid = df_seq[df_seq["approval_norm"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_norm"] == "approve").astype(int)
    df_approved = df_valid[df_valid["approval_norm"] == "approve"]

    if len(df_valid) == 0:
        print(f"  No valid approve/deny decisions for {ordering}")
        return

    df_final = df_valid[df_valid["is_final_agent"] == True].copy()
    agents_display = ordering.replace("_", " → ")

    # Figure 1: Chain Overview Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Sequential Chain Analysis\n{agents_display}\n(n={len(df_final):,} chains)",
        fontsize=12,
        fontweight="bold",
    )

    # 1.1 Approval Rate by Agent Position
    ax = axes[0, 0]
    pos_rates = df_valid.groupby("agent_position")["approved"].mean()
    ax.plot(
        pos_rates.index,
        pos_rates.values,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#3498db",
    )
    ax.fill_between(pos_rates.index, pos_rates.values, alpha=0.3, color="#3498db")
    ax.set_xlabel("Agent Position")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rate Evolution Through Chain")
    ax.set_ylim(0, 1)

    # 1.2 Final Approval Rate by Ethnicity
    ax = axes[0, 1]
    if len(df_final) > 0 and df_final["ethnicity_signal"].notna().any():
        eth_rates = (
            df_final.groupby("ethnicity_signal")["approved"].mean().sort_values()
        )
        colors = [
            (
                "#e74c3c"
                if r < eth_rates.mean() - 0.05
                else "#27ae60" if r > eth_rates.mean() + 0.05 else "#3498db"
            )
            for r in eth_rates
        ]
        bars = ax.barh(eth_rates.index, eth_rates.values, color=colors)
        ax.set_xlabel("Approval Rate")
        ax.set_title("Final Approval Rate by Ethnicity")
        ax.set_xlim(0, 1)

    # 1.3 Interest Rate Evolution
    ax = axes[0, 2]
    if df_approved["interest_rate"].notna().any():
        valid_ir = df_approved[df_approved["interest_rate"].between(0, 50)]
        if len(valid_ir) > 0:
            ir_by_pos = valid_ir.groupby("agent_position")["interest_rate"].agg(
                ["mean", "std"]
            )
            ax.errorbar(
                ir_by_pos.index,
                ir_by_pos["mean"],
                yerr=ir_by_pos["std"].fillna(0),
                marker="s",
                linewidth=2,
                markersize=8,
                capsize=5,
                color="#e74c3c",
            )
            ax.set_xlabel("Agent Position")
            ax.set_ylabel("Interest Rate (%)")
            ax.set_title("Interest Rate Evolution")

    # 1.4 Agent Agreement Analysis
    ax = axes[1, 0]
    chain_decisions = df_valid.groupby("chain_id")["approval_norm"].agg(
        lambda x: x.nunique()
    )
    agreement_counts = chain_decisions.value_counts().sort_index()
    ax.bar(agreement_counts.index, agreement_counts.values, color="#9b59b6")
    ax.set_xlabel("Number of Unique Decisions in Chain")
    ax.set_ylabel("Number of Chains")
    ax.set_title("Agent Agreement Distribution")

    # 1.5 Approval Type Distribution (Final)
    ax = axes[1, 1]
    if len(df_final) > 0 and df_final["approval_type"].notna().any():
        type_counts = df_final["approval_type"].value_counts()
        colors_type = {
            "STANDARD_TERMS": "#27ae60",
            "SUBOPTIMAL_TERMS": "#f39c12",
            "MANUAL_REVIEW": "#3498db",
            "DENIAL": "#e74c3c",
        }
        ax.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.1f%%",
            colors=[
                colors_type.get(str(t).upper(), "#95a5a6") for t in type_counts.index
            ],
        )
        ax.set_title("Final Approval Type Distribution")

    # 1.6 Confidence Evolution
    ax = axes[1, 2]
    if df_valid["confidence_probability"].notna().any():
        conf_by_pos = df_valid.groupby("agent_position")["confidence_probability"].agg(
            ["mean", "std"]
        )
        ax.errorbar(
            conf_by_pos.index,
            conf_by_pos["mean"],
            yerr=conf_by_pos["std"].fillna(0),
            marker="D",
            linewidth=2,
            markersize=8,
            capsize=5,
            color="#27ae60",
        )
        ax.set_xlabel("Agent Position")
        ax.set_ylabel("Confidence Probability")
        ax.set_title("Confidence Evolution")

    plt.tight_layout()
    plt.savefig(seq_dir / "chain_overview.png", dpi=300, bbox_inches="tight")
    plt.savefig(seq_dir / "chain_overview.pdf", bbox_inches="tight")
    plt.close()

    # Figure 2: Per-Agent Analysis
    agents = df_valid["agent"].unique()
    n_agents = len(agents)

    if n_agents > 0:
        fig, axes = plt.subplots(2, n_agents, figsize=(4 * n_agents, 8))
        if n_agents == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle(
            f"Per-Agent Analysis: {agents_display}", fontsize=12, fontweight="bold"
        )

        for i, agent in enumerate(agents):
            df_agent = df_valid[df_valid["agent"] == agent]
            df_agent_approved = df_agent[df_agent["approval_norm"] == "approve"]

            ax = axes[0, i]
            if df_agent["ethnicity_signal"].notna().any():
                eth_rates = df_agent.groupby("ethnicity_signal")["approved"].mean()
                ax.bar(range(len(eth_rates)), eth_rates.values, color="#3498db")
                ax.set_xticks(range(len(eth_rates)))
                ax.set_xticklabels(
                    [e.replace("_Signal", "") for e in eth_rates.index],
                    rotation=45,
                    ha="right",
                )
                ax.set_ylim(0, 1)
                ax.set_ylabel("Approval Rate" if i == 0 else "")
                ax.set_title(f"{agent}\n(Position {i+1})")

            ax = axes[1, i]
            if (
                len(df_agent_approved) > 0
                and df_agent_approved["interest_rate"].notna().any()
            ):
                valid_ir = df_agent_approved["interest_rate"].dropna()
                valid_ir = valid_ir[(valid_ir > 0) & (valid_ir < 50)]
                if len(valid_ir) > 0:
                    sns.histplot(valid_ir, ax=ax, bins=15, color="#e74c3c")
                    ax.axvline(valid_ir.mean(), color="black", linestyle="--")
            ax.set_xlabel("Interest Rate (%)" if i == n_agents // 2 else "")
            ax.set_ylabel("Count" if i == 0 else "")

        plt.tight_layout()
        plt.savefig(seq_dir / "per_agent_analysis.png", dpi=300, bbox_inches="tight")
        plt.savefig(seq_dir / "per_agent_analysis.pdf", bbox_inches="tight")
        plt.close()

    # Figure 3: Decision Flow Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    decision_by_position = df_valid.pivot_table(
        values="approved",
        index="agent_position",
        columns="ethnicity_signal",
        aggfunc="mean",
    )
    if not decision_by_position.empty:
        decision_by_position.columns = [
            c.replace("_Signal", "") for c in decision_by_position.columns
        ]
        sns.heatmap(
            decision_by_position,
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Approval Rate"},
        )
        ax.set_xlabel("Ethnicity")
        ax.set_ylabel("Agent Position")
        ax.set_title("Approval Rate by Position and Ethnicity")

    plt.tight_layout()
    plt.savefig(seq_dir / "decision_flow.png", dpi=300, bbox_inches="tight")
    plt.savefig(seq_dir / "decision_flow.pdf", bbox_inches="tight")
    plt.close()

    # Figure 4: First vs Final Transition
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"First Agent → Final Decision Analysis\n{agents_display}",
        fontsize=12,
        fontweight="bold",
    )

    df_first = df_valid[df_valid["agent_position"] == 1][
        ["chain_id", "approval_norm"]
    ].copy()
    df_first.columns = ["chain_id", "first_decision"]
    df_transitions = df_final[["chain_id", "approval_norm"]].merge(
        df_first, on="chain_id", how="inner"
    )
    df_transitions.columns = ["chain_id", "final_decision", "first_decision"]

    if len(df_transitions) > 0:
        ax = axes[0]
        transition_counts = pd.crosstab(
            df_transitions["first_decision"],
            df_transitions["final_decision"],
            margins=False,
        )
        for col in ["approve", "deny"]:
            if col not in transition_counts.columns:
                transition_counts[col] = 0
        for idx in ["approve", "deny"]:
            if idx not in transition_counts.index:
                transition_counts.loc[idx] = 0
        transition_counts = transition_counts.reindex(
            index=["approve", "deny"], columns=["approve", "deny"], fill_value=0
        )

        total = transition_counts.values.sum()
        annot_text = np.array(
            [
                [
                    f"{transition_counts.iloc[i, j]:,}\n({transition_counts.iloc[i, j]/total:.1%})"
                    for j in range(2)
                ]
                for i in range(2)
            ]
        )

        sns.heatmap(
            transition_counts,
            annot=annot_text,
            fmt="",
            cmap="RdYlGn",
            ax=ax,
            cbar_kws={"label": "Count"},
            xticklabels=["Approve", "Deny"],
            yticklabels=["Approve", "Deny"],
        )
        ax.set_xlabel("Final Decision", fontsize=11)
        ax.set_ylabel("First Agent Decision", fontsize=11)
        ax.set_title("Decision Transition Counts", fontsize=11, fontweight="bold")

        ax = axes[1]
        transition_probs = transition_counts.div(
            transition_counts.sum(axis=1), axis=0
        ).fillna(0)
        annot_probs = np.array(
            [[f"{transition_probs.iloc[i, j]:.1%}" for j in range(2)] for i in range(2)]
        )
        sns.heatmap(
            transition_probs,
            annot=annot_probs,
            fmt="",
            cmap="RdYlGn",
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Probability"},
            xticklabels=["Approve", "Deny"],
            yticklabels=["Approve", "Deny"],
        )
        ax.set_xlabel("Final Decision", fontsize=11)
        ax.set_ylabel("First Agent Decision", fontsize=11)
        ax.set_title(
            "P(Final Decision | First Agent Decision)", fontsize=11, fontweight="bold"
        )

        first_approve = df_transitions["first_decision"] == "approve"
        final_approve = df_transitions["final_decision"] == "approve"
        agreement_rate = (
            (first_approve & final_approve) | (~first_approve & ~final_approve)
        ).mean()
        summary_text = f"Agreement Rate: {agreement_rate:.1%}"
        fig.text(
            0.5,
            0.02,
            summary_text,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(seq_dir / "first_vs_final_transition.png", dpi=300, bbox_inches="tight")
    plt.savefig(seq_dir / "first_vs_final_transition.pdf", bbox_inches="tight")
    plt.close()

    # Figure 5: First vs Final by Ethnicity
    if len(df_transitions) > 0:
        # Create a mapping from chain_id to ethnicity (use first occurrence if duplicates)
        ethnicity_map = df_final.drop_duplicates(subset=["chain_id"]).set_index(
            "chain_id"
        )["ethnicity_signal"]
        ethnicities = df_transitions["chain_id"].map(ethnicity_map)
        df_transitions["ethnicity"] = ethnicities
        unique_ethnicities = df_transitions["ethnicity"].dropna().unique()
        n_eth = len(unique_ethnicities)

        if n_eth > 0:
            n_cols = min(4, n_eth)
            n_rows = (n_eth + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            if n_eth == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle(
                f"First Agent → Final Decision by Ethnicity\n{agents_display}",
                fontsize=12,
                fontweight="bold",
            )

            for idx, ethnicity in enumerate(sorted(unique_ethnicities)):
                row, col = idx // n_cols, idx % n_cols
                ax = axes[row, col]

                df_eth = df_transitions[df_transitions["ethnicity"] == ethnicity]
                eth_counts = pd.crosstab(
                    df_eth["first_decision"], df_eth["final_decision"]
                )

                for c in ["approve", "deny"]:
                    if c not in eth_counts.columns:
                        eth_counts[c] = 0
                for i in ["approve", "deny"]:
                    if i not in eth_counts.index:
                        eth_counts.loc[i] = 0
                eth_counts = eth_counts.reindex(
                    index=["approve", "deny"], columns=["approve", "deny"], fill_value=0
                )

                eth_total = eth_counts.values.sum()
                eth_annot = np.array(
                    [
                        [
                            (
                                f"{eth_counts.iloc[i, j]:,}\n({eth_counts.iloc[i, j]/eth_total:.1%})"
                                if eth_total > 0
                                else "0"
                            )
                            for j in range(2)
                        ]
                        for i in range(2)
                    ]
                )

                sns.heatmap(
                    eth_counts,
                    annot=eth_annot,
                    fmt="",
                    cmap="RdYlGn",
                    ax=ax,
                    cbar=False,
                    xticklabels=["Approve", "Deny"],
                    yticklabels=["Approve", "Deny"],
                )
                eth_display = (
                    ethnicity.replace("_Signal", "") if ethnicity else "Unknown"
                )
                ax.set_title(f"{eth_display}\n(n={eth_total:,})", fontsize=10)
                ax.set_xlabel("Final" if row == n_rows - 1 else "")
                ax.set_ylabel("First" if col == 0 else "")

            for idx in range(n_eth, n_rows * n_cols):
                row, col = idx // n_cols, idx % n_cols
                axes[row, col].axis("off")

            plt.tight_layout()
            plt.savefig(
                seq_dir / "first_vs_final_by_ethnicity.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                seq_dir / "first_vs_final_by_ethnicity.pdf", bbox_inches="tight"
            )
            plt.close()

    # Save summary
    summary = {
        "ordering": ordering,
        "agents_display": agents_display,
        "total_chains": len(df_final),
        "final_approval_rate": (
            float(df_final["approved"].mean()) if len(df_final) > 0 else None
        ),
    }
    with open(seq_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved sequential analysis for {ordering}")


def create_parallel_analysis(df: pd.DataFrame, output_dir: Path):
    """Create analysis for parallel mode."""
    df_par = df[df["experiment_type"] == "parallel"].copy()

    if df_par.empty:
        print("  No parallel data found")
        return

    par_dir = output_dir / "parallel"
    par_dir.mkdir(parents=True, exist_ok=True)

    df_par["approval_norm"] = df_par["approval_decision"].apply(normalize_decision)
    df_valid = df_par[df_par["approval_norm"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_norm"] == "approve").astype(int)

    if len(df_valid) == 0:
        print("  No valid approve/deny decisions in parallel data")
        return

    # Figure 1: Agent Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Parallel Mode Analysis\n(n={len(df_valid):,} records)",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    agent_rates = df_valid.groupby("agent")["approved"].mean().sort_values()
    colors = [
        "#27ae60" if a == "Business Decision" else "#3498db" for a in agent_rates.index
    ]
    bars = ax.barh(agent_rates.index, agent_rates.values, color=colors)
    ax.set_xlabel("Approval Rate")
    ax.set_title("Approval Rate by Agent")
    ax.set_xlim(0, 1)
    for bar, rate in zip(bars, agent_rates.values):
        ax.text(
            rate + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1%}",
            va="center",
            fontsize=9,
        )

    ax = axes[1]
    if df_valid["ethnicity_signal"].notna().any():
        eth_rates = (
            df_valid.groupby("ethnicity_signal")["approved"].mean().sort_values()
        )
        colors = [
            (
                "#e74c3c"
                if r < eth_rates.mean() - 0.05
                else "#27ae60" if r > eth_rates.mean() + 0.05 else "#3498db"
            )
            for r in eth_rates
        ]
        bars = ax.barh(eth_rates.index, eth_rates.values, color=colors)
        ax.set_xlabel("Approval Rate")
        ax.set_title("Approval Rate by Ethnicity")
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(par_dir / "parallel_overview.png", dpi=300, bbox_inches="tight")
    plt.savefig(par_dir / "parallel_overview.pdf", bbox_inches="tight")
    plt.close()

    # Figure 2: Agent x Ethnicity Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot = df_valid.pivot_table(
        values="approved", index="agent", columns="ethnicity_signal", aggfunc="mean"
    )
    if not pivot.empty:
        pivot.columns = [c.replace("_Signal", "") for c in pivot.columns]
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Approval Rate"},
        )
        ax.set_title("Approval Rate: Agent × Ethnicity")

    plt.tight_layout()
    plt.savefig(par_dir / "agent_ethnicity_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(par_dir / "agent_ethnicity_heatmap.pdf", bbox_inches="tight")
    plt.close()

    summary = {
        "total_records": len(df_valid),
        "agents": list(df_valid["agent"].unique()),
        "overall_approval_rate": float(df_valid["approved"].mean()),
    }
    with open(par_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved parallel analysis ({len(df_valid)} records)")


def create_overall_comparison(df: pd.DataFrame, output_dir: Path):
    """Create overall comparison plots."""
    overall_dir = output_dir / "overall"
    overall_dir.mkdir(parents=True, exist_ok=True)

    df["approval_norm"] = df["approval_decision"].apply(normalize_decision)
    df_valid = df[df["approval_norm"].isin(["approve", "deny"])].copy()
    df_valid["approved"] = (df_valid["approval_norm"] == "approve").astype(int)

    if len(df_valid) == 0:
        print("  No valid data for overall analysis")
        return

    # Figure 1: Approval Rate by Experiment Type
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Overall Analysis\n(n={len(df_valid):,} records)",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    type_rates = df_valid.groupby("experiment_type")["approved"].mean().sort_values()
    ax.bar(type_rates.index, type_rates.values, color="#3498db")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rate by Experiment Type")
    ax.set_ylim(0, 1)

    ax = axes[1]
    if df_valid["ethnicity_signal"].notna().any():
        eth_rates = (
            df_valid.groupby("ethnicity_signal")["approved"].mean().sort_values()
        )
        colors = [
            (
                "#e74c3c"
                if r < eth_rates.mean() - 0.05
                else "#27ae60" if r > eth_rates.mean() + 0.05 else "#3498db"
            )
            for r in eth_rates
        ]
        bars = ax.barh(eth_rates.index, eth_rates.values, color=colors)
        ax.set_xlabel("Approval Rate")
        ax.set_title("Approval Rate by Ethnicity (All Modes)")
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(overall_dir / "overall_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(overall_dir / "overall_comparison.pdf", bbox_inches="tight")
    plt.close()

    # Figure 2: Ethnicity x Mode Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    if df_valid["ethnicity_signal"].notna().any():
        pivot = df_valid.pivot_table(
            values="approved",
            index="experiment_type",
            columns="ethnicity_signal",
            aggfunc="mean",
        )
        if not pivot.empty:
            pivot.columns = [c.replace("_Signal", "") for c in pivot.columns]
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1%",
                cmap="RdYlGn",
                ax=ax,
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Approval Rate"},
            )
            ax.set_title("Approval Rate: Experiment Type × Ethnicity")

    plt.tight_layout()
    plt.savefig(
        overall_dir / "mode_ethnicity_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(overall_dir / "mode_ethnicity_heatmap.pdf", bbox_inches="tight")
    plt.close()

    summary = {
        "total_records": len(df_valid),
        "experiment_types": df_valid["experiment_type"].value_counts().to_dict(),
        "overall_approval_rate": float(df_valid["approved"].mean()),
    }
    with open(overall_dir / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved overall comparison")


def run_analysis_for_model(base_dir: str, model_name: str, output_base: str):
    """Run full analysis for a single model."""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name} from {base_dir}")
    print(f"{'='*60}")

    # Create output directory
    output_dir = Path(output_base) / model_name.replace(":", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    loader = SmallModelDataLoader(base_dir, model_name)
    df = loader.get_combined_df()

    if df.empty:
        print(f"No data loaded for {model_name}")
        return

    # Create overall analysis
    print("\n[1] Creating overall comparison...")
    create_overall_comparison(df, output_dir)

    # Create baseline analysis for each agent
    print("\n[2] Creating baseline agent analysis...")
    baseline_agents = df[df["experiment_type"] == "baseline"]["agent"].unique()
    for agent in baseline_agents:
        create_baseline_analysis(df, agent, output_dir)

    # Create sequential analysis for each ordering
    if "agent_order" in df.columns:
        print("\n[3] Creating sequential ordering analysis...")
        orderings = (
            df[df["experiment_type"] == "sequential"]["agent_order"].dropna().unique()
        )
        if len(orderings) > 0:
            for ordering in orderings:
                create_sequential_analysis(df, ordering, output_dir)
        else:
            print("  No sequential orderings found")
    else:
        print("\n[3] No sequential data available")

    # Create parallel analysis
    print("\n[4] Creating parallel analysis...")
    create_parallel_analysis(df, output_dir)

    # Count files
    png_count = len(list(output_dir.rglob("*.png")))
    pdf_count = len(list(output_dir.rglob("*.pdf")))
    json_count = len(list(output_dir.rglob("*.json")))

    print(f"\n{'='*60}")
    print(f"Completed {model_name}")
    print(f"  Output directory: {output_dir}")
    print(f"  Files: {png_count} PNG, {pdf_count} PDF, {json_count} JSON")
    print(f"{'='*60}")


def main():
    """Run all analysis."""
    print("=" * 70)
    print("FairWatch Small Model Analysis")
    print("=" * 70)

    # Analysis for outputs_simple
    print("\n" + "=" * 70)
    print("OUTPUTS_SIMPLE ANALYSIS")
    print("=" * 70)

    for model in ["llama3.2", "mistral:latest"]:
        run_analysis_for_model(
            "outputs_simple", model, "plotting/outputs/outputs_simple"
        )

    # Analysis for outputs_complex
    print("\n" + "=" * 70)
    print("OUTPUTS_COMPLEX ANALYSIS")
    print("=" * 70)

    for model in ["llama3.2", "mistral:latest"]:
        run_analysis_for_model(
            "outputs_complex", model, "plotting/outputs/outputs_complex"
        )

    # Final summary
    print("\n" + "=" * 70)
    print("ALL ANALYSIS COMPLETE")
    print("=" * 70)

    for folder in ["outputs_simple", "outputs_complex"]:
        output_path = Path(f"plotting/outputs/{folder}")
        if output_path.exists():
            total_png = len(list(output_path.rglob("*.png")))
            total_pdf = len(list(output_path.rglob("*.pdf")))
            print(f"\n{folder}:")
            for model_dir in output_path.iterdir():
                if model_dir.is_dir():
                    model_png = len(list(model_dir.rglob("*.png")))
                    print(f"  {model_dir.name}: {model_png} PNG files")
            print(f"  Total: {total_png} PNG, {total_pdf} PDF")


if __name__ == "__main__":
    main()
