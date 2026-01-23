"""
Master Plotting Script for FairWatch
Runs all plotting modules and generates comprehensive visualization outputs.
"""

import os
import sys
from pathlib import Path

# Add plotting directory to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Run all plotting scripts."""
    print("=" * 60)
    print("FairWatch Multi-Agent System Analysis - Plot Generation")
    print("=" * 60)

    # Import after path setup
    from data_loader import load_data

    # Load data once
    print("\n[1/7] Loading data...")
    df, raw_data = load_data("70bresults")
    print(f"Loaded {len(df)} total records")
    print(f"Experiment types: {df['experiment_type'].value_counts().to_dict()}")
    print(f"Agents: {df['agent'].nunique()} unique agents")

    # Create main output directory
    output_base = Path("plotting/outputs")
    output_base.mkdir(parents=True, exist_ok=True)

    # Run each plotting module
    modules = [
        ("plot_demographic_bias", "Demographic Bias Analysis"),
        ("plot_agent_comparison", "Agent Comparison"),
        ("plot_sequential_analysis", "Sequential Chain Analysis"),
        ("plot_mode_comparison", "Mode Comparison"),
        ("plot_publication_figures", "Publication Figures"),
        ("plot_supplementary", "Supplementary Analysis"),
    ]

    for i, (module_name, description) in enumerate(modules, 2):
        print(f"\n[{i}/7] Generating {description} plots...")
        try:
            module = __import__(module_name)
            module.main()
        except Exception as e:
            print(f"Error in {module_name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Plot Generation Complete!")
    print("=" * 60)

    # Count generated files
    total_plots = 0
    for subdir in output_base.iterdir():
        if subdir.is_dir():
            plots = list(subdir.glob("*.png")) + list(subdir.glob("*.pdf"))
            total_plots += len(plots)
            print(f"  {subdir.name}: {len(plots)} files")

    print(f"\nTotal: {total_plots} plot files generated")
    print(f"Output directory: {output_base.absolute()}")


if __name__ == "__main__":
    main()
