"""
Cross-Product Dataset Generator - Final Version
No locations - country is always USA
Loan amounts calculated as multipliers of income (1x, 5x)
"""

import pandas as pd
import hashlib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def load_input_lists(input_file: str) -> dict:
    """Load input lists from Python file"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("input_lists", input_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return {
        'names': module.NAMES,
        'credit_scores': module.CREDIT_SCORES,
        'visa_status': module.VISA_STATUS,
        'income_levels': module.INCOME,
        'income_mapping': module.INCOME_MAPPING,
        'age': module.AGE,
        'loan_multipliers': module.LOAN_MULTIPLIERS,
    }


def load_template(template_path: str) -> str:
    """Load prompt template"""
    with open(template_path, 'r') as f:
        return f.read()


def calculate_loan_amount(income: int, multiplier: float) -> int:
    """
    Calculate loan amount as multiplier of income
    
    Args:
        income: Annual income in dollars
        multiplier: 1.0 or 5.0
    
    Returns:
        Loan amount rounded to nearest $1000
    """
    loan = income * multiplier
    # Round to nearest $1000
    return int(round(loan / 1000) * 1000)


def generate_cross_product(input_lists: dict, sample_rate: float = 1.0) -> list:
    """
    Generate cross-product of all input dimensions
    
    Args:
        input_lists: Dictionary of input lists
        sample_rate: Fraction of combinations to generate (1.0 = all)
    
    Returns:
        List of dictionaries with all combinations
    """
    LOG.info("Generating cross-product combinations...")
    
    combinations = []
    
    for name in input_lists['names']:
        for credit_score in input_lists['credit_scores']:
            for visa_status in input_lists['visa_status']:
                for income_level in input_lists['income_levels']:
                    # Map income level to actual dollar amount
                    income = input_lists['income_mapping'][income_level]
                    
                    for age in input_lists['age']:
                        for loan_multiplier in input_lists['loan_multipliers']:
                            # Calculate loan amount
                            loan_amount = calculate_loan_amount(income, loan_multiplier)
                            
                            combination = {
                                'name': name,
                                'credit_score': credit_score,
                                'visa_status': visa_status,
                                'income_level': income_level,
                                'income': income,
                                'age': age,
                                'loan_multiplier': loan_multiplier,
                                'loan_amount': loan_amount,
                                'country': 'USA',  # Always USA
                            }
                            
                            combinations.append(combination)
    
    LOG.info(f"Generated {len(combinations):,} total combinations")
    
    # Sample if requested
    if sample_rate < 1.0:
        import random
        random.seed(42)
        sample_size = int(len(combinations) * sample_rate)
        combinations = random.sample(combinations, sample_size)
        LOG.info(f"Sampled {len(combinations):,} combinations ({sample_rate*100:.1f}%)")
    
    return combinations


def generate_prompt(template: str, data: dict) -> str:
    """Fill template with data"""
    try:
        return template.format(**data)
    except KeyError as e:
        LOG.error(f"Missing key in template: {e}")
        LOG.error(f"Available keys: {list(data.keys())}")
        raise


def hash_prompt(prompt: str) -> str:
    """Generate hash of prompt for deduplication"""
    return hashlib.md5(prompt.encode()).hexdigest()[:12]


def create_dataset(
    input_file: str,
    template_simple: str,
    template_complex: str,
    output_simple: str,
    output_complex: str,
    sample_rate: float = 1.0
):
    """
    Create complete dataset with prompts and metadata
    
    Args:
        input_file: Path to input_lists.py
        template_simple: Path to simple template
        template_complex: Path to complex template
        output_simple: Output CSV for simple prompts
        output_complex: Output CSV for complex prompts
        sample_rate: Fraction of combinations to generate
    """
    LOG.info("=" * 80)
    LOG.info("CROSS-PRODUCT DATASET GENERATION - FINAL")
    LOG.info("=" * 80)
    
    # Load inputs
    input_lists = load_input_lists(input_file)
    LOG.info(f"\nInput dimensions:")
    LOG.info(f"  Names: {len(input_lists['names'])}")
    LOG.info(f"  Credit scores: {len(input_lists['credit_scores'])}")
    LOG.info(f"  Visa statuses: {len(input_lists['visa_status'])}")
    LOG.info(f"  Income levels: {len(input_lists['income_levels'])}")
    LOG.info(f"  Ages: {len(input_lists['age'])}")
    LOG.info(f"  Loan multipliers: {len(input_lists['loan_multipliers'])}")
    
    total_theoretical = (
        len(input_lists['names']) *
        len(input_lists['credit_scores']) *
        len(input_lists['visa_status']) *
        len(input_lists['income_levels']) *
        len(input_lists['age']) *
        len(input_lists['loan_multipliers'])
    )
    LOG.info(f"\nTheoretical max combinations: {total_theoretical:,}")
    
    template_s = load_template(template_simple)
    template_c = load_template(template_complex)
    LOG.info(f"\nLoaded templates:")
    LOG.info(f"  Simple: {len(template_s)} chars")
    LOG.info(f"  Complex: {len(template_c)} chars")
    
    # Generate cross-product
    combinations = generate_cross_product(input_lists, sample_rate)
    
    # Generate prompts for both templates
    simple_data = []
    complex_data = []
    
    LOG.info(f"\nGenerating prompts...")
    for i, combo in enumerate(combinations):
        if (i + 1) % 1000 == 0:
            LOG.info(f"  Processed {i + 1:,}/{len(combinations):,} combinations...")
        
        # Generate simple prompt
        try:
            prompt_s = generate_prompt(template_s, combo)
            prompt_s_hash = hash_prompt(prompt_s)
            
            simple_data.append({
                'prompt_id': f'simple_{prompt_s_hash}',
                'template': 'simple',
                'prompt': prompt_s,
                **combo
            })
        except Exception as e:
            LOG.warning(f"Skipping simple prompt {i}: {e}")
        
        # Generate complex prompt
        try:
            prompt_c = generate_prompt(template_c, combo)
            prompt_c_hash = hash_prompt(prompt_c)
            
            complex_data.append({
                'prompt_id': f'complex_{prompt_c_hash}',
                'template': 'complex',
                'prompt': prompt_c,
                **combo
            })
        except Exception as e:
            LOG.warning(f"Skipping complex prompt {i}: {e}")
    
    # Convert to DataFrames
    df_simple = pd.DataFrame(simple_data)
    df_complex = pd.DataFrame(complex_data)
    
    # Deduplicate by prompt
    LOG.info("\nDeduplicating...")
    simple_before = len(df_simple)
    df_simple = df_simple.drop_duplicates(subset=['prompt'], keep='first')
    LOG.info(f"  Simple: {simple_before:,} → {len(df_simple):,} (removed {simple_before - len(df_simple):,} duplicates)")
    
    complex_before = len(df_complex)
    df_complex = df_complex.drop_duplicates(subset=['prompt'], keep='first')
    LOG.info(f"  Complex: {complex_before:,} → {len(df_complex):,} (removed {complex_before - len(df_complex):,} duplicates)")
    
    # Save to CSV
    LOG.info(f"\nSaving to CSV...")
    df_simple.to_csv(output_simple, index=False)
    LOG.info(f"  Simple prompts: {output_simple} ({len(df_simple):,} rows)")
    
    df_complex.to_csv(output_complex, index=False)
    LOG.info(f"  Complex prompts: {output_complex} ({len(df_complex):,} rows)")
    
    # Summary statistics
    LOG.info("\n" + "=" * 80)
    LOG.info("DATASET SUMMARY")
    LOG.info("=" * 80)
    LOG.info(f"Total unique prompts: {len(df_simple) + len(df_complex):,}")
    LOG.info(f"  Simple template: {len(df_simple):,}")
    LOG.info(f"  Complex template: {len(df_complex):,}")
    
    LOG.info("\nVariation coverage:")
    LOG.info(f"  Unique names: {df_simple['name'].nunique()}")
    LOG.info(f"  Unique credit scores: {sorted(df_simple['credit_score'].unique())}")
    LOG.info(f"  Unique visa statuses: {df_simple['visa_status'].nunique()}")
    LOG.info(f"  Unique income levels: {sorted(df_simple['income_level'].unique())}")
    LOG.info(f"  Unique ages: {sorted(df_simple['age'].unique())}")
    LOG.info(f"  Unique loan multipliers: {sorted(df_simple['loan_multiplier'].unique())}")
    
    # Loan amount ranges
    LOG.info(f"\nLoan amount ranges:")
    LOG.info(f"  Min: ${df_simple['loan_amount'].min():,}")
    LOG.info(f"  Max: ${df_simple['loan_amount'].max():,}")
    LOG.info(f"  Mean: ${df_simple['loan_amount'].mean():,.0f}")
    
    # Name distribution
    LOG.info("\nName distribution:")
    name_counts = df_simple['name'].value_counts()
    for name in sorted(name_counts.index):
        count = name_counts[name]
        pct = (count / len(df_simple)) * 100
        LOG.info(f"  {name}: {count:,} ({pct:.1f}%)")
    
    LOG.info("\n✅ Dataset generation complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate cross-product dataset for personal loans")
    parser.add_argument("--input", default="data/input_lists.py", help="Input lists file")
    parser.add_argument("--template-simple", default="data/template_simple.txt", help="Simple template")
    parser.add_argument("--template-complex", default="data/template_complex.txt", help="Complex template")
    parser.add_argument("--output-simple", default="data/prompts_simple.csv", help="Output CSV for simple prompts")
    parser.add_argument("--output-complex", default="data/prompts_complex.csv", help="Output CSV for complex prompts")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Fraction of combinations to sample (1.0 = 100%%)")
    
    args = parser.parse_args()
    
    create_dataset(
        input_file=args.input,
        template_simple=args.template_simple,
        template_complex=args.template_complex,
        output_simple=args.output_simple,
        output_complex=args.output_complex,
        sample_rate=args.sample_rate
    )