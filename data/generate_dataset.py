
import pandas as pd
import hashlib
import logging
import random
import argparse
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from jinja2 import Template

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

class ApplicantProfile(BaseModel):
    name: str
    ethnicity_signal: str
    credit_score: int
    visa_status: str
    income: int
    age: int
    loan_amount: int
    country: str = "USA"

def load_input_lists(input_file: str) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("input_lists", input_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        'names': module.NAMES, 'credit_scores': module.CREDIT_SCORES,
        'visa_status': module.VISA_STATUS, 'income_mapping': module.INCOME_MAPPING,
        'age': module.AGE, 'loan_multipliers': module.LOAN_MULTIPLIERS,
    }

def generate_combinations(inputs: dict) -> List[Dict[str, Any]]:
    combinations = []
    all_names = []
    for signal, names in inputs['names'].items():
        for name in names: all_names.append({'name': name, 'ethnicity_signal': signal})

    for name_data in all_names:
        for score in inputs['credit_scores']:
            for visa in inputs['visa_status']:
                for inc_level, inc_val in inputs['income_mapping'].items():
                    for age in inputs['age']:
                        for mult in inputs['loan_multipliers']:
                            loan_amt = int(round((inc_val * mult) / 1000) * 1000)
                            raw = {
                                "name": name_data['name'], "ethnicity_signal": name_data['ethnicity_signal'],
                                "credit_score": score, "visa_status": visa, "income": inc_val,
                                "age": age, "loan_amount": loan_amt
                            }
                            try: combinations.append(ApplicantProfile(**raw).model_dump())
                            except: continue
    return combinations

def create_dataset(input_file, template_path, output_path, sample_rate=1.0):
    inputs = load_input_lists(input_file)
    combos = generate_combinations(inputs)
    
    if sample_rate < 1.0:
        random.seed(42)
        combos = random.sample(combos, max(1, int(len(combos) * sample_rate)))

    with open(template_path, 'r') as f: jinja_template = Template(f.read())

    dataset = []
    for c in combos:
        try:
            c['dti_ratio'] = 0.35; c['is_young'] = c['age'] < 30; c['is_high_income'] = c['income'] > 100000
            prompt_text = jinja_template.render(**c)
            row = c.copy()
            row['prompt_id'] = hashlib.md5(prompt_text.encode()).hexdigest()[:12]
            row['prompt'] = prompt_text
            dataset.append(row)
        except Exception as e: LOG.error(f"Templating error: {e}")

    df = pd.DataFrame(dataset)
    df.to_csv(output_path, index=False)
    LOG.info(f"Saved {len(df)} prompts to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/input_lists.py")
    parser.add_argument("--template", default="prompts/complex_input.j2")
    parser.add_argument("--output", default="data/prompts_complex.csv")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    args = parser.parse_args()
    create_dataset(args.input, args.template, args.output, args.sample_rate)
