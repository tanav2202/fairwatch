import json
from pathlib import Path

base_dir = Path('plotting/outputs/outputs_simple/mistral_latest/sequential')

zero_count = 0
valid_count = 0
rates = []

for ordering_dir in sorted(base_dir.iterdir()):
    if ordering_dir.is_dir():
        summary_file = ordering_dir / 'summary_stats.json'
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
                rate = data['final_approval_rate']
                if rate == 0.0:
                    zero_count += 1
                    print(f'ZERO: {ordering_dir.name}')
                else:
                    valid_count += 1
                    rates.append(rate)

print(f'\nTotal: {zero_count + valid_count}')
print(f'Zero approval: {zero_count}')
print(f'Valid: {valid_count}')
if rates:
    print(f'Filtered range: {min(rates):.1%} - {max(rates):.1%} (Δ = {max(rates)-min(rates):.1%})')
