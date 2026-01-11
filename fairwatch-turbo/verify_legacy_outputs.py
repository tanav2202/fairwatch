
import json
import argparse
import glob
import os
from typing import Dict, Any, List

def check_file(filepath: str) -> Dict[str, Any]:
    stats = {
        "filename": os.path.basename(filepath),
        "total_chains": 0,
        "valid_chains": 0,
        "null_confidences": 0,
        "null_rates": 0,
        "parse_errors": 0, # Fallback denials
        "missing_business_decision": 0,
        "schema_compliant": True,
        "status": "PASS"
    }
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        chains = data.get('results', [])
        stats['total_chains'] = len(chains)
        
        for chain in chains:
            # 1. Check Business Decision
            biz = chain.get('business_decision')
            if not biz:
                stats['missing_business_decision'] += 1
                stats['schema_compliant'] = False
            else:
                # Check for Fallback Parsing Error
                # In convert script, fallback has reasoning: "Parsing Failure - Manual Review Required"
                reasoning = biz.get('reasoning', {})
                if isinstance(reasoning, dict):
                    rat = reasoning.get('synthesis_rationale', "")
                    if "Parsing Failure" in rat:
                        stats['parse_errors'] += 1
                
            # 2. Check Nulls in Business Decision
            if biz:
                if biz.get('confidence_probability') is None:
                    stats['null_confidences'] += 1
                if biz.get('interest_rate') is None:
                    stats['null_rates'] += 1
                    
            # 3. Check Conversation History
            if 'conversation_history' not in chain:
                 stats['schema_compliant'] = False
                 
        stats['valid_chains'] = stats['total_chains'] - stats['parse_errors'] - stats['missing_business_decision']
        
        if stats['parse_errors'] > 0:
            stats['status'] = "WARNING (Parsing Errors)"
        if stats['missing_business_decision'] > 0:
             stats['status'] = "FAIL (Missing Data)"
        if not stats['schema_compliant']:
             stats['status'] = "FAIL (Schema)"
             
    except Exception as e:
        stats['status'] = f"CRITICAL FILE ERROR: {str(e)}"
        
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory to scan")
    parser.add_argument("--pattern", type=str, default="*_legacy.json", help="File pattern")
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.dir, args.pattern))
    files.sort()
    
    if not files:
        print(f"No files found in {args.dir} matching {args.pattern}")
        return

    print(f"{'FILENAME':<60} | {'TOTAL':<8} | {'ERRORS':<8} | {'NULLS':<8} | {'STATUS'}")
    print("-" * 110)
    
    for f in files:
        s = check_file(f)
        print(f"{s['filename']:<60} | {s['total_chains']:<8} | {s['parse_errors']:<8} | {s['null_confidences']+s['null_rates']:<8} | {s['status']}")

if __name__ == "__main__":
    main()
