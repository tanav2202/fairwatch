import json
import os
import glob
import sys
import argparse

def verify_strict(filepath):
    # just checking if the file is valid and follows our strict schema
    print(f"Verifying: {os.path.basename(filepath)}")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [FATAL] Invalid JSON: {e}")
        return False

    if "results" not in data or not isinstance(data["results"], list):
        print("  [FATAL] Missing 'results' list at root.")
        return False

    results = data["results"]
    if not results:
        print("  [FATAL] 'results' list is empty.")
        return False

    # keys we expect in every record
    required_keys = [
        "mode", "initial_prompt", "conversation_history", 
        "all_agent_outputs", "final_output", "business_decision", 
        "prompt_id", "ordering"
    ]
    
    # keys we expect inside business_decision
    required_biz_keys = [
        "approval_decision", "interest_rate", "confidence_probability", "reasoning"
    ]

    total = len(results)
    failures = 0

    for idx, r in enumerate(results):
        # check top-level keys exist
        missing = [k for k in required_keys if k not in r]
        if missing:
            print(f"  [FAIL] Record {idx}: Missing keys {missing}")
            failures += 1
            continue

        # check business decision specifically
        biz = r.get("business_decision")
        if not biz or not isinstance(biz, dict):
             print(f"  [FAIL] Record {idx}: 'business_decision' is missing or null.")
             failures += 1
             continue

        missing_biz = [k for k in required_biz_keys if k not in biz]
        if missing_biz:
            print(f"  [FAIL] Record {idx}: 'business_decision' missing keys {missing_biz}")
            failures += 1
            continue

        # check values: confidence must be > 0
        conf = biz.get("confidence_probability")
        if conf is None or not isinstance(conf, (int, float)) or conf <= 0:
             print(f"  [FAIL] Record {idx}: Invalid confidence_probability: {conf}")
             failures += 1
             continue
        
        # interest rate shouldn't be null
        rate = biz.get("interest_rate")
        if rate is None:
            print(f"  [FAIL] Record {idx}: Interest rate is None (null).")
            failures += 1
            continue

        # look for system errors or parsing failures in the text
        reasoning = str(biz.get("reasoning", ""))
        error_terms = ["System Error", "Parsing Failure", "Manual Review Required"]
        found_errs = [t for t in error_terms if t in reasoning]
        if found_errs:
             print(f"  [FAIL] Record {idx}: Found error terms {found_errs} in reasoning.")
             failures += 1
             continue

    if failures == 0:
        print(f"  [PASS] Checked {total} records. All strictly valid.")
        return True
    else:
        print(f"  [FAIL] Found {failures} invalid records out of {total}.")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Specific file to verify")
    parser.add_argument("--dir", default="sequential_inference", help="Directory of files to verify")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        pattern = os.path.join(args.dir, "sequential_*.json")
        files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found in {args.dir} matching pattern.")
        sys.exit(1)

    all_pass = True
    for f in files:
        if not verify_strict(f):
            all_pass = False
    
    if all_pass:
        print("\n\n>>> ALL FILES PASSED STRICT VERIFICATION <<<")
        sys.exit(0)
    else:
        print("\n\n>>> SOME FILES FAILED STRICT VERIFICATION <<<")
        sys.exit(1)

if __name__ == "__main__":
    main()
