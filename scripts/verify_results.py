import json
import os
import glob
import sys

def verify_strict(filepath):
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

    required_keys = [
        "mode", "initial_prompt", "conversation_history", 
        "all_agent_outputs", "final_output", "business_decision", 
        "prompt_id", "ordering"
    ]
    
    required_biz_keys = [
        "approval_decision", "interest_rate", "confidence_probability", "reasoning"
    ]

    total = len(results)
    failures = 0

    for idx, r in enumerate(results):
        # 1. Check Top-Level Keys
        missing = [k for k in required_keys if k not in r]
        if missing:
            print(f"  [FAIL] Record {idx}: Missing keys {missing}")
            failures += 1
            continue

        # 2. Check Business Decision
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

        # 3. Value Checks
        # Confidence > 0
        conf = biz.get("confidence_probability")
        if conf is None or not isinstance(conf, (int, float)) or conf <= 0:
             print(f"  [FAIL] Record {idx}: Invalid confidence_probability: {conf}")
             failures += 1
             continue
        
        # Interest Rate (can be 0 if denied, but must be prevalent)
        rate = biz.get("interest_rate")
        if rate is None:
            print(f"  [FAIL] Record {idx}: Interest rate is None (null).")
            failures += 1
            continue

        # Error Strings in Reasoning
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
    directory = "/DATA1/ai24resch11001/nikhil/fairwatch_vllm_turbo/batch4_outputs"
    pattern = os.path.join(directory, "sequential_*_legacy.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("No files found matching pattern.")
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
