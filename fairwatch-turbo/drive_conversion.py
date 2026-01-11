
import os
import glob
import subprocess
import time
from pathlib import Path

DIR = "/DATA1/ai24resch11001/nikhil/fairwatch_vllm_turbo/batch4_outputs"
PORTS = [8005, 8006, 8007]

def main():
    turbo_files = glob.glob(os.path.join(DIR, "sequential_*.json"))
    turbo_files = [f for f in turbo_files if "_legacy.json" not in f]
    
    print(f"Found {len(turbo_files)} Turbo files to convert.")
    
    procs = []
    p_idx = 0
    
    mapping_rules = {
        "consumer": "consumer_advocate",
        "data": "data_science",
        "risk": "risk_manager",
        "regulatory": "regulatory"
    }

    for f in turbo_files:
        filename = os.path.basename(f)
        # e.g. sequential_consumer_data_risk_regulatory.json
        ordering_str = filename.replace("sequential_", "").replace(".json", "")
        
        # Map components
        parts = ordering_str.split('_')
        mapped_parts = []
        for p in parts:
            mapped_parts.append(mapping_rules.get(p, p)) # Fallback to same if not found
            
        final_ordering = ",".join(mapped_parts)
        outfile = f.replace(".json", "_legacy.json")
        port = PORTS[p_idx % len(PORTS)]
        p_idx += 1
        
        cmd = [
            "python3", "convert_vllm_to_legacy.py",
            "--input", f,
            "--output", outfile,
            "--ordering", final_ordering,
            "--port", str(port)
        ]
        
        print(f"Launching on port {port}: {filename} -> Ordering: {final_ordering}")
        p = subprocess.Popen(cmd)
        procs.append(p)
        
        # Throttling: Wait if we have 3 running
        if len(procs) >= len(PORTS):
             # Wait for at least one to finish?
             # Simple block: wait for batch of 3 to finish before starting next batch
             # To maximize throughput we should use a pool, but this simple batching is safer for now.
             if p_idx % len(PORTS) == 0:
                 print("Waiting for batch to complete...")
                 for p in procs:
                     p.wait()
                 procs = []

    # Wait for remaining
    for p in procs:
        p.wait()
        
    print("All conversions completed.")

if __name__ == "__main__":
    main()
