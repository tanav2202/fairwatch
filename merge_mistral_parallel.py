import json
import glob
import os

def merge_shards(pattern, output_file):
    files = sorted(glob.glob(pattern))
    print(f"Merging {len(files)} files matching {pattern} into {output_file}")
    
    all_results = []
    for f in files:
        with open(f, 'r') as fh:
            data = json.load(fh)
            results = data.get('results', [])
            all_results.extend(results)
            print(f"  Loaded {len(results)} from {f}")

    # Sort by prompt_id if present
    all_results.sort(key=lambda x: x.get('prompt_id', 0))

    final_data = {
        "metadata": {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "total_records": len(all_results),
            "status": "complete",
            "shards_merged": len(files)
        },
        "results": all_results,
        "status": "complete"
    }

    with open(output_file, 'w') as fh:
        json.dump(final_data, fh, indent=2)
    print(f"✅ Successfully merged into {output_file} (Total: {len(all_results)})")

if __name__ == "__main__":
    # Merge Complex
    merge_shards(
        "outputs_complex/mistral:latest/parallel/shard_gpu*.json", 
        "outputs_complex/mistral:latest/parallel/parallel_synthesis_complex.json"
    )
    
    # Merge Simple (copy them first or point to the origin)
    # Origin: /DATA1/ai24resch11001/nikhil/fairwatch_mistral_turbo/parallel_shard_gpu*.json
    # I'll point directly to the origin for Simple to be safe.
    os.makedirs("outputs_simple/mistral:latest/parallel", exist_ok=True)
    merge_shards(
        "/DATA1/ai24resch11001/nikhil/fairwatch_mistral_turbo/parallel_shard_gpu*.json",
        "outputs_simple/mistral:latest/parallel/parallel_synthesis_simple.json"
    )
