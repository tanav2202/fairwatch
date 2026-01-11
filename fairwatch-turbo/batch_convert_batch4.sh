#!/bin/bash

DIR="/DATA1/ai24resch11001/nikhil/fairwatch_vllm_turbo/batch4_outputs"
PORTS=(8005 8006 8007)
P_IDX=0

for file in $DIR/sequential_*.json; do
  # Skip existing legacy files
  if [[ "$file" == *"_legacy.json"* ]]; then
    continue
  fi
  
  filename=$(basename -- "$file")
  ordering="${filename#sequential_}"
  ordering="${ordering%.json}"
  
  outfile="${file%.json}_legacy.json"
  
  port=${PORTS[$P_IDX]}
  
  echo "Converting $filename on port $port..."
  
  python3 convert_vllm_to_legacy.py --input "$file" --output "$outfile" --ordering "$ordering" --port "$port" &
  
  P_IDX=$(( (P_IDX + 1) % 3 ))
  
  # Simple throttling to avoid OOM or overload
  if [[ $P_IDX -eq 0 ]]; then
     wait # Wait for batch of 3
  fi
  
done

wait
echo "All conversions complete."
