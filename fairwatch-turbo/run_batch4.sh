#!/bin/bash

# Configuration
LIMIT=5760
CONCURRENCY=2000
OUTPUT_DIR="/DATA1/ai24resch11001/nikhil/fairwatch_vllm_turbo/batch4_outputs"
mkdir -p $OUTPUT_DIR

wait_for_port() {
    PORT=$1
    echo "Waiting for Port $PORT..."
    while ! nc -z localhost $PORT; do   
      sleep 5
    done
    echo "Port $PORT is open!"
}

run_order() {
    ORDER_NUM=$1
    ORDER_STRING=$2
    FILENAME=$3
    PORT=$4
    
    echo "Starting Order $ORDER_NUM on Port $PORT..."
    
    # 1. Run Benchmark
    python3 /DATA1/ai24resch11001/nikhil/fairwatch_vllm_turbo/orchestrator_turbo.py \
        --ordering "$ORDER_STRING" \
        --limit $LIMIT \
        --output "$OUTPUT_DIR/${FILENAME}.json" \
        --port $PORT \
        --concurrency $CONCURRENCY
        
    if [ $? -ne 0 ]; then
        echo "ERROR: Order $ORDER_NUM Failed during benchmark."
        return 1
    fi

    echo "Order $ORDER_NUM Benchmark Complete. Starting Conversion..."
    
    # 2. Convert to Legacy
    python3 /DATA1/ai24resch11001/nikhil/fairwatch_vllm_turbo/convert_vllm_to_legacy.py \
        --input "$OUTPUT_DIR/${FILENAME}.json" \
        --output "$OUTPUT_DIR/${FILENAME}_legacy.json" \
        --ordering "$ORDER_STRING" \
        --port $PORT
        
    echo "Order $ORDER_NUM Finished."
}

# Stream Selector
STREAM=$1

if [ "$STREAM" == "A" ]; then
    echo "Launching STREAM A (GPU 0 / Port 8006)..."
    wait_for_port 8006
    run_order 7 "risk_manager,data_science,consumer_advocate,regulatory" "sequential_risk_data_consumer_regulatory" 8006
    run_order 16 "data_science,risk_manager,consumer_advocate,regulatory" "sequential_data_risk_consumer_regulatory" 8006
    run_order 19 "data_science,consumer_advocate,risk_manager,regulatory" "sequential_data_consumer_risk_regulatory" 8006
    run_order 22 "consumer_advocate,regulatory,risk_manager,data_science" "sequential_consumer_regulatory_risk_data" 8006

elif [ "$STREAM" == "B" ]; then
    echo "Launching STREAM B (GPU 1 / Port 8007)..."
    wait_for_port 8007
    run_order 8 "risk_manager,consumer_advocate,regulatory,data_science" "sequential_risk_consumer_regulatory_data" 8007
    run_order 17 "data_science,regulatory,risk_manager,consumer_advocate" "sequential_data_regulatory_risk_consumer" 8007
    run_order 20 "consumer_advocate,risk_manager,regulatory,data_science" "sequential_consumer_risk_regulatory_data" 8007
    run_order 23 "consumer_advocate,regulatory,data_science,risk_manager" "sequential_consumer_regulatory_data_risk" 8007

elif [ "$STREAM" == "C" ]; then
    echo "Launching STREAM C (GPU 2 / Port 8005)..."
    
    # Wait for Batch 3 conversion
    echo "Checking if Batch 3 conversion is running..."
    while pgrep -f "convert_vllm_to_legacy.py" > /dev/null; do
        echo "Batch 3 Conversion still active. Waiting 60s..."
        sleep 60
    done
    echo "GPU 2 Free. Starting Stream C."
    
    wait_for_port 8005
    run_order 15 "data_science,risk_manager,regulatory,consumer_advocate" "sequential_data_risk_regulatory_consumer" 8005
    run_order 18 "data_science,regulatory,consumer_advocate,risk_manager" "sequential_data_regulatory_consumer_risk" 8005
    run_order 21 "consumer_advocate,risk_manager,data_science,regulatory" "sequential_consumer_risk_data_regulatory" 8005
    run_order 24 "consumer_advocate,data_science,risk_manager,regulatory" "sequential_consumer_data_risk_regulatory" 8005
else
    echo "Usage: ./run_batch4.sh [A|B|C]"
fi
