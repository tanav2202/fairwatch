#!/bin/bash
# launch_turbo.sh - Optimized vLLM Server + Speculative Decoding

export HF_TOKEN=placeholder

# Tune threads
export OMP_NUM_THREADS=1

# Turbo Configuration
# --speculative-model: Uses 1B model to draft tokens for 3B model (Speedup ~1.5x)
# --num-speculative-tokens: Number of tokens to draft per step
# --max-num-seqs: Higher concurrency (default is usually 256)
# --gpu-memory-utilization: 0.85 safe for 1B+3B on 48GB card

echo "Starting vLLM Turbo Server on Port 8005 (GPU 2)..."

CUDA_VISIBLE_DEVICES=2 nohup /DATA1/ai24resch11001/miniconda3/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --max-num-seqs 1024 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --port 8005 \
    --trust-remote-code \
    > vllm_turbo.log 2>&1 &

echo "Server launching in background. Log: vllm_turbo.log"
