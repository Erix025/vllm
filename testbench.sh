#!/bin/bash
export DATASET_DIR=/home/infer/lx/ASC24-LLM-inference-optimization/scrambled_sampled_dataset.json
export MODEL_DIR=/home/infer/models/Llama-2-7b-chat-hf

python benchmark.py \
    --num-samples 1000 \
    --dataset ${DATASET_DIR} \
    --model ${MODEL_DIR} \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --quantization atom 
