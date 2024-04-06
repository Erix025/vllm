#!/bin/bash
export DATASET_DIR=/home/infer/lcx/ASC24-LLM-inference-optimization/scrambled_sampled_dataset.json
export MODEL_DIR=/home/infer/models/AquilaChat2-34B

python benchmark.py \
    --num-samples 1000 \
    --dataset ${DATASET_DIR} \
    --model ${MODEL_DIR} \
    --tensor-parallel-size 8 \
    --trust-remote-code
    # --quantization atom \
