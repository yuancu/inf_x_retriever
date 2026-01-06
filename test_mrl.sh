#!/bin/bash

# Test Matryoshka Representation Learning (MRL) on a single dataset
# This script evaluates the model with different embedding dimensions

DATASET=${1:-nfcorpus}
echo "Testing MRL on dataset: $DATASET"
echo "================================"

# Test full dimensionality (baseline)
echo ""
echo "===== Testing FULL dimensions ====="
python run_beir.py \
    --dataset $DATASET \
    --model inf \
    --doc_max_length 8192 \
    --encode_batch_size 64

# Test common MRL dimensions
for DIM in 1024 512 256 128 64; do
    echo ""
    echo "===== Testing with $DIM dimensions ====="
    python run_beir.py \
        --dataset $DATASET \
        --model inf \
        --doc_max_length 8192 \
        --encode_batch_size 64 \
        --embedding_dim $DIM
done

echo ""
echo "===== MRL Testing Complete ====="
echo "Compare results in:"
echo "  outputs_beir/${DATASET}_inf/results.json (full)"
echo "  outputs_beir/${DATASET}_inf_dim1024/results.json"
echo "  outputs_beir/${DATASET}_inf_dim512/results.json"
echo "  outputs_beir/${DATASET}_inf_dim256/results.json"
echo "  outputs_beir/${DATASET}_inf_dim128/results.json"
echo "  outputs_beir/${DATASET}_inf_dim64/results.json"
