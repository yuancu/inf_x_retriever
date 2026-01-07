#!/bin/bash

# Quick test script to verify sentence-transformer support
# Uses a small model on a single dataset

echo "Testing sentence-transformer support with all-MiniLM-L6-v2"
echo "============================================================"
echo ""
echo "This will run a quick evaluation on nfcorpus dataset"
echo "to verify the sentence-transformer integration works."
echo ""

python run_beir.py \
    --dataset nfcorpus \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --encode_batch_size 32

echo ""
echo "Test complete! Check outputs_beir/nfcorpus_sentence-transformers_all-MiniLM-L6-v2/results.json"
