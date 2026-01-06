#!/bin/bash

# Run evaluation on all three BEIR datasets

echo "===== Evaluating on nfcorpus ====="
python run_beir.py \
    --dataset nfcorpus \
    --model inf \
    --doc_max_length 8192 \
    --encode_batch_size 64

echo ""
echo "===== Evaluating on scidocs ====="
python run_beir.py \
    --dataset scidocs \
    --model inf \
    --doc_max_length 8192 \
    --encode_batch_size 64

echo ""
echo "===== Evaluating on scifact ====="
python run_beir.py \
    --dataset scifact \
    --model inf \
    --doc_max_length 8192 \
    --encode_batch_size 64

echo ""
echo "===== All evaluations complete ====="
echo "Results are in outputs_beir/[dataset]_inf/results.json"
