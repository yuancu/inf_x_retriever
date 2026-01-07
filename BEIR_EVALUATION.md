# BEIR Evaluation for inf-retriever-v1-pro

This directory contains scripts to evaluate the inf-retriever-v1-pro model on BEIR datasets without using query rewriting.

## Setup

Ensure you have the required dependencies installed:
```bash
pip install beir transformers torch
```

## Usage

### Evaluate on a single dataset

#### Using inf-retriever (default)

```bash
python run_beir.py --dataset nfcorpus
```

#### Using sentence-transformer models

```bash
# Example with all-MiniLM-L6-v2
python run_beir.py --dataset nfcorpus --model sentence-transformers/all-MiniLM-L6-v2

# Example with GTE model
python run_beir.py --dataset nfcorpus --model Alibaba-NLP/gte-large-en-v1.5

# Example with custom batch size
python run_beir.py --dataset nfcorpus --model BAAI/bge-large-en-v1.5 --encode_batch_size 32
```

Available datasets:
- `nfcorpus`: NF-Corpus (Nutrition/Medical)
- `scidocs`: SciDocs (Scientific Papers)
- `scifact`: SciFact (Scientific Claims)

### Evaluate on all three datasets

```bash
bash run_all_beir.sh
```

### Command line arguments

- `--dataset`: BEIR dataset name (required, choices: nfcorpus, scidocs, scifact)
- `--model`: Model ID to use (default: inf)
  - Use `inf` for infly/inf-retriever-v1-pro
  - Use any HuggingFace model ID for sentence-transformers (e.g., `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-large-en-v1.5`)
- `--doc_max_length`: Maximum document length (default: 8192) - applies to inf model only
- `--encode_batch_size`: Batch size for encoding (default: 16)
- `--embedding_dim`: Use only the first x dimensions for retrieval (optional, for MRL evaluation)
- `--output_dir`: Directory to save results (default: outputs_beir)
- `--cache_dir`: Directory to cache embeddings (default: cache_beir)
- `--beir_data_dir`: Directory to store BEIR datasets (default: beir_datasets)

## Output

Results will be saved in `outputs_beir/[dataset]_inf/`:
- `score.json`: Raw retrieval scores for each query-document pair
- `results.json`: Evaluation metrics (NDCG@k, MAP@k, Recall@k, P@k, MRR)

## Key Differences from run.py

1. **No Query Rewriting**: This script directly uses the retriever model without any query rewriting/alignment step
2. **BEIR Datasets**: Uses standard BEIR datasets instead of xlangai/bright
3. **Standard Instruction**: Uses the default instruction: "Instruct: Given a web search query, retrieve relevant passages that answer the query"
4. **No Excluded IDs**: BEIR datasets don't have excluded document IDs
5. **Multi-Model Support**: Supports both inf-retriever and sentence-transformer compatible models

## Model Support

### inf-retriever (model_id: "inf")
- Uses `infly/inf-retriever-v1-pro` from HuggingFace
- Manual tokenization with last-token pooling
- Supports long contexts (up to 8192 tokens by default)

### sentence-transformers (any other model_id)
- Uses the `sentence-transformers` library
- Automatic encoding with `model.encode()`
- Works with any HuggingFace model compatible with sentence-transformers
- Examples: `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-large-en-v1.5`, `Alibaba-NLP/gte-large-en-v1.5`
- Embeddings are automatically normalized

## Testing Matryoshka Representation Learning (MRL)

To test if the model supports MRL (using reduced dimensionality), use the `--embedding_dim` argument:

```bash
# Test with 512 dimensions
python run_beir.py --dataset nfcorpus --embedding_dim 512

# Test with 256 dimensions
python run_beir.py --dataset nfcorpus --embedding_dim 256

# Test with 128 dimensions
python run_beir.py --dataset nfcorpus --embedding_dim 128
```

### Comprehensive MRL Testing

Use the `test_mrl.sh` script to automatically test multiple dimensions on a single dataset:

```bash
# Test on nfcorpus (default)
bash test_mrl.sh

# Test on a specific dataset
bash test_mrl.sh scidocs
```

This will evaluate the model with full dimensions and then with 1024, 512, 256, 128, and 64 dimensions.

### Comparing MRL Results

After running the MRL tests, use the comparison script to analyze the results:

```bash
# Compare results for nfcorpus
python compare_mrl_results.py --dataset nfcorpus

# Compare with custom dimensions
python compare_mrl_results.py --dataset scidocs --dimensions 512 256 128

# Compare with custom metrics
python compare_mrl_results.py --dataset scifact --metrics NDCG@10 MAP@10 Recall@50
```

This will display a table showing how performance changes across dimensions and automatically analyze MRL support quality.

### Interpreting Results

The embeddings are truncated to only use the first x dimensions for similarity computation. If the model is trained with MRL, you should see **graceful degradation** in performance as dimensions decrease. If not trained with MRL, performance may **drop sharply**.

Results with different dimensions will be saved in separate directories (e.g., `outputs_beir/nfcorpus_inf_dim512/`).

## Comparing Different Models

You can compare the performance of different retrieval models on the same dataset:

```bash
# Evaluate inf-retriever
python run_beir.py --dataset nfcorpus --model inf

# Evaluate a sentence-transformer model
python run_beir.py --dataset nfcorpus --model sentence-transformers/all-MiniLM-L6-v2

# Compare results
cat outputs_beir/nfcorpus_inf/results.json
cat outputs_beir/nfcorpus_sentence-transformers_all-MiniLM-L6-v2/results.json
```

## Notes

- Document embeddings are cached to avoid recomputation. The cache is model-specific and dataset-specific.
- The instruction prefix is applied to queries for all models, which may affect performance for some sentence-transformer models.
- For sentence-transformer models, the `--doc_max_length` parameter is ignored (the model's default max length is used).
