# Sentence-Transformer Support

This document explains the sentence-transformer integration added to the BEIR evaluation script.

## Overview

The evaluation script now supports two types of retrieval models:

1. **inf-retriever** (`model_id="inf"`)
   - Custom implementation using `infly/inf-retriever-v1-pro`
   - Manual tokenization + last-token pooling
   - Supports long contexts (8192 tokens)

2. **sentence-transformers** (any other `model_id`)
   - Uses the `sentence-transformers` library
   - Automatic encoding via `model.encode()`
   - Works with any HuggingFace model compatible with sentence-transformers

## Implementation Details

### Architecture

```
run_beir.py
    ├─ Determines retrieval function based on model_id
    │  ├─ model_id == "inf" → RETRIEVAL_FUNCS['inf']
    │  └─ model_id != "inf" → RETRIEVAL_FUNCS['sentence_transformer']
    │
retrievers.py
    ├─ retrieval_inf()
    │  └─ Manual: tokenizer → model → last_token_pool → normalize
    └─ retrieval_sentence_transformer()
       └─ Automatic: model.encode(normalize_embeddings=True)
```

### Key Features

Both retrieval functions support:
- ✓ Document embedding caching (model-specific)
- ✓ Instruction prefix for queries
- ✓ Batch processing
- ✓ MRL dimension truncation (`--embedding_dim`)
- ✓ Re-normalization after truncation

### Code Changes

**retrievers.py**
- Added `from sentence_transformers import SentenceTransformer`
- Added `retrieval_sentence_transformer()` function
- Updated `RETRIEVAL_FUNCS` dict with new function

**run_beir.py**
- Changed import from `retrieval_inf` to `RETRIEVAL_FUNCS`
- Removed model choices restriction
- Added routing logic to select retrieval function
- Updated output directory to sanitize model names (replace `/` with `_`)

## Usage Examples

### Basic Usage

```bash
# inf-retriever
python run_beir.py --dataset nfcorpus --model inf

# sentence-transformer models
python run_beir.py --dataset nfcorpus --model sentence-transformers/all-MiniLM-L6-v2
python run_beir.py --dataset nfcorpus --model BAAI/bge-large-en-v1.5
python run_beir.py --dataset nfcorpus --model Alibaba-NLP/gte-large-en-v1.5
```

### With MRL Testing

```bash
# Test sentence-transformer with dimension truncation
python run_beir.py \
    --dataset nfcorpus \
    --model BAAI/bge-large-en-v1.5 \
    --embedding_dim 512
```

### Batch Size Configuration

```bash
# Adjust batch size for different hardware
python run_beir.py \
    --dataset nfcorpus \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --encode_batch_size 64
```

## Cache Behavior

Document embeddings are cached at:
```
cache_beir/doc_emb/{model_id_sanitized}/{dataset}/long_False_{batch_size}.npy
```

Example cache paths:
- `cache_beir/doc_emb/inf/nfcorpus/long_False_16.npy`
- `cache_beir/doc_emb/sentence-transformers_all-MiniLM-L6-v2/nfcorpus/long_False_32.npy`

Cache is invalidated if:
- Model ID changes
- Dataset changes
- Batch size changes
- Cache file is corrupted or incomplete

## Testing

Quick test to verify sentence-transformer support:

```bash
bash test_sentence_transformer.sh
```

This runs a single evaluation with `all-MiniLM-L6-v2` on nfcorpus.

## Notes

- The instruction prefix is applied to ALL models (including sentence-transformers)
- For sentence-transformers, `--doc_max_length` is ignored (model's default is used)
- Embeddings are normalized by default (via `normalize_embeddings=True`)
- After dimension truncation, embeddings are re-normalized to maintain unit length
