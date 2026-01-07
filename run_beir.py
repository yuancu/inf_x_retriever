import os
import argparse
import json
from tqdm import tqdm
from retrievers import RETRIEVAL_FUNCS, calculate_retrieval_metrics
from beir import util
from beir.datasets.data_loader import GenericDataLoader

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['nfcorpus', 'scidocs', 'scifact'])
    parser.add_argument('--model', type=str, default='inf',
                        help='Model ID: "inf" for infly/inf-retriever-v1-pro, or any HuggingFace model ID for sentence-transformers')
    parser.add_argument('--doc_max_length', type=int, default=8192)
    parser.add_argument('--encode_batch_size', type=int, default=16)
    parser.add_argument('--embedding_dim', type=int, default=None,
                        help='Use only the first x dimensions for retrieval (for MRL evaluation)')
    parser.add_argument('--output_dir', type=str, default='outputs_beir')
    parser.add_argument('--cache_dir', type=str, default='cache_beir')
    parser.add_argument('--beir_data_dir', type=str, default='beir_datasets')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    # Setup output directory
    # Use simplified model name for directory (replace / with _)
    model_name = args.model.replace('/', '_')
    dim_suffix = f"_dim{args.embedding_dim}" if args.embedding_dim else ""
    args.output_dir = os.path.join(args.output_dir, f"{args.dataset}_{model_name}{dim_suffix}")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    score_file_path = os.path.join(args.output_dir, 'score.json')

    # Download and load BEIR dataset
    print(f"Loading BEIR dataset: {args.dataset}")
    dataset_path = os.path.join(args.beir_data_dir, args.dataset)
    if not os.path.exists(dataset_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
        data_path = util.download_and_unzip(url, args.beir_data_dir)
    else:
        data_path = dataset_path

    # Load corpus, queries, and qrels
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    print(f"Loaded {len(queries)} queries and {len(corpus)} documents")

    # Prepare documents and queries
    doc_ids = list(corpus.keys())
    documents = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in doc_ids]

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    # For BEIR, there are no excluded_ids, so we create an empty dict
    excluded_ids = {qid: [] for qid in query_ids}

    if not os.path.isfile(score_file_path):
        print("Running retrieval...")

        # Define instruction for retriever
        instructions = {
            'query': "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
        }

        kwargs = {
            'doc_max_length': args.doc_max_length,
            'encode_batch_size': args.encode_batch_size,
        }

        if args.checkpoint is not None:
            kwargs['checkpoint'] = args.checkpoint

        if args.embedding_dim is not None:
            kwargs['embedding_dim'] = args.embedding_dim
            print(f"Using only first {args.embedding_dim} dimensions for retrieval")

        # Determine which retrieval function to use
        if args.model == 'inf':
            retrieval_func = RETRIEVAL_FUNCS['inf']
        else:
            # Use sentence-transformer for any other model ID
            retrieval_func = RETRIEVAL_FUNCS['sentence_transformer']

        scores = retrieval_func(
            queries=query_texts,
            query_ids=query_ids,
            documents=documents,
            excluded_ids=excluded_ids,
            instructions=instructions,
            doc_ids=doc_ids,
            task=args.dataset,
            cache_dir=args.cache_dir,
            long_context=False,
            model_id=args.model,
            **kwargs
        )

        with open(score_file_path, 'w') as f:
            json.dump(scores, f, indent=2)
    else:
        print(f"Loading existing scores from {score_file_path}")
        with open(score_file_path) as f:
            scores = json.load(f)

    # Convert qrels to the format expected by calculate_retrieval_metrics
    # qrels format: {qid: {doc_id: relevance}}
    ground_truth = {}
    for qid in qrels:
        ground_truth[qid] = {}
        for doc_id, relevance in qrels[qid].items():
            if relevance > 0:  # Only consider relevant documents
                ground_truth[qid][doc_id] = relevance

    print(f"\nEvaluating on {args.dataset}...")
    print(f"Output directory: {args.output_dir}")
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {os.path.join(args.output_dir, 'results.json')}")
