#!/usr/bin/env python3
"""
Compare retrieval results across different embedding dimensions to analyze MRL performance.
"""

import os
import json
import argparse
from pathlib import Path


def load_results(output_dir, dataset, dimensions):
    """Load results for all specified dimensions."""
    results = {}

    # Load full dimension results
    full_path = Path(output_dir) / f"{dataset}_inf" / "results.json"
    if full_path.exists():
        with open(full_path) as f:
            results['full'] = json.load(f)

    # Load reduced dimension results
    for dim in dimensions:
        dim_path = Path(output_dir) / f"{dataset}_inf_dim{dim}" / "results.json"
        if dim_path.exists():
            with open(dim_path) as f:
                results[dim] = json.load(f)

    return results


def print_comparison_table(results, metrics=['NDCG@10', 'Recall@10', 'Recall@100', 'MRR']):
    """Print a comparison table of results."""
    if not results:
        print("No results found!")
        return

    # Print header
    print("\n" + "="*80)
    print("MRL Performance Comparison")
    print("="*80)

    # Determine column width
    dim_labels = list(results.keys())
    col_width = max(len(str(label)) for label in dim_labels) + 2

    # Print table header
    header = f"{'Metric':<15}"
    for label in dim_labels:
        header += f"{str(label):>{col_width}}"
    header += f"  {'Drop %':>8}"
    print(header)
    print("-" * len(header))

    # Print metrics
    baseline_key = dim_labels[0]  # Assume first key is baseline (usually 'full')

    for metric in metrics:
        if metric not in results[baseline_key]:
            continue

        row = f"{metric:<15}"
        baseline_value = results[baseline_key][metric]

        for label in dim_labels:
            value = results[label].get(metric, 0.0)
            row += f"{value:>{col_width}.4f}"

        # Calculate drop from baseline
        last_value = results[dim_labels[-1]].get(metric, 0.0)
        if baseline_value > 0:
            drop_pct = ((baseline_value - last_value) / baseline_value) * 100
            row += f"  {drop_pct:>7.2f}%"
        else:
            row += f"  {'N/A':>8}"

        print(row)

    print("="*80)

    # Print analysis
    print("\nAnalysis:")
    if len(dim_labels) > 1:
        baseline_value = results[baseline_key]['NDCG@10']
        last_value = results[dim_labels[-1]]['NDCG@10']
        drop_pct = ((baseline_value - last_value) / baseline_value) * 100

        if drop_pct < 10:
            print(f"✓ Excellent MRL support: Only {drop_pct:.2f}% NDCG@10 drop at smallest dimension")
        elif drop_pct < 20:
            print(f"○ Good MRL support: {drop_pct:.2f}% NDCG@10 drop at smallest dimension")
        elif drop_pct < 40:
            print(f"△ Moderate MRL support: {drop_pct:.2f}% NDCG@10 drop at smallest dimension")
        else:
            print(f"✗ Poor/No MRL support: {drop_pct:.2f}% NDCG@10 drop at smallest dimension")
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare MRL results across dimensions')
    parser.add_argument('--dataset', type=str, default='nfcorpus',
                        choices=['nfcorpus', 'scidocs', 'scifact'],
                        help='Dataset to analyze')
    parser.add_argument('--output_dir', type=str, default='outputs_beir',
                        help='Output directory containing results')
    parser.add_argument('--dimensions', type=int, nargs='+',
                        default=[1024, 512, 256, 128, 64],
                        help='Dimensions to compare')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['NDCG@10', 'Recall@10', 'Recall@100', 'MRR'],
                        help='Metrics to display')
    args = parser.parse_args()

    results = load_results(args.output_dir, args.dataset, args.dimensions)

    if not results:
        print(f"No results found for {args.dataset} in {args.output_dir}")
        print("Make sure you've run the evaluation first!")
        return

    print_comparison_table(results, args.metrics)


if __name__ == '__main__':
    main()
