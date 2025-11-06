"""
Analyze text explanations to count valid values, zeros, and NaNs
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_text_explanations():
    """Analyze all text explanations in the results"""

    print("=" * 80)
    print("TEXT EXPLANATIONS ANALYSIS")
    print("=" * 80)

    # Find the latest experiment with benchmark_results.json
    results_dir = Path("results")
    experiments_with_results = []

    for exp_dir in results_dir.glob("experiment_*"):
        if (exp_dir / "benchmark_results.json").exists():
            experiments_with_results.append(exp_dir)

    if not experiments_with_results:
        print("No experiments with benchmark_results.json found")
        return

    latest_experiment = max(experiments_with_results, key=lambda x: x.stat().st_mtime)
    print(f"Analyzing: {latest_experiment.name}")

    benchmark_file = latest_experiment / "benchmark_results.json"

    with open(benchmark_file, 'r') as f:
        data = json.load(f)

    # Text dataset indicators
    text_datasets = ['imdb', 'ag_news', '20newsgroups', 'text']

    # Statistics collectors
    stats = defaultdict(lambda: {
        'total': 0,
        'valid': 0,
        'zero': 0,
        'nan': 0,
        'missing': 0,
        'values': []
    })

    # Navigate through the JSON structure
    if isinstance(data, dict):
        # Try different possible structures
        for key in ['results', 'experiments', 'data', 'benchmarks']:
            if key in data:
                process_results(data[key], text_datasets, stats)
                break
        else:
            # Maybe data itself is the results
            process_results(data, text_datasets, stats)

    # Print statistics
    print("\n" + "=" * 80)
    print("MONOTONICITY STATISTICS FOR TEXT EXPLANATIONS")
    print("=" * 80)

    for dataset in sorted(stats.keys()):
        if any(text_ind in dataset.lower() for text_ind in text_datasets):
            print(f"\n{dataset}:")
            dataset_stats = stats[dataset]

            print(f"  Total explanations: {dataset_stats['total']}")
            print(f"  Valid values (> 0): {dataset_stats['valid']} ({dataset_stats['valid']/max(dataset_stats['total'], 1)*100:.1f}%)")
            print(f"  Zero values: {dataset_stats['zero']} ({dataset_stats['zero']/max(dataset_stats['total'], 1)*100:.1f}%)")
            print(f"  NaN values: {dataset_stats['nan']} ({dataset_stats['nan']/max(dataset_stats['total'], 1)*100:.1f}%)")
            print(f"  Missing monotonicity: {dataset_stats['missing']} ({dataset_stats['missing']/max(dataset_stats['total'], 1)*100:.1f}%)")

            if dataset_stats['values']:
                valid_values = [v for v in dataset_stats['values'] if v > 0]
                if valid_values:
                    print(f"  Sample valid values: {valid_values[:5]}")

    # Summary across all text datasets
    print("\n" + "=" * 80)
    print("OVERALL TEXT SUMMARY")
    print("=" * 80)

    total_text = sum(stats[d]['total'] for d in stats if any(t in d.lower() for t in text_datasets))
    total_valid = sum(stats[d]['valid'] for d in stats if any(t in d.lower() for t in text_datasets))
    total_zero = sum(stats[d]['zero'] for d in stats if any(t in d.lower() for t in text_datasets))
    total_nan = sum(stats[d]['nan'] for d in stats if any(t in d.lower() for t in text_datasets))
    total_missing = sum(stats[d]['missing'] for d in stats if any(t in d.lower() for t in text_datasets))

    print(f"Total text explanations: {total_text}")
    print(f"Valid (> 0): {total_valid} ({total_valid/max(total_text, 1)*100:.1f}%)")
    print(f"Zero: {total_zero} ({total_zero/max(total_text, 1)*100:.1f}%)")
    print(f"NaN: {total_nan} ({total_nan/max(total_text, 1)*100:.1f}%)")
    print(f"Missing: {total_missing} ({total_missing/max(total_text, 1)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("METHOD-WISE BREAKDOWN")
    print("=" * 80)

    # Analyze by method
    method_stats = defaultdict(lambda: {'total': 0, 'valid': 0, 'zero': 0, 'nan': 0})

    # Look for detailed explanations directory
    detailed_dir = latest_experiment / "detailed_explanations"
    if detailed_dir.exists():
        for json_file in detailed_dir.glob("*.json"):
            if any(t in json_file.name.lower() for t in text_datasets):
                try:
                    with open(json_file, 'r') as f:
                        exp_data = json.load(f)

                    analyze_explanation_file(exp_data, method_stats)
                except:
                    pass

    for method in sorted(method_stats.keys()):
        method_stat = method_stats[method]
        if method_stat['total'] > 0:
            print(f"\n{method}:")
            print(f"  Total: {method_stat['total']}")
            print(f"  Valid: {method_stat['valid']} ({method_stat['valid']/method_stat['total']*100:.1f}%)")
            print(f"  Zero: {method_stat['zero']} ({method_stat['zero']/method_stat['total']*100:.1f}%)")
            print(f"  NaN: {method_stat['nan']} ({method_stat['nan']/method_stat['total']*100:.1f}%)")

def process_results(data, text_datasets, stats):
    """Recursively process results to find text monotonicity values"""

    if isinstance(data, dict):
        for key, value in data.items():
            # Check if this is a dataset name
            if any(text_ind in key.lower() for text_ind in text_datasets):
                # This might be a text dataset
                if isinstance(value, dict):
                    extract_monotonicity(value, key, stats)

            # Recurse
            if isinstance(value, (dict, list)):
                process_results(value, text_datasets, stats)

    elif isinstance(data, list):
        for item in data:
            process_results(item, text_datasets, stats)

def extract_monotonicity(data, dataset_name, stats):
    """Extract monotonicity values from a dataset's results"""

    if isinstance(data, dict):
        # Look for fidelity_metrics or similar
        if 'fidelity_metrics' in data:
            metrics = data['fidelity_metrics']
            if 'monotonicity' in metrics:
                value = metrics['monotonicity']
                analyze_value(value, dataset_name, stats)

        # Look for monotonicity directly
        if 'monotonicity' in data:
            value = data['monotonicity']
            analyze_value(value, dataset_name, stats)

        # Recurse
        for key, value in data.items():
            if isinstance(value, dict):
                extract_monotonicity(value, dataset_name, stats)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        extract_monotonicity(item, dataset_name, stats)

def analyze_value(value, dataset_name, stats):
    """Analyze a monotonicity value"""

    stats[dataset_name]['total'] += 1

    if value is None:
        stats[dataset_name]['missing'] += 1
    elif isinstance(value, (int, float)):
        if np.isnan(value):
            stats[dataset_name]['nan'] += 1
        elif value == 0:
            stats[dataset_name]['zero'] += 1
        elif value > 0:
            stats[dataset_name]['valid'] += 1
            stats[dataset_name]['values'].append(value)

def analyze_explanation_file(data, method_stats):
    """Analyze a detailed explanation file"""

    if isinstance(data, dict):
        # Look for method name
        method = data.get('method', 'unknown')

        # Look for explanations
        if 'explanations' in data:
            explanations = data['explanations']
            if isinstance(explanations, list):
                for exp in explanations:
                    if isinstance(exp, dict):
                        # Check if this has text content
                        if 'text_content' in exp or 'tokens' in exp:
                            method_stats[method]['total'] += 1

                            # Check for monotonicity score
                            if 'monotonicity' in exp:
                                value = exp['monotonicity']
                                if value is None:
                                    pass  # Missing
                                elif isinstance(value, (int, float)):
                                    if np.isnan(value):
                                        method_stats[method]['nan'] += 1
                                    elif value == 0:
                                        method_stats[method]['zero'] += 1
                                    else:
                                        method_stats[method]['valid'] += 1

if __name__ == "__main__":
    analyze_text_explanations()