"""
Generate tuning_summary.json from existing individual tuning result files

This script scans all tuning result JSON files and creates a consolidated
tuning_summary.json file for caching purposes.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def generate_tuning_summary():
    """Generate tuning summary from individual tuning result files"""

    # Define paths
    tuning_dir = Path("saved_models/tuning_results")

    # Also check old experiment directories for tuning results
    results_dir = Path("results")

    # Create tuning directory if it doesn't exist
    tuning_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for tuning result files...")
    print(f"  - Primary location: {tuning_dir}")
    print(f"  - Experiment folders: {results_dir}")

    # Collect all tuning result files
    tuning_files = []

    # From saved_models/tuning_results/
    if tuning_dir.exists():
        tuning_files.extend(list(tuning_dir.glob("tuning_*.json")))

    # From results/experiment_*/tuning_results/
    if results_dir.exists():
        for exp_dir in results_dir.glob("experiment_*/tuning_results"):
            tuning_files.extend(list(exp_dir.glob("tuning_*.json")))

    print(f"\nFound {len(tuning_files)} tuning result files")

    if not tuning_files:
        print("No tuning files found. Nothing to do.")
        return

    # Parse all tuning results
    # Group by dataset_model to keep only the latest result for each combination
    results_by_key = {}  # key: (dataset, model) -> result data

    for filepath in tuning_files:
        try:
            with open(filepath, 'r') as f:
                result = json.load(f)

            # Extract key info
            dataset_name = result.get('dataset_name')
            model_name = result.get('model_name')

            if not dataset_name or not model_name:
                print(f"  [SKIP] Missing dataset/model name: {filepath.name}")
                continue

            key = f"{dataset_name}_{model_name}"

            # Keep the most recent result for each combination
            # Compare by timestamp
            result_timestamp = result.get('timestamp', '')

            if key in results_by_key:
                existing_timestamp = results_by_key[key]['timestamp']
                if result_timestamp > existing_timestamp:
                    results_by_key[key] = result
                    print(f"  [UPDATE] {key} (newer timestamp)")
                else:
                    print(f"  [SKIP] {key} (older timestamp)")
            else:
                results_by_key[key] = result
                print(f"  [ADD] {key}")

        except Exception as e:
            print(f"  [ERROR] Failed to parse {filepath.name}: {e}")
            continue

    print(f"\nProcessed {len(results_by_key)} unique dataset-model combinations")

    # Create summary structure
    summary = {
        'tuning_summary': {
            'total_datasets': len(set(r['dataset_name'] for r in results_by_key.values())),
            'total_models': len(results_by_key),
            'successful_tunings': len(results_by_key),
            'failed_tunings': 0,
            'timestamp': datetime.now().isoformat(),
            'generated_from': f"{len(tuning_files)} individual tuning files"
        },
        'best_parameters': {},
        'performance_summary': {}
    }

    # Populate best parameters and performance summary
    for key, result in results_by_key.items():
        summary['best_parameters'][key] = result.get('best_params', {})
        summary['performance_summary'][key] = {
            'cv_score': result.get('best_score'),
            'test_accuracy': result.get('test_accuracy'),
            'test_f1': result.get('test_f1'),
            'test_roc_auc': result.get('test_roc_auc'),
            'tuning_time': result.get('tuning_time'),
            'timestamp': result.get('timestamp')
        }

    # Save summary file
    summary_file = tuning_dir / "tuning_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUCCESS] Generated tuning summary file:")
    print(f"  Location: {summary_file.absolute()}")
    print(f"  Datasets: {summary['tuning_summary']['total_datasets']}")
    print(f"  Models: {summary['tuning_summary']['total_models']}")
    print(f"\nCached tuning results:")

    # Group by dataset for better display
    by_dataset = defaultdict(list)
    for key in sorted(summary['best_parameters'].keys()):
        dataset, model = key.rsplit('_', 1)
        by_dataset[dataset].append(model)

    for dataset in sorted(by_dataset.keys()):
        models = ', '.join(sorted(by_dataset[dataset]))
        print(f"  {dataset}: {models}")

    print(f"\nTuning results are now cached. Future runs will skip re-tuning for these combinations.")

if __name__ == "__main__":
    print("="*80)
    print("Tuning Summary Generator")
    print("="*80)
    print()

    generate_tuning_summary()

    print()
    print("="*80)
    print("Done!")
    print("="*80)
