#!/usr/bin/env python3
"""
Utility script to list and manage saved models across all experiments
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Note: 'tabulate' not installed. Using simple table format.")


def get_all_saved_models(results_dir: Path = None):
    """Find all saved models in the results directory"""
    if results_dir is None:
        results_dir = Path("results")

    if not results_dir.exists():
        print(f"[X] Results directory not found: {results_dir}")
        return []

    all_models = []

    # Find all saved_models directories
    for experiment_dir in results_dir.glob("experiment_*/saved_models/metadata"):
        if experiment_dir.is_dir():
            experiment_name = experiment_dir.parent.parent.name

            # Read all metadata files
            for metadata_file in experiment_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        metadata['experiment'] = experiment_name
                        metadata['metadata_path'] = str(metadata_file)
                        all_models.append(metadata)
                except Exception as e:
                    print(f"[!]  Error reading {metadata_file}: {e}")

    return all_models


def format_size(size_bytes):
    """Format size in bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_model_size(metadata_path: str):
    """Get the size of the model file"""
    try:
        metadata_file = Path(metadata_path)
        model_dir = metadata_file.parent.parent / "models"

        # Find corresponding model file
        model_id = metadata_file.stem.replace("_metadata", "")

        total_size = 0
        for model_file in model_dir.glob(f"{model_id}.*"):
            if model_file.is_file():
                total_size += model_file.stat().st_size

        return total_size
    except Exception:
        return 0


def list_models(results_dir: Path = None, group_by: str = None):
    """List all saved models with details"""
    models = get_all_saved_models(results_dir)

    if not models:
        print("[X] No saved models found.")
        return

    print(f"\n[OK] Found {len(models)} saved models\n")

    # Sort by timestamp (most recent first)
    models.sort(key=lambda x: x.get('saved_timestamp', ''), reverse=True)

    if group_by == 'dataset':
        # Group by dataset
        from collections import defaultdict
        grouped = defaultdict(list)
        for model in models:
            grouped[model['dataset_name']].append(model)

        for dataset, dataset_models in grouped.items():
            print(f"\n📊 Dataset: {dataset} ({len(dataset_models)} models)")
            print("=" * 100)
            display_models_table(dataset_models)

    elif group_by == 'experiment':
        # Group by experiment
        from collections import defaultdict
        grouped = defaultdict(list)
        for model in models:
            grouped[model['experiment']].append(model)

        for exp, exp_models in grouped.items():
            print(f"\n🧪 Experiment: {exp} ({len(exp_models)} models)")
            print("=" * 100)
            display_models_table(exp_models)

    else:
        # Display all models
        display_models_table(models)


def display_models_table(models):
    """Display models in a formatted table"""
    table_data = []

    for model in models:
        # Extract key information
        dataset = model.get('dataset_name', 'Unknown')
        model_type = model.get('model_type', 'Unknown')
        timestamp = model.get('saved_timestamp', 'Unknown')

        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp)
            timestamp_short = dt.strftime("%Y-%m-%d %H:%M")
        except:
            timestamp_short = timestamp[:16] if timestamp else "Unknown"

        # Get performance metrics
        perf = model.get('performance_metrics', {})
        test_acc = perf.get('test_accuracy', 'N/A')
        train_acc = perf.get('train_accuracy', 'N/A')

        # Format accuracy
        if isinstance(test_acc, (int, float)):
            test_acc = f"{test_acc:.4f}"
        if isinstance(train_acc, (int, float)):
            train_acc = f"{train_acc:.4f}"

        # Get model size
        size = get_model_size(model.get('metadata_path', ''))
        size_str = format_size(size)

        # Model framework
        framework = model.get('framework', 'Unknown')

        table_data.append([
            dataset[:20],
            model_type[:20],
            framework[:10],
            test_acc,
            train_acc,
            size_str,
            timestamp_short
        ])

    headers = ["Dataset", "Model", "Framework", "Test Acc", "Train Acc", "Size", "Saved"]

    if HAS_TABULATE:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        # Simple table format without tabulate
        col_widths = [20, 20, 10, 10, 10, 10, 16]
        header_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
        row_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])

        print(header_fmt.format(*headers))
        print("-" * (sum(col_widths) + len(col_widths) * 3 - 3))
        for row in table_data:
            print(row_fmt.format(*row))


def get_summary_stats(results_dir: Path = None):
    """Get summary statistics about saved models"""
    models = get_all_saved_models(results_dir)

    if not models:
        print("[X] No saved models found.")
        return

    # Calculate statistics
    from collections import Counter

    total_models = len(models)
    datasets = Counter(m['dataset_name'] for m in models)
    model_types = Counter(m['model_type'] for m in models)
    frameworks = Counter(m['framework'] for m in models)
    experiments = Counter(m['experiment'] for m in models)

    total_size = sum(get_model_size(m.get('metadata_path', '')) for m in models)

    print(f"\n📊 Summary Statistics")
    print("=" * 60)
    print(f"Total Models: {total_models}")
    print(f"Total Size: {format_size(total_size)}")
    print(f"\nDatasets ({len(datasets)}):")
    for dataset, count in datasets.most_common():
        print(f"  • {dataset}: {count} models")

    print(f"\nModel Types ({len(model_types)}):")
    for model_type, count in model_types.most_common():
        print(f"  • {model_type}: {count} models")

    print(f"\nFrameworks ({len(frameworks)}):")
    for framework, count in frameworks.most_common():
        print(f"  • {framework}: {count} models")

    print(f"\nExperiments ({len(experiments)}):")
    for exp, count in experiments.most_common(5):  # Top 5
        print(f"  • {exp}: {count} models")
    if len(experiments) > 5:
        print(f"  • ... and {len(experiments) - 5} more experiments")


def search_models(query: str, results_dir: Path = None):
    """Search for models by name, dataset, or model type"""
    models = get_all_saved_models(results_dir)

    if not models:
        print("[X] No saved models found.")
        return

    # Filter models
    query_lower = query.lower()
    filtered = [
        m for m in models
        if query_lower in m.get('dataset_name', '').lower() or
           query_lower in m.get('model_type', '').lower() or
           query_lower in m.get('model_id', '').lower()
    ]

    if not filtered:
        print(f"[X] No models found matching '{query}'")
        return

    print(f"\n[OK] Found {len(filtered)} models matching '{query}'\n")
    display_models_table(filtered)


def main():
    parser = argparse.ArgumentParser(description="Manage saved models")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory path")
    parser.add_argument("--list", action="store_true",
                       help="List all saved models")
    parser.add_argument("--summary", action="store_true",
                       help="Show summary statistics")
    parser.add_argument("--group-by", choices=['dataset', 'experiment'],
                       help="Group models by dataset or experiment")
    parser.add_argument("--search", type=str,
                       help="Search for models by name")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.summary:
        get_summary_stats(results_dir)
    elif args.search:
        search_models(args.search, results_dir)
    elif args.list or args.group_by:
        list_models(results_dir, args.group_by)
    else:
        # Default: show summary
        get_summary_stats(results_dir)
        print("\n" + "=" * 60)
        print("💡 Use --list to see all models")
        print("💡 Use --group-by dataset/experiment to group models")
        print("💡 Use --search <query> to search for specific models")


if __name__ == "__main__":
    main()
