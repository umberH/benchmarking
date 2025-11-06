#!/usr/bin/env python3
"""
List all saved models grouped by dataset and model type
Shows which models will be loaded from cache in interactive mode
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def format_size(bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def main():
    models_dir = Path("saved_models/models")
    metadata_dir = Path("saved_models/metadata")

    if not metadata_dir.exists():
        print("No saved models found!")
        return

    # Group by dataset
    models_by_dataset = defaultdict(lambda: defaultdict(list))
    total_size = 0
    total_count = 0

    for metadata_file in sorted(metadata_dir.glob("*_metadata.json")):
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)

            dataset = data.get('dataset_name', 'unknown')
            model_type = data.get('model_type', 'unknown')
            model_id = data.get('model_id', 'unknown')
            test_acc = data.get('performance_metrics', {}).get('test_accuracy', 'N/A')
            training_time = data.get('performance_metrics', {}).get('training_time', 0)
            saved_time = data.get('saved_timestamp', 'Unknown')
            framework = data.get('framework', 'unknown')
            model_file = data.get('model_file', '')

            # Get model file size
            model_path = models_dir / model_file
            if model_path.exists():
                size = model_path.stat().st_size
                total_size += size
            else:
                size = 0

            models_by_dataset[dataset][model_type].append({
                'model_id': model_id,
                'test_acc': test_acc,
                'training_time': training_time,
                'saved_time': saved_time,
                'framework': framework,
                'size': size,
                'model_file': model_file
            })
            total_count += 1

        except Exception as e:
            print(f"Error reading {metadata_file}: {e}")

    # Display results
    print("=" * 100)
    print("ALL SAVED MODELS - Will be loaded automatically in --interactive and --comprehensive modes")
    print("=" * 100)
    print(f"\nTotal: {total_count} models | Total Size: {format_size(total_size)}\n")

    # Group by data type
    tabular_models = {}
    image_models = {}
    text_models = {}

    for dataset in sorted(models_by_dataset.keys()):
        # Categorize by dataset type
        if any(x in dataset for x in ['mnist', 'cifar', 'fashion']):
            image_models[dataset] = models_by_dataset[dataset]
        elif any(x in dataset for x in ['imdb', 'news', 'ag_']):
            text_models[dataset] = models_by_dataset[dataset]
        else:
            tabular_models[dataset] = models_by_dataset[dataset]

    # Display by category
    categories = [
        ("TABULAR DATASETS", tabular_models),
        ("IMAGE DATASETS", image_models),
        ("TEXT DATASETS", text_models)
    ]

    for category_name, category_models in categories:
        if not category_models:
            continue

        print(f"\n{'='*100}")
        print(f"{category_name}")
        print(f"{'='*100}")

        for dataset in sorted(category_models.keys()):
            models = category_models[dataset]
            print(f"\n[Dataset: {dataset}] ({len(models)} models)")
            print("-" * 100)

            for model_type in sorted(models.keys()):
                model_list = models[model_type]
                # Use the latest model if multiple exist
                model_info = model_list[-1]

                # Format training time
                train_time = model_info['training_time']
                if train_time > 3600:
                    time_str = f"{train_time/3600:.1f}h"
                elif train_time > 60:
                    time_str = f"{train_time/60:.1f}m"
                else:
                    time_str = f"{train_time:.1f}s"

                # Format test accuracy
                acc = model_info['test_acc']
                if isinstance(acc, (int, float)):
                    acc_str = f"{acc:.4f}"
                else:
                    acc_str = str(acc)

                print(f"  [{model_type:20s}] | Acc: {acc_str:8s} | Size: {format_size(model_info['size']):10s} | Train: {time_str:8s} | {model_info['framework']:8s}")

                if len(model_list) > 1:
                    print(f"      [!] {len(model_list)} versions found (showing latest)")

    # Summary statistics
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}")

    tabular_count = sum(len(models) for models in tabular_models.values())
    image_count = sum(len(models) for models in image_models.values())
    text_count = sum(len(models) for models in text_models.values())

    print(f"Tabular Models: {tabular_count}")
    print(f"Image Models  : {image_count}")
    print(f"Text Models   : {text_count}")
    print(f"Total Models  : {total_count}")
    print(f"Total Size    : {format_size(total_size)}")

    # Time saved estimate
    total_training_time = 0
    for dataset_models in models_by_dataset.values():
        for model_list in dataset_models.values():
            total_training_time += model_list[-1]['training_time']

    print(f"\nEstimated training time saved by using cached models: {total_training_time/3600:.1f} hours")

    print(f"\n{'='*100}")
    print("To use these models automatically:")
    print("   python main.py --interactive")
    print("   python main.py --comprehensive")
    print("   python main.py --comprehensive --use-tuned-params  (for tuned hyperparameters)")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
