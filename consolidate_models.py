#!/usr/bin/env python3
"""
Consolidate saved models from multiple experiments into a central location
"""

import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def consolidate_models(results_dir: Path, output_dir: Path, dry_run: bool = False):
    """
    Consolidate all models from experiment directories into a central location

    Args:
        results_dir: Directory containing experiment results
        output_dir: Target directory for consolidated models
        dry_run: If True, only show what would be done without actually copying
    """
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    # Create output directories
    if not dry_run:
        (output_dir / "models").mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    models_found = 0
    models_copied = 0
    models_skipped = 0

    print(f"\n🔍 Scanning for models in: {results_dir}")
    print(f"📁 Target directory: {output_dir}")

    if dry_run:
        print("⚠️  DRY RUN MODE - No files will be copied\n")
    else:
        print("✅ Files will be copied\n")

    # Find all saved_models directories
    for experiment_dir in results_dir.glob("experiment_*/saved_models"):
        if not experiment_dir.is_dir():
            continue

        experiment_name = experiment_dir.parent.name
        print(f"\n📂 Processing experiment: {experiment_name}")

        metadata_dir = experiment_dir / "metadata"
        models_dir = experiment_dir / "models"

        if not metadata_dir.exists():
            print(f"  ⚠️  No metadata directory found, skipping")
            continue

        # Process each model
        for metadata_file in metadata_dir.glob("*_metadata.json"):
            models_found += 1

            try:
                # Read metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                model_id = metadata.get('model_id', 'unknown')
                dataset = metadata.get('dataset_name', 'unknown')
                model_type = metadata.get('model_type', 'unknown')
                model_file = metadata.get('model_file', '')

                # Check if model already exists in target
                target_metadata = output_dir / "metadata" / metadata_file.name
                target_model = output_dir / "models" / model_file

                if target_metadata.exists():
                    # Check if it's the same or newer
                    with open(target_metadata, 'r') as f:
                        existing = json.load(f)

                    existing_time = datetime.fromisoformat(existing.get('saved_timestamp', '1970-01-01'))
                    new_time = datetime.fromisoformat(metadata.get('saved_timestamp', '1970-01-01'))

                    if new_time <= existing_time:
                        print(f"  ⏭️  Skipping {model_id} (older or same version exists)")
                        models_skipped += 1
                        continue
                    else:
                        print(f"  🔄 Updating {model_id} (newer version found)")

                # Copy model file
                source_model = models_dir / model_file
                if source_model.exists():
                    if not dry_run:
                        shutil.copy2(source_model, target_model)
                        shutil.copy2(metadata_file, target_metadata)

                    # Add source experiment to metadata
                    if not dry_run:
                        with open(target_metadata, 'r') as f:
                            updated_metadata = json.load(f)
                        updated_metadata['source_experiment'] = experiment_name
                        with open(target_metadata, 'w') as f:
                            json.dump(updated_metadata, f, indent=2)

                    models_copied += 1
                    print(f"  ✅ {'Would copy' if dry_run else 'Copied'}: {dataset} / {model_type}")
                else:
                    print(f"  ⚠️  Model file not found: {source_model}")

            except Exception as e:
                print(f"  ❌ Error processing {metadata_file.name}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Summary:")
    print(f"  • Models found: {models_found}")
    print(f"  • Models {'would be' if dry_run else ''} copied: {models_copied}")
    print(f"  • Models skipped (duplicates): {models_skipped}")

    if not dry_run:
        print(f"\n✅ Models consolidated to: {output_dir}")
    else:
        print(f"\n💡 Run without --dry-run to actually copy files")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate saved models from multiple experiments"
    )
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="saved_models_consolidated",
                       help="Target directory for consolidated models")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually copying")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    consolidate_models(results_dir, output_dir, args.dry_run)


if __name__ == "__main__":
    main()
