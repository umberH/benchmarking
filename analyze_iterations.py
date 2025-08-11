#!/usr/bin/env python3
"""
Script to analyze and manage iteration results
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.iteration_manager import IterationManager, analyze_iterations, export_iterations_to_csv


def main():
    """Main function for iteration analysis"""
    parser = argparse.ArgumentParser(description="Analyze XAI benchmarking iteration results")
    parser.add_argument("--results-dir", type=str, default="results/iterations",
                       help="Directory containing iteration results")
    parser.add_argument("--action", type=str, choices=["summary", "report", "csv", "best", "failed", "cleanup"],
                       default="summary", help="Action to perform")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--metric", type=str, default="faithfulness",
                       help="Metric to use for ranking (for 'best' action)")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top results to show (for 'best' action)")
    parser.add_argument("--days-old", type=int, default=30,
                       help="Remove files older than this many days (for 'cleanup' action)")
    
    args = parser.parse_args()
    
    # Initialize iteration manager
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        return
    
    manager = IterationManager(results_dir)
    
    # Perform requested action
    if args.action == "summary":
        print("📊 ITERATION SUMMARY")
        print("=" * 50)
        summary = manager.get_iteration_summary()
        
        print(f"Total iterations: {summary['total_iterations']}")
        print(f"Datasets: {', '.join(summary['datasets'])}")
        print(f"Models: {', '.join(summary['models'])}")
        print(f"Methods: {', '.join(summary['methods'])}")
        
        val_stats = summary['validation_stats']
        print(f"\nValidation Statistics:")
        print(f"  Valid: {val_stats['valid_count']}")
        print(f"  Invalid: {val_stats['invalid_count']}")
        print(f"  Success rate: {val_stats['valid_ratio']:.2%}")
        
        if 'date_range' in summary:
            print(f"\nDate Range:")
            print(f"  Earliest: {summary['date_range']['earliest']}")
            print(f"  Latest: {summary['date_range']['latest']}")
    
    if args.action == "summary":
        def pretty_float(val):
            try:
                import numpy as np
                if isinstance(val, float) or (hasattr(np, 'floating') and isinstance(val, np.floating)):
                    return f"{float(val):.4f}"
            except Exception:
                pass
            return str(val)

        def print_section(title):
            print(f"\n{'='*10} {title} {'='*10}")

        print("� ITERATION SUMMARY")
        print("=" * 50)
        summary = manager.get_iteration_summary()
        print(f"Total iterations: {summary['total_iterations']}")
        print(f"Datasets: {', '.join(summary['datasets'])}")
        print(f"Models: {', '.join(summary['models'])}")
        print(f"Methods: {', '.join(summary['methods'])}")
        val_stats = summary['validation_stats']
        print(f"\nValidation Statistics:")
        print(f"  Valid: {val_stats['valid_count']}")
        print(f"  Invalid: {val_stats['invalid_count']}")
        print(f"  Success rate: {val_stats['valid_ratio']:.2%}")
        if 'date_range' in summary:
            print(f"\nDate Range:")
            print(f"  Earliest: {summary['date_range']['earliest']}")
            print(f"  Latest:   {summary['date_range']['latest']}")
        print("\nSample Iteration Details:")
        for it in summary.get('sample_iterations', [])[:3]:
            print(f"- {it['filename']}: {it['dataset']} | {it['model']} | {it['explanation_method']} | Valid: {it['validation_status']}")

        # Pretty-print all iterations' model and explanation evaluation stats
        print("\n\n� ALL ITERATION EVALUATION STATS")
        print("=" * 50)
        all_iterations = manager.get_all_iterations()
        for idx, it in enumerate(all_iterations, 1):
            print_section(f"Iteration {idx}: {it.get('filename', '')}")
            rd = it.get('result_data', {})
            print(f"Dataset: {rd.get('dataset', '')}")
            print(f"Model: {rd.get('model', '')}")
            print(f"Explanation: {rd.get('explanation_method', '')}")
            print(f"Valid: {rd.get('validation_status', '')}")
            # Model performance
            if 'model_performance' in rd:
                print_section("Model Performance")
                perf = rd['model_performance']
                if isinstance(perf, dict):
                    for k, v in perf.items():
                        print(f"{k:20}: {pretty_float(v)}")
                else:
                    print(perf)
            # Explanation evaluation
            if 'evaluation_results' in rd:
                print_section("Explanation Evaluation Results")
                evals = rd['evaluation_results']
                if isinstance(evals, dict):
                    for k, v in evals.items():
                        print(f"{k:20}: {pretty_float(v)}")
                else:
                    print(evals)
            print("\n" + "-"*40)
        if not best_combinations:
            print("No results found!")
            return
        
        for i, combo in enumerate(best_combinations, 1):
            print(f"{i}. {combo['dataset']} + {combo['model']} + {combo['explanation_method']}")
            print(f"   Score: {combo['score']:.4f}")
            print(f"   Key: {combo['iteration_key']}")
            print()
    
    elif args.action == "failed":
        print("❌ FAILED ITERATIONS")
        print("=" * 50)
        
        failed_iterations = manager.get_failed_iterations()
        
        if not failed_iterations:
            print("No failed iterations found!")
            return
        
        for iteration in failed_iterations:
            result_data = iteration.get('result_data', {})
            iteration_info = iteration.get('iteration_info', {})
            
            print(f"Iteration {iteration_info.get('iteration_number')}:")
            print(f"  Dataset: {result_data.get('dataset')}")
            print(f"  Model: {result_data.get('model')}")
            print(f"  Method: {result_data.get('explanation_method')}")
            print(f"  Timestamp: {iteration_info.get('timestamp')}")
            print(f"  Validation errors: {result_data.get('validation_errors', [])}")
            print()
    
    elif args.action == "cleanup":
        print(f"🧹 CLEANING UP FILES OLDER THAN {args.days_old} DAYS")
        print("=" * 50)
        
        # Show what will be removed
        iteration_files = manager.get_iteration_files()
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=args.days_old)
        
        files_to_remove = []
        for iteration_file in iteration_files:
            try:
                mtime = datetime.fromtimestamp(iteration_file.stat().st_mtime)
                if mtime < cutoff_date:
                    files_to_remove.append(iteration_file)
            except Exception:
                pass
        
        if not files_to_remove:
            print("No old files to remove!")
            return
        
        print(f"Found {len(files_to_remove)} files to remove:")
        for file in files_to_remove:
            print(f"  - {file.name}")
        
        # Ask for confirmation
        response = input("\nProceed with cleanup? (y/N): ")
        if response.lower() == 'y':
            manager.cleanup_old_iterations(args.days_old)
            print("Cleanup completed!")
        else:
            print("Cleanup cancelled.")


if __name__ == "__main__":
    main() 