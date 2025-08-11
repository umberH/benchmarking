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
    
    elif args.action == "report":
        print("📋 GENERATING COMPARISON REPORT")
        print("=" * 50)
        
        output_file = Path(args.output) if args.output else None
        report = manager.create_comparison_report(output_file)
        print(report)
    
    elif args.action == "csv":
        if not args.output:
            print("Error: --output is required for CSV export")
            return
        
        print(f"📊 EXPORTING TO CSV: {args.output}")
        print("=" * 50)
        export_iterations_to_csv(results_dir, Path(args.output))
        print(f"Data exported to {args.output}")
    
    elif args.action == "best":
        print(f"🏆 TOP {args.top_k} PERFORMING COMBINATIONS (by {args.metric})")
        print("=" * 50)
        
        best_combinations = manager.get_best_performing_combinations(args.metric, args.top_k)
        
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