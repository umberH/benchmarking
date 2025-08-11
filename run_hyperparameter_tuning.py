#!/usr/bin/env python3
"""
Script to run hyperparameter tuning for XAI benchmarking models
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logging
from src.utils.config import load_config
from src.data.data_manager import DataManager
from src.utils.hyperparameter_tuning import HyperparameterTuner, run_hyperparameter_tuning


def main():
    """Main function for hyperparameter tuning"""
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for XAI benchmarking")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--output-dir", type=str, default="tuning_results",
                       help="Output directory for tuning results")
    parser.add_argument("--datasets", type=str, nargs="+",
                       help="Specific datasets to tune (default: all mandatory)")
    parser.add_argument("--models", type=str, nargs="+",
                       help="Specific models to tune (default: all compatible)")
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--scoring", type=str, default="accuracy",
                       help="Scoring metric for optimization")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Timeout in seconds")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🔧 XAI Benchmarking - Hyperparameter Tuning")
    print("=" * 60)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Update tuning configuration with command line arguments
        if 'hyperparameter_tuning' not in config:
            config['hyperparameter_tuning'] = {}
        
        config['hyperparameter_tuning'].update({
            'cv_folds': args.cv_folds,
            'scoring': args.scoring,
            'n_jobs': args.n_jobs,
            'verbose': args.verbose,
            'timeout': args.timeout
        })
        
        # Initialize data manager
        logger.info("Initializing data manager")
        data_manager = DataManager(config)
        
        # Load datasets
        if args.datasets:
            logger.info(f"Loading specified datasets: {args.datasets}")
            datasets = {}
            for dataset_name in args.datasets:
                try:
                    dataset = data_manager.load_dataset(dataset_name)
                    datasets[dataset_name] = dataset
                except Exception as e:
                    logger.error(f"Failed to load dataset {dataset_name}: {e}")
        else:
            logger.info("Loading all mandatory datasets")
            datasets = data_manager.load_datasets()
        
        if not datasets:
            logger.error("No datasets loaded. Please check your configuration.")
            return
        
        logger.info(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
        
        # Initialize hyperparameter tuner
        logger.info("Initializing hyperparameter tuner")
        tuner = HyperparameterTuner(config, Path(args.output_dir))
        
        # Run tuning
        logger.info("Starting hyperparameter tuning")
        results = tuner.tune_all_models(datasets)
        
        # Generate report
        logger.info("Generating tuning report")
        report = tuner.generate_tuning_report(results)
        
        # Save report
        report_file = Path(args.output_dir) / "tuning_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 HYPERPARAMETER TUNING COMPLETED")
        print("=" * 60)
        
        # Count results
        total_models = sum(len(dataset_results) for dataset_results in results.values())
        successful_tunings = sum(
            1 for dataset_results in results.values() 
            for result in dataset_results.values() 
            if result['success']
        )
        failed_tunings = total_models - successful_tunings
        
        print(f"📊 SUMMARY:")
        print(f"  - Total datasets: {len(datasets)}")
        print(f"  - Total models: {total_models}")
        print(f"  - Successful tunings: {successful_tunings}")
        print(f"  - Failed tunings: {failed_tunings}")
        print(f"  - Success rate: {successful_tunings/total_models:.2%}")
        
        print(f"\n📁 RESULTS SAVED TO:")
        print(f"  - Tuning results: {args.output_dir}/")
        print(f"  - Summary file: {args.output_dir}/tuning_summary.json")
        print(f"  - Report file: {report_file}")
        
        # Show best performers
        if successful_tunings > 0:
            print(f"\n🏆 TOP 3 BEST PERFORMERS:")
            best_performers = []
            
            for dataset_name, dataset_results in results.items():
                for model_name, result in dataset_results.items():
                    if result['success']:
                        best_performers.append({
                            'dataset': dataset_name,
                            'model': model_name,
                            'cv_score': result['best_score'],
                            'test_accuracy': result['test_accuracy']
                        })
            
            # Sort by CV score
            best_performers.sort(key=lambda x: x['cv_score'], reverse=True)
            
            for i, performer in enumerate(best_performers[:3], 1):
                print(f"  {i}. {performer['dataset']} + {performer['model']}")
                print(f"     CV Score: {performer['cv_score']:.4f}")
                print(f"     Test Accuracy: {performer['test_accuracy']:.4f}")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"  1. Use the tuned parameters in your benchmark runs")
        print(f"  2. Check the detailed report: {report_file}")
        print(f"  3. Load parameters using: load_tuned_parameters('{args.output_dir}', dataset, model)")
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 