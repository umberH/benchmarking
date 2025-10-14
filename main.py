#!/usr/bin/env python3
"""
Main entry point for XAI benchmarking framework
"""

import argparse
import logging
import sys
from pathlib import Path

from src.utils.logger import setup_logging
from src.utils.config import load_config, validate_config
from src.benchmark import XAIBenchmark


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="XAI Benchmarking Framework")
    
    # Configuration
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Base output directory for results (experiments will be organized in subdirectories)")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Custom experiment name (will be combined with timestamp)")
    
    # Run modes
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--run-all", action="store_true",
                       help="Run all combinations of datasets, models, and explanations")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive benchmarking across all models, explanations, and evaluations with markdown report generation")
    
    # Hyperparameter tuning options
    parser.add_argument("--tune-hyperparameters", action="store_true",
                       help="Run hyperparameter tuning for all models and datasets")
    parser.add_argument("--auto-tune", action="store_true",
                       help="Automatically tune hyperparameters before training each model (if not already tuned)")
    parser.add_argument("--use-tuned-params", action="store_true",
                       help="Use tuned hyperparameters in benchmark runs")
    parser.add_argument("--train-models-only", action="store_true",
                       help="Only train and save models (optionally with tuning), skip explanations and evaluations")
    
    # Tuning-specific options
    parser.add_argument("--tuning-output-dir", type=str, default="saved_models/tuning_results",
                       help="Output directory for tuning results")
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Number of cross-validation folds for tuning")
    parser.add_argument("--scoring", type=str, default="accuracy",
                       help="Scoring metric for hyperparameter optimization")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Create log file for ALL run modes (not just interactive)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Determine log filename based on run mode
    if args.train_models_only:
        log_filename = f"train_models_{timestamp}.log"
    elif args.interactive:
        log_filename = f"interactive_{timestamp}.log"
    elif args.comprehensive:
        log_filename = f"comprehensive_{timestamp}.log"
    elif args.run_all:
        log_filename = f"run_all_{timestamp}.log"
    elif args.tune_hyperparameters:
        log_filename = f"tuning_{timestamp}.log"
    else:
        log_filename = f"pipeline_{timestamp}.log"

    log_file = str(log_dir / log_filename)

    # Setup logging with file output for ALL modes
    setup_logging(getattr(logging, args.log_level), log_file=log_file)
    logger = logging.getLogger(__name__)

    # Always inform user about log location
    logger.info(f"Log file: {log_file}")
    print(f"[LOG] Output being saved to: {log_file}")
    
    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        validate_config(config)
        
        # Create timestamped experiment directory for separation
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment-specific subdirectory
        experiment_name = f"experiment_{timestamp}"
        if args.experiment_name:
            experiment_name = f"{args.experiment_name}_{timestamp}"
        
        base_output_dir = Path(args.output_dir)
        output_dir = base_output_dir / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Results will be saved to: {output_dir}")
        
        # Initialize benchmark
        benchmark = XAIBenchmark(config, output_dir, auto_tune=args.auto_tune)
        
        # Run hyperparameter tuning if requested
        if args.tune_hyperparameters:
            logger.info("Running hyperparameter tuning")
            
            # Update tuning configuration
            if 'hyperparameter_tuning' not in config:
                config['hyperparameter_tuning'] = {}
            
            config['hyperparameter_tuning'].update({
                'cv_folds': args.cv_folds,
                'scoring': args.scoring
            })
            
            # Run tuning
            tuning_results = benchmark.run_hyperparameter_tuning()
            
            logger.info("Hyperparameter tuning completed")
            logger.info(f"Tuning results saved to {args.tuning_output_dir}")
            
            # Print summary
            if tuning_results:
                successful_tunings = sum(
                    1 for dataset_results in tuning_results.values() 
                    for result in dataset_results.values() 
                    if result['success']
                )
                total_tunings = sum(len(dataset_results) for dataset_results in tuning_results.values())
                
                print(f"\n🎉 Hyperparameter tuning completed!")
                print(f"📊 Summary:")
                print(f"  - Successful tunings: {successful_tunings}/{total_tunings}")
                print(f"  - Success rate: {successful_tunings/total_tunings:.2%}")
                print(f"  - Results saved to: {args.tuning_output_dir}")
                
                if args.use_tuned_params:
                    print(f"\n💡 You can now run benchmarks with tuned parameters using --use-tuned-params")
            
            return
        
        # Run benchmark modes
        if args.train_models_only:
            logger.info("Running TRAIN MODELS ONLY mode")
            benchmark.train_models_only(use_tuned_params=args.use_tuned_params)
        elif args.comprehensive:
            logger.info("Running comprehensive benchmarking with markdown report generation")
            benchmark.run_comprehensive(use_tuned_params=args.use_tuned_params)
        elif args.run_all:
            logger.info("Running all combinations with incremental saving")
            benchmark.run_all(use_tuned_params=args.use_tuned_params)
        elif args.interactive:
            logger.info("Running interactive mode")
            benchmark.run_interactive(use_tuned_params=args.use_tuned_params)
        else:
            logger.info("Running full pipeline")
            benchmark.run_full_pipeline(use_tuned_params=args.use_tuned_params)
        
        logger.info(f"Benchmarking completed successfully! Results saved to: {output_dir}")
        
        # Print summary with experiment organization
        print(f"\n🎉 Benchmarking completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Experiment: {experiment_name}")
        
        if args.comprehensive:
            print(f"📄 Comprehensive markdown report: {output_dir}/comprehensive_report.md")
        
        # Show experiment organization info
        print(f"\n📂 Experiment Organization:")
        print(f"   Base directory: {base_output_dir}")
        print(f"   This experiment: {output_dir}")
        print(f"   💡 Each run creates a separate timestamped folder for easy experiment tracking!")
        
        if args.use_tuned_params:
            print(f"🔧 Used tuned hyperparameters for model training")
        
    except Exception as e:
        logger.error(f"Error during benchmarking: {str(e)}")
        raise


if __name__ == "__main__":
    main() 