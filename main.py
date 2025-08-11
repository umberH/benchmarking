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
                       help="Output directory for results")
    
    # Run modes
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--run-all", action="store_true",
                       help="Run all combinations of datasets, models, and explanations")
    
    # Hyperparameter tuning options
    parser.add_argument("--tune-hyperparameters", action="store_true",
                       help="Run hyperparameter tuning for all models and datasets")
    parser.add_argument("--use-tuned-params", action="store_true",
                       help="Use tuned hyperparameters in benchmark runs")
    
    # Tuning-specific options
    parser.add_argument("--tuning-output-dir", type=str, default="tuning_results",
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
    
    # Setup logging
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        validate_config(config)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize benchmark
        benchmark = XAIBenchmark(config, output_dir)
        
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
        if args.run_all:
            logger.info("Running all combinations with incremental saving")
            benchmark.run_all(use_tuned_params=args.use_tuned_params)
        elif args.interactive:
            logger.info("Running interactive mode")
            benchmark.run_interactive(use_tuned_params=args.use_tuned_params)
        else:
            logger.info("Running full pipeline")
            benchmark.run_full_pipeline(use_tuned_params=args.use_tuned_params)
        
        logger.info("Benchmarking completed successfully!")
        
        # Print summary
        print(f"\n🎉 Benchmarking completed!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        if args.use_tuned_params:
            print(f"🔧 Used tuned hyperparameters for model training")
        
    except Exception as e:
        logger.error(f"Error during benchmarking: {str(e)}")
        raise


if __name__ == "__main__":
    main() 