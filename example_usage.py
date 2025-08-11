#!/usr/bin/env python3
"""
Example usage of XAI Benchmarking Framework
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logging
from src.utils.config import load_config
from src.benchmark import XAIBenchmark


def example_tabular_benchmark():
    """Example: Benchmark XAI methods on tabular data"""
    print("Running tabular data benchmark...")
    
    # Load configuration
    config = load_config("configs/default_config.yaml")
    
    # Create output directory
    output_dir = Path("results/tabular_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize benchmark
    benchmark = XAIBenchmark(config, output_dir)
    
    # Run full pipeline
    benchmark.run_full_pipeline()


def example_interactive_mode():
    """Example: Interactive mode for exploring the framework"""
    print("Running interactive mode...")
    
    # Load configuration
    config = load_config("configs/default_config.yaml")
    
    # Create output directory
    output_dir = Path("results/interactive_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize benchmark
    benchmark = XAIBenchmark(config, output_dir)
    
    # Run interactive mode
    benchmark.run_interactive()


def example_custom_config():
    """Example: Using a custom configuration"""
    print("Running custom configuration example...")
    
    # Create a custom configuration
    custom_config = {
        "data": {
            "tabular_datasets": [
                {"name": "iris", "source": "sklearn", "description": "Iris dataset"}
            ]
        },
        "models": {
            "tabular": [
                {"name": "decision_tree", "description": "Decision Tree", "library": "sklearn"},
                {"name": "random_forest", "description": "Random Forest", "library": "sklearn"}
            ]
        },
        "explanations": {
            "feature_attribution": [
                {"name": "shap", "description": "SHAP", "library": "shap"},
                {"name": "lime", "description": "LIME", "library": "lime"}
            ]
        },
        "evaluation": {
            "fidelity": [
                {"name": "faithfulness", "description": "Faithfulness metric"}
            ],
            "time_complexity": [
                {"name": "explanation_time", "description": "Explanation generation time"}
            ]
        },
        "experiment": {
            "preprocessing": {"test_size": 0.2, "random_state": 42},
            "training": {"max_iter": 100},
            "explanation": {"num_samples": 50},
            "evaluation": {"num_runs": 3}
        },
        "output": {
            "results_dir": "results/custom_example",
            "save_models": True,
            "save_explanations": True,
            "generate_plots": True
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    # Create output directory
    output_dir = Path("results/custom_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize benchmark with custom config
    benchmark = XAIBenchmark(custom_config, output_dir)
    
    # Run full pipeline
    benchmark.run_full_pipeline()


def main():
    """Main function to run examples"""
    print("XAI Benchmarking Framework - Example Usage")
    print("=" * 50)
    
    # Setup logging
    setup_logging(logging.INFO)
    
    # Run examples
    examples = [
        ("Tabular Benchmark", example_tabular_benchmark),
        ("Interactive Mode", example_interactive_mode),
        ("Custom Configuration", example_custom_config),
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\nSelect an example to run (1-3), or press Enter to run all:")
    choice = input().strip()
    
    if choice == "":
        # Run all examples
        for name, func in examples:
            print(f"\n{'='*20} {name} {'='*20}")
            try:
                func()
                print(f"{name} completed successfully!")
            except Exception as e:
                print(f"Error in {name}: {e}")
                import traceback
                traceback.print_exc()
    elif choice in ["1", "2", "3"]:
        # Run specific example
        idx = int(choice) - 1
        name, func = examples[idx]
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            func()
            print(f"{name} completed successfully!")
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main() 