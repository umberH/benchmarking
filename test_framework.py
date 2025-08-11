#!/usr/bin/env python3
"""
Test script for XAI Benchmarking Framework
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logging
from src.utils.config import load_config
from src.data.data_manager import DataManager
from src.models.model_factory import ModelFactory
from src.explanations.explanation_factory import ExplanationFactory
from src.evaluation.evaluator import Evaluator


def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    
    # Load configuration
    config = load_config("configs/default_config.yaml")
    
    # Initialize data manager
    data_manager = DataManager(config)
    
    # Get available datasets
    datasets = data_manager.get_available_datasets()
    print(f"Available datasets: {datasets}")
    
    # Test loading a tabular dataset
    try:
        dataset = data_manager.load_dataset("iris", "tabular")
        print(f"Successfully loaded iris dataset: {dataset.get_info()}")
        return True
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False


def test_model_creation():
    """Test model creation functionality"""
    print("\nTesting model creation...")
    
    # Load configuration
    config = load_config("configs/default_config.yaml")
    
    # Initialize model factory
    model_factory = ModelFactory(config)
    
    # Get available models
    models = model_factory.get_available_models()
    print(f"Available models: {models}")
    
    # Test creating a simple model
    try:
        model = model_factory.create_model("decision_tree", "tabular")
        print(f"Successfully created decision tree model: {model.get_info()}")
        return True
    except Exception as e:
        print(f"Error creating model: {e}")
        return False


def test_explanation_creation():
    """Test explanation method creation"""
    print("\nTesting explanation method creation...")
    
    # Load configuration
    config = load_config("configs/default_config.yaml")
    
    # Initialize explanation factory
    explanation_factory = ExplanationFactory(config)
    
    # Get available methods
    methods = explanation_factory.get_available_methods()
    print(f"Available explanation methods: {methods}")
    
    # Test creating a simple explainer
    try:
        explainer = explanation_factory.create_explainer("shap", "tabular")
        print(f"Successfully created SHAP explainer: {explainer.get_info()}")
        return True
    except Exception as e:
        print(f"Error creating explainer: {e}")
        return False


def test_simple_pipeline():
    """Test a simple end-to-end pipeline"""
    print("\nTesting simple pipeline...")
    
    try:
        # Load configuration
        config = load_config("configs/default_config.yaml")
        
        # Load data
        data_manager = DataManager(config)
        dataset = data_manager.load_dataset("iris", "tabular")
        
        # Create and train model
        model_factory = ModelFactory(config)
        model = model_factory.create_model("decision_tree", "tabular")
        
        # Get data
        X_train, X_test, y_train, y_test = dataset.get_data()
        
        # Train model
        print("Training model...")
        model.train(X_train, y_train)
        
        # Evaluate model
        accuracy = model.evaluate(X_test, y_test)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Create explainer
        explanation_factory = ExplanationFactory(config)
        explainer = explanation_factory.create_explainer("shap", "tabular")
        
        # Generate explanations
        print("Generating explanations...")
        explanations = explainer.explain(X_test[:5], model)
        print(f"Generated explanations for {len(explanations)} samples")
        
        # Evaluate explanations
        evaluator = Evaluator()
        metrics = evaluator.evaluate(explanations, model, X_test[:5], y_test[:5])
        print(f"Explanation metrics: {metrics}")
        
        print("Simple pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in simple pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("XAI Benchmarking Framework - Test Suite")
    print("=" * 50)
    
    # Setup logging
    setup_logging(logging.INFO)
    
    # Run tests
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Explanation Creation", test_explanation_creation),
        ("Simple Pipeline", test_simple_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("All tests passed! The framework is ready to use.")
    else:
        print("Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main() 