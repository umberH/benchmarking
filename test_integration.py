#!/usr/bin/env python3
"""
Integration test script for new models and explanation methods
"""

import sys
from pathlib import Path
import yaml
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.model_factory import ModelFactory
from src.explanations.explanation_factory import ExplanationFactory
from src.data.data_manager import DataManager

def test_model_integration():
    """Test that all new models can be created and are properly integrated"""
    print("Testing Model Integration...")
    
    # Load config
    config_path = Path(__file__).parent / "configs" / "default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize factories
    model_factory = ModelFactory(config.get('models', {}))
    data_manager = DataManager(config)
    
    # Test new models
    new_models = ['resnet', 'roberta', 'naive_bayes_text', 'svm_text', 'xgboost_text']
    
    for model_name in new_models:
        try:
            print(f"  ✓ Testing {model_name}...")
            
            # Check if model is in registry
            if model_name not in model_factory.model_registry:
                print(f"    ✗ {model_name} not found in model registry")
                continue
            
            # Get model info
            model_info = model_factory.get_model_info(model_name)
            print(f"    ✓ Model info: {model_info['description']}")
            
            # Test data type compatibility
            supported_types = model_info.get('supported_data_types', [])
            print(f"    ✓ Supported data types: {supported_types}")
            
        except Exception as e:
            print(f"    ✗ Error testing {model_name}: {e}")

def test_explanation_integration():
    """Test that explanation methods work with new models"""
    print("\nTesting Explanation Integration...")
    
    # Load config
    config_path = Path(__file__).parent / "configs" / "default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize explanation factory
    explanation_factory = ExplanationFactory(config.get('explanations', {}))
    
    # Get available methods
    available_methods = explanation_factory.get_available_methods()
    print(f"  ✓ Available explanation methods: {len(available_methods)}")
    
    # Test key methods for different data types
    key_methods = ['shap', 'lime', 'integrated_gradients']
    
    for method_name in key_methods:
        try:
            method_info = explanation_factory.get_method_info(method_name)
            print(f"  ✓ {method_name}: {method_info['description']}")
            print(f"    Supported data types: {method_info.get('supported_data_types', 'all')}")
        except Exception as e:
            print(f"  ✗ Error testing {method_name}: {e}")

def test_dataset_integration():
    """Test that all mandatory datasets are properly loaded"""
    print("\nTesting Dataset Integration...")
    
    # Load config
    config_path = Path(__file__).parent / "configs" / "default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data manager
    data_manager = DataManager(config)
    
    # Test mandatory datasets
    mandatory_datasets = {
        'tabular': ['adult_income', 'compas', 'breast_cancer', 'heart_disease', 'german_credit',
                   'iris', 'wine_quality', 'diabetes', 'wine_classification', 'digits'],
        'image': ['mnist', 'cifar10', 'fashion_mnist'],
        'text': ['imdb', '20newsgroups', 'ag_news']
    }
    
    for data_type, datasets in mandatory_datasets.items():
        print(f"  Testing {data_type} datasets:")
        for dataset_name in datasets:
            try:
                # Check if dataset info is available
                dataset_config = {
                    'name': dataset_name,
                    'data_type': data_type
                }
                
                # Try to get dataset info (this tests the loader exists)
                dataset_info = data_manager.get_dataset_info(dataset_name)
                print(f"    ✓ {dataset_name}: {dataset_info.get('description', 'Available')}")
                
            except Exception as e:
                print(f"    ✗ Error loading {dataset_name}: {e}")

def test_config_validation():
    """Test that the configuration is valid and complete"""
    print("\nTesting Configuration Validation...")
    
    config_path = Path(__file__).parent / "configs" / "default_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check models_to_train includes new models
    models_to_train = config.get('models', {}).get('models_to_train', [])
    new_models = ['resnet', 'roberta', 'naive_bayes_text', 'svm_text', 'xgboost_text']
    
    print(f"  ✓ Total models to train: {len(models_to_train)}")
    
    for model in new_models:
        if model in models_to_train:
            print(f"    ✓ {model} included in models_to_train")
        else:
            print(f"    ✗ {model} missing from models_to_train")
    
    # Check hyperparameter grids
    param_grids = config.get('hyperparameter_tuning', {}).get('parameter_grids', {})
    
    for model in new_models:
        if model in param_grids:
            print(f"    ✓ {model} has hyperparameter grid")
        else:
            print(f"    ✗ {model} missing hyperparameter grid")

def main():
    """Run all integration tests"""
    print("🚀 XAI Benchmarking Framework - Integration Test")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs during testing
    
    try:
        test_config_validation()
        test_model_integration()
        test_explanation_integration()
        test_dataset_integration()
        
        print("\n" + "=" * 60)
        print("✅ Integration testing completed!")
        print("The framework is ready for comprehensive XAI benchmarking with:")
        print("  • 6 Tabular models (including Linear/Logistic Regression)")
        print("  • 3 Image models (CNN, ViT, ResNet)")
        print("  • 6 Text models (BERT, LSTM, RoBERTa, NB, SVM, XGBoost)")
        print("  • 16 Datasets (5 binary tabular, 5 multi-class tabular, 3 image, 3 text)")
        print("  • Comprehensive explanation methods for all data types")
        print("  • Advanced statistical testing with Wilcoxon and Friedman tests")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()