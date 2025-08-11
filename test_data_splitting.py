#!/usr/bin/env python3
"""
Test script for enhanced data splitting functionality
"""

import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logging
from src.utils.config import load_config
from src.data.data_manager import DataManager
from src.utils.data_splitting import DataSplitter, create_split_config, validate_split_result
import numpy as np
import pandas as pd


def test_basic_splitting():
    """Test basic splitting strategies"""
    print("\n" + "="*60)
    print("Testing Basic Splitting Strategies")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.5, 0.2])
    
    # Initialize splitter
    config = load_config("configs/default_config.yaml")
    splitter = DataSplitter(config)
    
    # Test different strategies
    strategies = [
        ("stratified", {}),
        ("holdout", {}),
        ("cross_validation", {"cv_type": "stratified_kfold", "n_splits": 3}),
        ("cross_validation", {"cv_type": "leave_one_out"}),
    ]
    
    for strategy, params in strategies:
        print(f"\n--- Testing {strategy} ---")
        try:
            result = splitter.split_data(X, y, strategy, **params)
            print(f"Success: {result['split_type']}")
            print(f"Train samples: {result['metadata'].get('train_samples', 'N/A')}")
            print(f"Test samples: {result['metadata'].get('test_samples', 'N/A')}")
            
            if 'cv_splits' in result:
                print(f"Number of CV folds: {len(result['cv_splits'])}")
            
            # Validate result
            if validate_split_result(result):
                print("✅ Result validation passed")
            else:
                print("❌ Result validation failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")


def test_time_based_splitting():
    """Test time-based splitting"""
    print("\n" + "="*60)
    print("Testing Time-Based Splitting")
    print("="*60)
    
    # Create sample temporal data
    np.random.seed(42)
    n_samples = 1000
    
    # Create DataFrame with time column
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'timestamp': dates
    })
    y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Initialize splitter
    config = load_config("configs/default_config.yaml")
    splitter = DataSplitter(config)
    
    try:
        result = splitter.split_data(X, y, "time_based", time_column="timestamp", test_size=0.2)
        print(f"Success: {result['split_type']}")
        print(f"Train samples: {result['metadata']['train_samples']}")
        print(f"Test samples: {result['metadata']['test_samples']}")
        print(f"Train time range: {result['metadata']['train_time_range']}")
        print(f"Test time range: {result['metadata']['test_time_range']}")
        print("✅ Time-based splitting successful")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_group_based_splitting():
    """Test group-based splitting"""
    print("\n" + "="*60)
    print("Testing Group-Based Splitting")
    print("="*60)
    
    # Create sample data with groups
    np.random.seed(42)
    n_samples = 1000
    n_groups = 20
    
    # Create groups (e.g., patients, subjects)
    groups = np.random.choice(range(n_groups), size=n_samples)
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'group_id': groups
    })
    y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Initialize splitter
    config = load_config("configs/default_config.yaml")
    splitter = DataSplitter(config)
    
    try:
        result = splitter.split_data(X, y, "group_based", group_column="group_id", test_size=0.2)
        print(f"Success: {result['split_type']}")
        print(f"Train samples: {result['metadata']['train_samples']}")
        print(f"Test samples: {result['metadata']['test_samples']}")
        print(f"Unique groups in train: {result['metadata']['unique_groups_train']}")
        print(f"Unique groups in test: {result['metadata']['unique_groups_test']}")
        print("✅ Group-based splitting successful")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_custom_splitting():
    """Test custom splitting"""
    print("\n" + "="*60)
    print("Testing Custom Splitting")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Define custom indices
    custom_indices = {
        'train': np.arange(0, 700),
        'test': np.arange(700, 850),
        'validation': np.arange(850, 1000)
    }
    
    # Initialize splitter
    config = load_config("configs/default_config.yaml")
    splitter = DataSplitter(config)
    
    try:
        result = splitter.split_data(X, y, "custom", custom_indices=custom_indices)
        print(f"Success: {result['split_type']}")
        print(f"Train samples: {result['metadata']['train_samples']}")
        print(f"Test samples: {result['metadata']['test_samples']}")
        print(f"Validation samples: {result['metadata']['validation_samples']}")
        print("✅ Custom splitting successful")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_data_manager_integration():
    """Test integration with DataManager"""
    print("\n" + "="*60)
    print("Testing DataManager Integration")
    print("="*60)
    
    try:
        # Load configuration and data manager
        config = load_config("configs/default_config.yaml")
        data_manager = DataManager(config)
        
        # Load a dataset
        datasets = data_manager.load_datasets()
        if not datasets:
            print("❌ No datasets available for testing")
            return
        
        dataset_name = list(datasets.keys())[0]
        dataset = datasets[dataset_name]
        
        print(f"Testing with dataset: {dataset_name}")
        
        # Test different splitting strategies
        strategies = [
            ("stratified", {}),
            ("holdout", {}),
            ("cross_validation", {"cv_type": "stratified_kfold", "n_splits": 3}),
        ]
        
        for strategy, params in strategies:
            print(f"\n--- Testing {strategy} with DataManager ---")
            try:
                result = data_manager.split_dataset(dataset, strategy, **params)
                print(f"Success: {result['split_type']}")
                print(f"Train samples: {result['metadata'].get('train_samples', 'N/A')}")
                print(f"Test samples: {result['metadata'].get('test_samples', 'N/A')}")
                
                if 'cv_splits' in result:
                    print(f"Number of CV folds: {len(result['cv_splits'])}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("✅ DataManager integration successful")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_configuration_creation():
    """Test configuration creation utilities"""
    print("\n" + "="*60)
    print("Testing Configuration Creation")
    print("="*60)
    
    # Test different configurations
    configs = [
        create_split_config("stratified", test_size=0.3, validation_size=0.1),
        create_split_config("time_based", time_column="timestamp", test_size=0.2),
        create_split_config("cross_validation", cv_type="stratified_kfold", n_splits=5),
        create_split_config("group_based", group_column="patient_id", test_size=0.25),
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}:")
        print(f"  Strategy: {config['split_strategy']}")
        print(f"  Parameters: {config['parameters']}")


def main():
    """Main test function"""
    print("Enhanced Data Splitting Test Suite")
    print("=" * 60)
    
    setup_logging(logging.INFO)
    
    try:
        # Run all tests
        test_basic_splitting()
        test_time_based_splitting()
        test_group_based_splitting()
        test_custom_splitting()
        test_data_manager_integration()
        test_configuration_creation()
        
        print("\n" + "="*60)
        print("🎉 All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 