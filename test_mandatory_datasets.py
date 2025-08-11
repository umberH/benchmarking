#!/usr/bin/env python3
"""
Test script for mandatory datasets in XAI Benchmarking Framework
"""
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logging
from src.utils.config import load_config
from src.data.data_manager import DataManager

def test_mandatory_datasets():
    """Test loading of mandatory datasets"""
    print("Testing mandatory datasets...")
    
    # Setup logging
    setup_logging(logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config('configs/default_config.yaml')
        
        # Initialize data manager
        data_manager = DataManager(config)
        
        # Test loading all datasets (including mandatory ones)
        logger.info("Loading all datasets...")
        datasets = data_manager.load_datasets()
        
        print(f"\nLoaded {len(datasets)} datasets:")
        for name, dataset in datasets.items():
            print(f"  - {name}: {type(dataset).__name__}")
            
            # Print dataset info
            if hasattr(dataset, 'X_train'):
                print(f"    Train samples: {len(dataset.X_train)}")
                print(f"    Test samples: {len(dataset.X_test)}")
                if hasattr(dataset, 'feature_names'):
                    print(f"    Features: {len(dataset.feature_names)}")
                print(f"    Target classes: {len(set(dataset.y_train))}")
        
        # Check if mandatory datasets are loaded
        mandatory_datasets = ['adult_income', 'compas', 'mnist', 'imdb']
        missing_datasets = []
        
        for dataset_name in mandatory_datasets:
            if dataset_name not in datasets:
                missing_datasets.append(dataset_name)
        
        if missing_datasets:
            print(f"\n⚠️  Missing mandatory datasets: {missing_datasets}")
        else:
            print(f"\n✅ All mandatory datasets loaded successfully!")
        
        # Test individual dataset loading
        print("\nTesting individual dataset loading...")
        for dataset_name in mandatory_datasets:
            try:
                dataset = data_manager.load_dataset(dataset_name)
                print(f"  ✅ {dataset_name}: Loaded successfully")
            except Exception as e:
                print(f"  ❌ {dataset_name}: Failed to load - {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing mandatory datasets: {e}")
        return False

def test_dataset_creation():
    """Test individual dataset loading"""
    print("\nTesting individual dataset loading...")
    
    try:
        # Load configuration
        config = load_config('configs/default_config.yaml')
        data_manager = DataManager(config)
        
        # Test individual dataset loading
        mandatory_datasets = ['adult_income', 'compas', 'mnist', 'imdb']
        
        for dataset_name in mandatory_datasets:
            try:
                # Retrieve the actual config for the mandatory dataset
                dataset_config = data_manager._get_mandatory_datasets().get(dataset_name)
                if dataset_config is None:
                    print(f"  ❌ {dataset_name}: Configuration not found for mandatory dataset.")
                    continue
                
                dataset = data_manager._load_mandatory_dataset(dataset_name, dataset_config)
                print(f"  ✅ {dataset_name}: Dataset loaded successfully")
                if hasattr(dataset, 'X_train'): # Check for tabular/image datasets
                    print(f"    Train samples: {len(dataset.X_train)}")
                    print(f"    Test samples: {len(dataset.X_test)}")
                elif hasattr(dataset, 'train_texts'): # Check for text datasets
                    print(f"    Train samples: {len(dataset.train_texts)}")
                    print(f"    Test samples: {len(dataset.test_texts)}")
            except Exception as e:
                print(f"  ❌ {dataset_name}: Failed to load dataset - {e}")
        
        return True
        
    except Exception as e:
        print(f"Error testing dataset loading: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("MANDATORY DATASETS TEST")
    print("=" * 60)
    
    # Test mandatory datasets
    success1 = test_mandatory_datasets()
    
    # Test synthetic dataset creation
    success2 = test_dataset_creation()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 