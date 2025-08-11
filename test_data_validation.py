#!/usr/bin/env python3
"""
Test script for data validation functionality
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logging
from src.utils.config import load_config
from src.data.data_manager import DataManager
from src.utils.data_validation import DataValidator, validate_all_datasets


def test_data_validation():
    """Test data validation functionality"""
    print("Testing data validation...")
    
    # Load configuration
    config = load_config("configs/default_config.yaml")
    
    # Initialize data manager
    data_manager = DataManager(config)
    
    # Load datasets
    datasets = data_manager.load_datasets()
    
    if not datasets:
        print("No datasets loaded. Please check your configuration.")
        return False
    
    print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    
    # Initialize validator
    validator = DataValidator(config)
    
    # Validate each dataset
    for dataset_name, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"Validating dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Validate dataset
        validation_result = validator.validate_dataset(dataset, dataset_name)
        
        # Print summary
        status = "✅ VALID" if validation_result['is_valid'] else "❌ INVALID"
        print(f"Status: {status}")
        print(f"Warnings: {len(validation_result['warnings'])}")
        print(f"Errors: {len(validation_result['errors'])}")
        
        # Print warnings
        if validation_result['warnings']:
            print("\n⚠️  Warnings:")
            for warning in validation_result['warnings']:
                print(f"  - {warning}")
        
        # Print errors
        if validation_result['errors']:
            print("\n❌ Errors:")
            for error in validation_result['errors']:
                print(f"  - {error}")
        
        # Print recommendations
        if validation_result['recommendations']:
            print("\n💡 Recommendations:")
            for rec in validation_result['recommendations']:
                print(f"  - {rec}")
        
        # Print key metrics
        details = validation_result['validation_details']
        
        if 'dataset_size' in details:
            size_info = details['dataset_size']
            print(f"\n📊 Dataset Size:")
            print(f"  - Training: {size_info['n_train']} samples")
            print(f"  - Test: {size_info['n_test']} samples")
            print(f"  - Total: {size_info['n_total']} samples")
        
        if 'class_balance' in details:
            balance_info = details['class_balance']
            print(f"\n⚖️  Class Balance:")
            print(f"  - Training imbalance ratio: {balance_info['train_imbalance_ratio']:.3f}")
            print(f"  - Test imbalance ratio: {balance_info['test_imbalance_ratio']:.3f}")
        
        if 'data_quality' in details:
            quality_info = details['data_quality']
            print(f"\n🔍 Data Quality:")
            print(f"  - Training missing ratio: {quality_info['train_missing_ratio']:.3f}")
            print(f"  - Test missing ratio: {quality_info['test_missing_ratio']:.3f}")
        
        if 'outliers' in details:
            outlier_info = details['outliers']
            print(f"\n📈 Outliers:")
            print(f"  - Total outliers: {outlier_info['total_outliers']}")
    
    # Test batch validation
    print(f"\n{'='*60}")
    print("Testing batch validation...")
    print(f"{'='*60}")
    
    validation_results = validate_all_datasets(
        datasets, 
        config, 
        Path("validation_reports")
    )
    
    # Summary
    valid_count = sum(1 for result in validation_results.values() if result['is_valid'])
    total_count = len(validation_results)
    
    print(f"\n📋 Validation Summary:")
    print(f"  - Total datasets: {total_count}")
    print(f"  - Valid datasets: {valid_count}")
    print(f"  - Invalid datasets: {total_count - valid_count}")
    
    if valid_count == total_count:
        print("✅ All datasets passed validation!")
        return True
    else:
        print("⚠️  Some datasets failed validation. Check the reports above.")
        return False


def main():
    """Main test function"""
    print("XAI Benchmarking Framework - Data Validation Test")
    print("=" * 60)
    
    # Setup logging
    setup_logging(logging.INFO)
    
    try:
        success = test_data_validation()
        
        if success:
            print("\n🎉 Data validation test completed successfully!")
        else:
            print("\n⚠️  Data validation test completed with warnings/errors.")
            
    except Exception as e:
        print(f"\n❌ Data validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 