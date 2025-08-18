#!/usr/bin/env python3
"""
Test script for the new --comprehensive flag functionality
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.utils.config import load_config
from src.benchmark import XAIBenchmark

def test_comprehensive_functionality():
    """Test the comprehensive benchmarking functionality"""
    print("🧪 Testing comprehensive benchmarking functionality...")
    
    # Load default config
    try:
        config = load_config("configs/default_config.yaml")
        print("✅ Successfully loaded configuration")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Initialize benchmark
            benchmark = XAIBenchmark(config, output_dir)
            print("✅ Successfully initialized benchmark")
            
            # Check if run_comprehensive method exists
            if hasattr(benchmark, 'run_comprehensive'):
                print("✅ run_comprehensive method found")
            else:
                print("❌ run_comprehensive method not found")
                return False
            
            # Check if _generate_comprehensive_markdown_report method exists
            if hasattr(benchmark, '_generate_comprehensive_markdown_report'):
                print("✅ _generate_comprehensive_markdown_report method found")
            else:
                print("❌ _generate_comprehensive_markdown_report method not found")
                return False
            
            # Test markdown generation with dummy data
            dummy_results = [
                {
                    'dataset': 'test_dataset',
                    'dataset_type': 'tabular',
                    'model': 'test_model',
                    'model_type': 'decision_tree',
                    'explanation_method': 'test_explanation',
                    'explanation_type': 'local',
                    'model_performance': {'accuracy': 0.85},
                    'explanation_info': {'n_explanations': 10},
                    'evaluations': {
                        'faithfulness': 0.75,
                        'stability': 0.82,
                        'consistency': 0.68
                    },
                    'used_tuned_params': False,
                    'validation_status': True
                }
            ]
            
            # Test markdown report generation
            benchmark._generate_comprehensive_markdown_report(dummy_results)
            
            # Check if markdown file was created
            markdown_file = output_dir / "comprehensive_report.md"
            if markdown_file.exists():
                print("✅ Markdown report generated successfully")
                
                # Check content
                with open(markdown_file, 'r') as f:
                    content = f.read()
                    if "Comprehensive XAI Benchmarking Report" in content:
                        print("✅ Markdown report contains expected header")
                    else:
                        print("❌ Markdown report missing expected header")
                        return False
                        
                    if "test_dataset" in content and "test_model" in content:
                        print("✅ Markdown report contains test data")
                    else:
                        print("❌ Markdown report missing test data")
                        return False
            else:
                print("❌ Markdown report not generated")
                return False
            
            print("🎉 All tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            return False

def test_main_script_integration():
    """Test that main.py can parse the new flag"""
    print("\n🧪 Testing main.py integration...")
    
    try:
        # Import the parse_args function
        sys.path.append('.')
        from main import parse_args
        
        # Test parsing with comprehensive flag
        test_args = ['--comprehensive', '--output-dir', 'test_output']
        
        # Mock sys.argv for testing
        original_argv = sys.argv
        sys.argv = ['main.py'] + test_args
        
        try:
            args = parse_args()
            if hasattr(args, 'comprehensive') and args.comprehensive:
                print("✅ --comprehensive flag parsed correctly")
                return True
            else:
                print("❌ --comprehensive flag not parsed correctly")
                return False
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"❌ Error testing main.py integration: {e}")
        return False

def show_usage_example():
    """Show usage example for the new flag"""
    print("\n📖 Usage Example:")
    print("="*50)
    print("To run comprehensive benchmarking:")
    print("  python main.py --comprehensive")
    print("")
    print("To run with tuned parameters:")
    print("  python main.py --comprehensive --use-tuned-params")
    print("")
    print("To specify output directory:")
    print("  python main.py --comprehensive --output-dir my_results")
    print("")
    print("This will:")
    print("  • Run all datasets × all models × all explanation methods × all evaluations")
    print("  • Save individual JSON results for each combination")
    print("  • Generate a comprehensive markdown report")
    print("  • Create a results table with each row representing a unique combination")

if __name__ == "__main__":
    print("🎯 Testing Comprehensive Benchmarking Flag")
    print("="*60)
    
    # Run tests
    test1_passed = test_comprehensive_functionality()
    test2_passed = test_main_script_integration()
    
    # Show results
    print(f"\n📊 Test Results:")
    print(f"  Comprehensive functionality: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Main script integration:     {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 All tests passed! The --comprehensive flag is ready to use.")
        show_usage_example()
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)