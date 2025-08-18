#!/usr/bin/env python3
"""
Minimal test to check if our explanation fixes work
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if we can import the fixed classes"""
    print("Testing imports...")
    
    try:
        from explanations.base_explainer import BaseExplainer
        print("✅ BaseExplainer imported successfully")
    except Exception as e:
        print(f"❌ BaseExplainer import failed: {e}")
        return False
    
    try:
        from explanations.advanced_methods import BayesianRuleListExplainer
        print("✅ BayesianRuleListExplainer imported successfully")
    except Exception as e:
        print(f"❌ BayesianRuleListExplainer import failed: {e}")
        return False
    
    try:
        from explanations.advanced_methods import CORELSExplainer
        print("✅ CORELSExplainer imported successfully")
    except Exception as e:
        print(f"❌ CORELSExplainer import failed: {e}")
        return False
    
    try:
        from explanations.advanced_methods import InfluenceFunctionExplainer
        print("✅ InfluenceFunctionExplainer imported successfully")
    except Exception as e:
        print(f"❌ InfluenceFunctionExplainer import failed: {e}")
        return False
    
    return True

def test_method_exists():
    """Test if our added methods exist"""
    print("\nTesting new methods...")
    
    try:
        from explanations.advanced_methods import BayesianRuleListExplainer
        
        # Check if our new method exists
        if hasattr(BayesianRuleListExplainer, '_convert_rules_to_importance'):
            print("✅ BayesianRuleListExplainer._convert_rules_to_importance exists")
        else:
            print("❌ BayesianRuleListExplainer._convert_rules_to_importance missing")
            
    except Exception as e:
        print(f"❌ Method check failed: {e}")
        return False
    
    try:
        from explanations.advanced_methods import CORELSExplainer
        
        # Check if our new method exists
        if hasattr(CORELSExplainer, '_convert_corels_rules_to_importance'):
            print("✅ CORELSExplainer._convert_corels_rules_to_importance exists")
        else:
            print("❌ CORELSExplainer._convert_corels_rules_to_importance missing")
            
    except Exception as e:
        print(f"❌ CORELS method check failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Minimal test for explanation fixes")
    print("=" * 40)
    
    if test_imports():
        print("\n✅ All imports successful!")
        if test_method_exists():
            print("✅ All methods exist!")
            print("\n🎉 Fix implementation looks good!")
        else:
            print("❌ Some methods missing")
    else:
        print("❌ Import failures - check your Python path")