#!/usr/bin/env python3
"""
Debug test to see what's happening with explanation generation
"""

import sys
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import traceback

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class MockDataset:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_info(self):
        return {'type': 'tabular'}

def test_basic_explanation_generation():
    print("🔍 Debug Test - Basic Explanation Generation")
    print("=" * 60)
    
    try:
        # Generate data
        X, y = make_classification(n_samples=50, n_features=3, random_state=42, n_informative=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print(f"✅ Data generated: {len(X_train)} train, {len(X_test)} test samples")
        
        # Train model
        model = RandomForestClassifier(n_estimators=3, random_state=42)
        model.fit(X_train, y_train)
        print(f"✅ Model trained, accuracy: {model.score(X_test, y_test):.3f}")
        
        # Create dataset
        dataset = MockDataset(X_train, X_test, y_train, y_test)
        
        # Test each explainer individually
        from explanations.advanced_methods import BayesianRuleListExplainer, CORELSExplainer
        
        explainers = [
            ('BayesianRuleList', BayesianRuleListExplainer),
            ('CORELS', CORELSExplainer)
        ]
        
        for name, explainer_class in explainers:
            print(f"\n🧪 Testing {name}...")
            
            try:
                config = {'name': name.lower(), 'type': name.lower()}
                explainer = explainer_class(config, model, dataset)
                
                # Test with FULL dataset (Fix 2 approach)
                print(f"  📊 Testing with full dataset...")
                result = explainer.explain(dataset)
                
                explanations = result.get('explanations', [])
                print(f"  📈 Generated {len(explanations)} explanations")
                
                if explanations:
                    first_exp = explanations[0]
                    feature_importance = first_exp.get('feature_importance', [])
                    print(f"  🔢 Feature importance: length={len(feature_importance)}")
                    if feature_importance:
                        non_zero = sum(1 for x in feature_importance if abs(x) > 1e-6)
                        print(f"  ✨ Non-zero values: {non_zero}")
                        print(f"  📋 Sample values: {feature_importance[:3]}")
                    else:
                        print("  ⚠️  No feature importance found")
                else:
                    print("  ❌ No explanations generated")
                
                # Test with EMPTY training data (original broken approach)
                print(f"  📊 Testing with empty training data...")
                empty_dataset = MockDataset(np.array([]).reshape(0, 3), X_test[:1], np.array([]), y_test[:1])
                result_empty = explainer.explain(empty_dataset)
                
                explanations_empty = result_empty.get('explanations', [])
                print(f"  📈 Generated {len(explanations_empty)} explanations with empty training")
                
                if explanations_empty:
                    first_exp_empty = explanations_empty[0]
                    feature_importance_empty = first_exp_empty.get('feature_importance', [])
                    print(f"  🔢 Feature importance: length={len(feature_importance_empty)}")
                    if feature_importance_empty:
                        non_zero_empty = sum(1 for x in feature_importance_empty if abs(x) > 1e-6)
                        print(f"  ✨ Non-zero values: {non_zero_empty}")
                    else:
                        print("  ⚠️  No feature importance found")
                
            except Exception as e:
                print(f"  ❌ Error testing {name}: {e}")
                traceback.print_exc()
        
        print(f"\n🎯 Debug Test Complete")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_explanation_generation()