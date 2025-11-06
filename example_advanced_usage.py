"""
Example: Using Advanced Explanation Methods
Shows how to use the 6 new advanced explanation methods for different data types
"""

import sys
import os
sys.path.append('src')

# Import the explanation factory to test method availability
from explanations.explanation_factory import ExplanationFactory
from utils.method_compatibility import MethodCompatibilityMatrix

def run_advanced_explanations_demo():
    """Demonstrate usage of advanced explanation methods"""
    
    # Test method availability
    config = {'methods': []}
    factory = ExplanationFactory(config)
    compatibility = MethodCompatibilityMatrix()
    
    print("🚀 Advanced XAI Methods Demo")
    print("="*50)
    
    # Test method availability
    print("\n📊 TESTING METHOD AVAILABILITY")
    print("-" * 30)
    
    advanced_methods = [
        'causal_shap',          # Causal structure-aware SHAP
        'shapley_flow',         # Layer-wise value propagation  
        'bayesian_rule_list',   # Interpretable if-then rules
        'influence_functions',  # Training example influence
        'shap_interactive',     # Main + interaction effects
        'corels'               # Optimal rule lists
    ]
    
    print("Testing if advanced methods are properly registered...")
    
    for method in advanced_methods:
        try:
            method_info = factory.get_method_info(method)
            print(f"✅ {method:20} → {method_info['supported_data_types']}")
        except Exception as e:
            print(f"❌ {method:20} → Error: {e}")
    
    print("\n📊 TESTING COMPATIBILITY MATRIX")
    print("-" * 30)
    
    # Test tabular compatibility
    tabular_methods = compatibility.get_compatible_methods('tabular', 'decision_tree')
    advanced_in_tabular = [m for m in advanced_methods if m in tabular_methods]
    print(f"Advanced methods available for tabular data: {advanced_in_tabular}")
    
    # Test image compatibility  
    image_methods = compatibility.get_compatible_methods('image', 'cnn')
    advanced_in_image = [m for m in advanced_methods if m in image_methods]
    print(f"Advanced methods available for image data: {advanced_in_image}")
    
    # Test text compatibility
    text_methods = compatibility.get_compatible_methods('text', 'bert')
    advanced_in_text = [m for m in advanced_methods if m in text_methods]
    print(f"Advanced methods available for text data: {advanced_in_text}")

def show_method_compatibility():
    """Show which methods work with which data types"""
    
    print("\n🔍 METHOD COMPATIBILITY MATRIX")
    print("="*50)
    
    compatibility = {
        'causal_shap': ['tabular'],
        'shapley_flow': ['tabular', 'image'],
        'bayesian_rule_list': ['tabular'],
        'influence_functions': ['tabular', 'image', 'text'],
        'shap_interactive': ['tabular'],
        'corels': ['tabular']
    }
    
    for method, data_types in compatibility.items():
        print(f"📌 {method:20} → {', '.join(data_types)}")
    
    print("\n💡 USAGE TIPS:")
    print("• For tabular data: All 6 methods available")
    print("• For image data: shapley_flow, influence_functions")
    print("• For text data: influence_functions")
    print("• Methods include advanced evaluation metrics automatically")

def show_advanced_metrics():
    """Show the advanced evaluation metrics included"""
    
    print("\n📊 ADVANCED EVALUATION METRICS")
    print("="*50)
    
    axiom_metrics = [
        'advanced_identity',         # Identity axiom
        'advanced_separability',     # Separability axiom  
        'advanced_non_sensitivity',  # Non-sensitivity axiom
        'advanced_stability'       # Compactness axiom
    ]
    
    info_metrics = [
        'advanced_correctness',      # Information gain
        'advanced_entropy',          # Shannon entropy
        'advanced_gini_coefficient', # Gini coefficient
        'advanced_kl_divergence'     # KL divergence
    ]
    
    print("🔬 Axiom-Based Metrics:")
    for metric in axiom_metrics:
        print(f"  • {metric}")
    
    print("\n📈 Information-Theoretic Metrics:")
    for metric in info_metrics:
        print(f"  • {metric}")
    
    print("\n📊 Statistical Analysis:")
    print("  • Confidence intervals")
    print("  • Statistical significance tests (t-tests, Mann-Whitney U)")
    print("  • Effect size calculations (Cohen's d)")
    print("  • Pairwise method comparisons")

if __name__ == "__main__":
    print("🎯 Advanced XAI Methods Usage Guide")
    print("="*60)
    
    # Show compatibility matrix
    show_method_compatibility()
    
    # Show advanced metrics
    show_advanced_metrics()
    
    # Run demo (uncomment to execute)
    print("\n🚀 To run the demo, uncomment the line below:")
    print("# run_advanced_explanations_demo()")
    
    # Uncomment to run actual demo:
    # run_advanced_explanations_demo()