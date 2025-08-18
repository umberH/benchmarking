"""
Quick test to verify advanced methods are properly configured
"""

import yaml
import os

def test_advanced_methods_in_config():
    """Test if all 6 advanced methods are in the configuration"""
    
    print("Testing Advanced Methods Configuration")
    print("="*50)
    
    # Load config
    config_path = 'configs/default_config.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expected advanced methods
    expected_methods = {
        'causal_shap': 'feature_attribution',
        'shapley_flow': 'feature_attribution', 
        'shap_interactive': 'feature_attribution',
        'influence_functions': 'example_based',
        'bayesian_rule_list': 'concept_based',
        'corels': 'concept_based'
    }
    
    print("\nChecking Method Registration:")
    print("-" * 40)
    
    explanations = config.get('explanations', {})
    found_methods = {}
    
    # Search through all categories
    for category, methods in explanations.items():
        if isinstance(methods, list):
            for method_config in methods:
                if isinstance(method_config, dict) and 'name' in method_config:
                    method_name = method_config['name']
                    found_methods[method_name] = category
    
    # Check each expected method
    for method_name, expected_category in expected_methods.items():
        if method_name in found_methods:
            actual_category = found_methods[method_name]
            if actual_category == expected_category:
                print(f"[OK] {method_name:20} -> {actual_category}")
            else:
                print(f"[WARN] {method_name:20} -> {actual_category} (expected {expected_category})")
        else:
            print(f"[ERROR] {method_name:20} -> NOT FOUND")
    
    print(f"\nSummary: {len([m for m in expected_methods if m in found_methods])}/{len(expected_methods)} methods found")

def show_data_type_compatibility():
    """Show which methods work with which data types"""
    
    print("\nData Type Compatibility Matrix")
    print("="*50)
    
    compatibility = {
        'causal_shap': ['tabular'],
        'shapley_flow': ['tabular', 'image'],
        'bayesian_rule_list': ['tabular'],
        'influence_functions': ['tabular', 'image', 'text'], 
        'shap_interactive': ['tabular'],
        'corels': ['tabular']
    }
    
    print("Method                   | Tabular | Image | Text  |")
    print("-" * 50)
    
    for method, data_types in compatibility.items():
        tabular = "[OK]" if 'tabular' in data_types else "[ - ]"
        image = "[OK]" if 'image' in data_types else "[ - ]"
        text = "[OK]" if 'text' in data_types else "[ - ]"
        
        print(f"{method:24} |  {tabular}   | {image}  | {text}  |")

def show_usage_examples():
    """Show practical usage examples"""
    
    print("\nUsage Examples")
    print("="*50)
    
    print("1. Command Line Usage:")
    print("   python src/benchmark.py --methods causal_shap,shapley_flow --data-type tabular")
    print("")
    
    print("2. For Tabular Data (all 6 methods):")
    print("   --methods causal_shap,shapley_flow,bayesian_rule_list,influence_functions,shap_interactive,corels")
    print("")
    
    print("3. For Image Data (2 methods):")
    print("   --methods shapley_flow,influence_functions --data-type image")
    print("")
    
    print("4. For Text Data (1 method):")
    print("   --methods influence_functions --data-type text")

def show_advanced_evaluation_metrics():
    """Show the advanced evaluation metrics"""
    
    print("\nAdvanced Evaluation Metrics")
    print("="*50)
    
    print("Axiom-Based Metrics:")
    axiom_metrics = [
        ('advanced_identity', 'Identity Axiom - explanations zero when input=baseline'),
        ('advanced_separability', 'Separability - features have distinguishable importance'),
        ('advanced_non_sensitivity', 'Non-sensitivity - stable under small perturbations'),
        ('advanced_compactness', 'Compactness - sparse and focused explanations')
    ]
    
    for metric, description in axiom_metrics:
        print(f"  * {metric:25} - {description}")
    
    print("\nInformation-Theoretic Metrics:")
    info_metrics = [
        ('advanced_correctness', 'Information gain about predictions'),
        ('advanced_entropy', 'Shannon entropy of importance distribution'),
        ('advanced_gini_coefficient', 'Concentration measure (Gini coefficient)'),
        ('advanced_kl_divergence', 'Distance from uniform distribution')
    ]
    
    for metric, description in info_metrics:
        print(f"  * {metric:25} - {description}")
    
    print("\nStatistical Analysis:")
    print("  * Confidence intervals for all metrics")
    print("  * Statistical significance tests (t-test, Mann-Whitney U)")
    print("  * Effect size calculations (Cohen's d)")
    print("  * Pairwise method comparisons")

if __name__ == "__main__":
    test_advanced_methods_in_config()
    show_data_type_compatibility()
    show_usage_examples()
    show_advanced_evaluation_metrics()