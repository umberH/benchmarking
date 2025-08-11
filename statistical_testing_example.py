#!/usr/bin/env python3
"""
Statistical Testing Example for XAI Benchmarking Framework
"""
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logging
from src.utils.config import load_config
from src.evaluation.statistical_tests import StatisticalTester


def generate_mock_results():
    """
    Generate mock benchmark results for demonstration
    """
    np.random.seed(42)
    
    # Mock results with different performance patterns
    mock_results = {
        'shap_method': {
            'metrics': {
                'faithfulness': np.random.normal(0.85, 0.1, 20),
                'monotonicity': np.random.normal(0.78, 0.08, 20),
                'completeness': np.random.normal(0.92, 0.05, 20),
                'explanation_time': np.random.normal(2.5, 0.3, 20),
                'stability': np.random.normal(0.88, 0.06, 20)
            }
        },
        'lime_method': {
            'metrics': {
                'faithfulness': np.random.normal(0.72, 0.12, 20),
                'monotonicity': np.random.normal(0.65, 0.10, 20),
                'completeness': np.random.normal(0.85, 0.08, 20),
                'explanation_time': np.random.normal(1.8, 0.4, 20),
                'stability': np.random.normal(0.75, 0.09, 20)
            }
        },
        'integrated_gradients': {
            'metrics': {
                'faithfulness': np.random.normal(0.91, 0.07, 20),
                'monotonicity': np.random.normal(0.89, 0.05, 20),
                'completeness': np.random.normal(0.94, 0.04, 20),
                'explanation_time': np.random.normal(3.2, 0.5, 20),
                'stability': np.random.normal(0.92, 0.04, 20)
            }
        },
        'prototype_method': {
            'metrics': {
                'faithfulness': np.random.normal(0.68, 0.15, 20),
                'monotonicity': np.random.normal(0.71, 0.11, 20),
                'completeness': np.random.normal(0.79, 0.09, 20),
                'explanation_time': np.random.normal(1.2, 0.2, 20),
                'stability': np.random.normal(0.82, 0.07, 20)
            }
        }
    }
    
    return mock_results


def demonstrate_statistical_tests():
    """
    Demonstrate comprehensive statistical testing capabilities
    """
    print("Statistical Testing Demonstration for XAI Benchmarking")
    print("=" * 60)
    
    # Load configuration
    config = load_config("configs/default_config.yaml")
    
    # Generate mock results
    mock_results = generate_mock_results()
    
    # Initialize statistical tester
    statistical_tester = StatisticalTester(config)
    
    # Run comprehensive statistical tests
    print("\n1. Running Statistical Tests...")
    statistical_results = statistical_tester.run_statistical_tests(mock_results)
    
    # Display results
    print("\n2. Statistical Test Results")
    print("-" * 40)
    
    # Summary
    summary = statistical_results['summary']
    print(f"Total tests performed: {summary['total_tests']}")
    print(f"Significant tests: {summary['significant_tests']}")
    print(f"Significance rate: {summary['significance_rate']:.2%}")
    
    if summary['significant_pairs']:
        print(f"\nSignificant differences found:")
        for pair in summary['significant_pairs'][:5]:  # Show first 5
            print(f"  - {pair['metric']}: {pair['pair']} (p={pair['p_value']:.4f})")
    
    if summary['recommendations']:
        print(f"\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
    
    # Detailed pairwise comparisons
    print("\n3. Detailed Pairwise Comparisons")
    print("-" * 40)
    
    for metric_name, pairwise_results in statistical_results['pairwise_comparisons'].items():
        print(f"\nMetric: {metric_name}")
        if 't_test' in pairwise_results:
            for pair_name, test_result in pairwise_results['t_test'].items():
                if 'error' not in test_result:
                    significance = "***" if test_result['significant'] else ""
                    print(f"  {pair_name}: t={test_result['statistic']:.3f}, "
                          f"p={test_result['p_value']:.4f} {significance}")
    
    # Effect sizes
    print("\n4. Effect Sizes")
    print("-" * 40)
    
    for metric_name, effect_sizes in statistical_results['effect_sizes'].items():
        print(f"\nMetric: {metric_name}")
        if 'cohens_d' in effect_sizes:
            for pair_name, cohens_d in effect_sizes['cohens_d'].items():
                if not isinstance(cohens_d, dict):  # Skip error entries
                    effect_magnitude = "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"
                    print(f"  {pair_name}: Cohen's d = {cohens_d:.3f} ({effect_magnitude})")
    
    # Power analysis
    print("\n5. Power Analysis")
    print("-" * 40)
    
    for metric_name, power_results in statistical_results['power_analysis'].items():
        print(f"\nMetric: {metric_name}")
        if 'power_estimates' in power_results:
            for pair_name, power_est in power_results['power_estimates'].items():
                if 'error' not in power_est:
                    power_status = "Adequate" if power_est['adequate_power'] else "Inadequate"
                    print(f"  {pair_name}: Power = {power_est['power']:.3f} ({power_status})")
    
    # Correlation analysis
    print("\n6. Metric Correlations")
    print("-" * 40)
    
    correlation_results = statistical_results['correlation_analysis']
    if 'metric_correlations' in correlation_results:
        for pair_name, corr_result in correlation_results['metric_correlations'].items():
            if 'error' not in corr_result:
                pearson_sig = "***" if corr_result['pearson_p'] < 0.05 else ""
                spearman_sig = "***" if corr_result['spearman_p'] < 0.05 else ""
                print(f"  {pair_name}:")
                print(f"    Pearson: r={corr_result['pearson_r']:.3f}, p={corr_result['pearson_p']:.4f} {pearson_sig}")
                print(f"    Spearman: r={corr_result['spearman_r']:.3f}, p={corr_result['spearman_p']:.4f} {spearman_sig}")
    
    return statistical_results


def create_statistical_visualizations(statistical_results):
    """
    Create visualizations for statistical test results
    """
    print("\n7. Creating Statistical Visualizations")
    print("-" * 40)
    
    # Create output directory for plots
    plots_dir = Path("results/statistical_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. P-value heatmap for pairwise comparisons
    print("Creating p-value heatmap...")
    create_pvalue_heatmap(statistical_results, plots_dir)
    
    # 2. Effect size visualization
    print("Creating effect size plot...")
    create_effect_size_plot(statistical_results, plots_dir)
    
    # 3. Power analysis plot
    print("Creating power analysis plot...")
    create_power_plot(statistical_results, plots_dir)
    
    # 4. Correlation matrix
    print("Creating correlation matrix...")
    create_correlation_matrix(statistical_results, plots_dir)
    
    print(f"Visualizations saved to: {plots_dir}")


def create_pvalue_heatmap(statistical_results, plots_dir):
    """Create heatmap of p-values for pairwise comparisons"""
    metrics = list(statistical_results['pairwise_comparisons'].keys())
    methods = ['shap_method', 'lime_method', 'integrated_gradients', 'prototype_method']
    
    # Create p-value matrix
    pvalue_matrix = np.full((len(methods), len(methods)), np.nan)
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i != j:
                pair_name = f"{method1}_vs_{method2}"
                # Get p-value from first metric (faithfulness)
                if 'faithfulness' in statistical_results['pairwise_comparisons']:
                    t_test_results = statistical_results['pairwise_comparisons']['faithfulness']['t_test']
                    if pair_name in t_test_results and 'error' not in t_test_results[pair_name]:
                        pvalue_matrix[i, j] = t_test_results[pair_name]['p_value']
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    mask = np.isnan(pvalue_matrix)
    sns.heatmap(pvalue_matrix, 
                annot=True, 
                fmt='.3f', 
                mask=mask,
                cmap='RdYlBu_r',
                center=0.05,
                cbar_kws={'label': 'P-value'},
                xticklabels=methods,
                yticklabels=methods)
    plt.title('P-values for Pairwise Comparisons (Faithfulness Metric)')
    plt.tight_layout()
    plt.savefig(plots_dir / 'pvalue_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_effect_size_plot(statistical_results, plots_dir):
    """Create effect size visualization"""
    metrics = list(statistical_results['effect_sizes'].keys())
    methods = ['shap_method', 'lime_method', 'integrated_gradients', 'prototype_method']
    
    # Extract Cohen's d values
    effect_data = []
    for metric in metrics[:3]:  # First 3 metrics
        if 'cohens_d' in statistical_results['effect_sizes'][metric]:
            for pair_name, cohens_d in statistical_results['effect_sizes'][metric]['cohens_d'].items():
                if not isinstance(cohens_d, dict):
                    effect_data.append({
                        'metric': metric,
                        'pair': pair_name,
                        'cohens_d': cohens_d
                    })
    
    if effect_data:
        df = pd.DataFrame(effect_data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='pair', y='cohens_d', hue='metric')
        plt.title('Effect Sizes (Cohen\'s d) for Method Comparisons')
        plt.xlabel('Method Pair')
        plt.ylabel('Cohen\'s d')
        plt.xticks(rotation=45)
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.savefig(plots_dir / 'effect_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_power_plot(statistical_results, plots_dir):
    """Create power analysis visualization"""
    power_data = []
    for metric_name, power_results in statistical_results['power_analysis'].items():
        if 'power_estimates' in power_results:
            for pair_name, power_est in power_results['power_estimates'].items():
                if 'error' not in power_est:
                    power_data.append({
                        'metric': metric_name,
                        'pair': pair_name,
                        'power': power_est['power'],
                        'adequate': power_est['adequate_power']
                    })
    
    if power_data:
        df = pd.DataFrame(power_data)
        
        plt.figure(figsize=(12, 6))
        colors = ['red' if not adequate else 'green' for adequate in df['adequate']]
        bars = plt.bar(range(len(df)), df['power'], color=colors, alpha=0.7)
        plt.axhline(y=0.8, color='black', linestyle='--', label='Adequate Power (0.8)')
        plt.title('Statistical Power for Method Comparisons')
        plt.xlabel('Method Pair')
        plt.ylabel('Power')
        plt.xticks(range(len(df)), df['pair'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'power_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_correlation_matrix(statistical_results, plots_dir):
    """Create correlation matrix visualization"""
    correlation_results = statistical_results['correlation_analysis']
    
    if 'metric_correlations' in correlation_results:
        # Extract correlation data
        metrics = ['faithfulness', 'monotonicity', 'completeness', 'explanation_time', 'stability']
        corr_matrix = np.zeros((len(metrics), len(metrics)))
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    pair_name = f"{metric1}_vs_{metric2}"
                    if pair_name in correlation_results['metric_correlations']:
                        corr_result = correlation_results['metric_correlations'][pair_name]
                        if 'error' not in corr_result:
                            corr_matrix[i, j] = corr_result['pearson_r']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    fmt='.3f',
                    cmap='RdBu_r',
                    center=0,
                    xticklabels=metrics,
                    yticklabels=metrics)
        plt.title('Correlation Matrix Between Metrics')
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()


def demonstrate_multiple_comparison_correction():
    """
    Demonstrate multiple comparison correction methods
    """
    print("\n8. Multiple Comparison Correction Demonstration")
    print("-" * 50)
    
    # Generate multiple p-values
    np.random.seed(42)
    p_values = np.random.uniform(0, 1, 20)
    
    # Apply different correction methods
    config = load_config("configs/default_config.yaml")
    statistical_tester = StatisticalTester(config)
    
    correction_methods = ['bonferroni', 'holm', 'fdr_bh', 'fdr_by']
    
    print("Original p-values (first 10):")
    print(p_values[:10])
    print()
    
    for method in correction_methods:
        try:
            corrected_p_values, significant = statistical_tester.apply_multiple_comparison_correction(
                p_values.tolist(), method
            )
            print(f"{method.upper()} correction:")
            print(f"  Significant tests: {sum(significant)}/{len(significant)}")
            print(f"  Corrected p-values (first 5): {corrected_p_values[:5]}")
            print()


def main():
    """Main function to run statistical testing demonstration"""
    # Setup logging
    setup_logging(logging.INFO)
    
    print("XAI Benchmarking Framework - Statistical Testing Demonstration")
    print("=" * 70)
    
    try:
        # Run comprehensive statistical tests
        statistical_results = demonstrate_statistical_tests()
        
        # Create visualizations
        create_statistical_visualizations(statistical_results)
        
        # Demonstrate multiple comparison correction
        demonstrate_multiple_comparison_correction()
        
        print("\n" + "=" * 70)
        print("Statistical testing demonstration completed successfully!")
        print("Check the 'results/statistical_plots' directory for visualizations.")
        
    except Exception as e:
        print(f"Error during statistical testing demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 