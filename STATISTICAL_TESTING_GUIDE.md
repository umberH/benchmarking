# Statistical Significance Testing Guide for XAI Benchmarking

## Table of Contents
1. [Overview](#overview)
2. [Statistical Tests Available](#statistical-tests-available)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Interpreting Results](#interpreting-results)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Overview

Statistical significance testing is crucial for XAI benchmarking to determine whether differences in performance between explanation methods are statistically meaningful rather than due to chance. This guide explains how to use the integrated statistical testing capabilities in the XAI benchmarking framework.

### Key Features

- **Multiple Statistical Tests**: T-test, Mann-Whitney U, Wilcoxon, ANOVA, Kruskal-Wallis, Friedman
- **Effect Size Calculations**: Cohen's d, Hedges' g, Eta squared
- **Power Analysis**: Sample size recommendations and power estimates
- **Multiple Comparison Corrections**: Bonferroni, Holm, FDR methods
- **Correlation Analysis**: Pearson and Spearman correlations between metrics
- **Visualizations**: Heatmaps, effect size plots, power analysis charts

## Statistical Tests Available

### 1. Pairwise Comparisons

#### T-Test (Parametric)
- **Use case**: Compare two methods when data is normally distributed
- **Assumptions**: Normal distribution, equal variances
- **Output**: t-statistic, p-value, means, standard deviations

#### Mann-Whitney U Test (Non-parametric)
- **Use case**: Compare two methods when data is not normally distributed
- **Assumptions**: Independent samples, ordinal data
- **Output**: U-statistic, p-value

#### Wilcoxon Signed-Rank Test
- **Use case**: Compare two methods with paired/related samples
- **Assumptions**: Paired observations, ordinal data
- **Output**: W-statistic, p-value

### 2. Multiple Comparisons

#### One-Way ANOVA
- **Use case**: Compare more than two methods simultaneously
- **Assumptions**: Normal distribution, equal variances, independence
- **Output**: F-statistic, p-value

#### Kruskal-Wallis Test
- **Use case**: Non-parametric alternative to ANOVA
- **Assumptions**: Independent samples, ordinal data
- **Output**: H-statistic, p-value

#### Friedman Test
- **Use case**: Compare multiple methods with repeated measures
- **Assumptions**: Related samples, ordinal data
- **Output**: Chi-square statistic, p-value

### 3. Post-Hoc Tests

#### Tukey's HSD Test
- **Use case**: Pairwise comparisons after significant ANOVA
- **Output**: Adjusted p-values for all pairs

### 4. Effect Size Measures

#### Cohen's d
- **Interpretation**:
  - Small: |d| < 0.2
  - Medium: 0.2 ≤ |d| < 0.5
  - Large: |d| ≥ 0.8

#### Hedges' g
- **Use case**: Bias-corrected version of Cohen's d for small samples

#### Eta Squared
- **Use case**: Effect size for ANOVA
- **Interpretation**:
  - Small: η² < 0.01
  - Medium: 0.01 ≤ η² < 0.06
  - Large: η² ≥ 0.14

## Configuration

### Basic Configuration

Add statistical testing parameters to your configuration file:

```yaml
evaluation:
  # Statistical testing parameters
  significance_level: 0.05
  multiple_comparison_correction: "bonferroni"
  statistical_tests:
    - "t_test"
    - "mann_whitney"
    - "wilcoxon"
    - "anova"
    - "kruskal_wallis"
    - "friedman"
  effect_size_calculation: true
  power_analysis: true
  correlation_analysis: true
```

### Advanced Configuration

```yaml
evaluation:
  # Statistical testing parameters
  significance_level: 0.05  # Alpha level for significance
  multiple_comparison_correction: "bonferroni"  # Correction method
  statistical_tests:
    - "t_test"
    - "mann_whitney"
    - "wilcoxon"
    - "anova"
    - "kruskal_wallis"
    - "friedman"
  effect_size_calculation: true
  power_analysis: true
  correlation_analysis: true
  
  # Power analysis settings
  power_analysis:
    target_power: 0.8
    min_effect_size: 0.2
    
  # Multiple comparison correction methods
  correction_methods:
    - "bonferroni"
    - "holm"
    - "fdr_bh"
    - "fdr_by"
```

## Usage Examples

### 1. Basic Statistical Testing

```python
from src.evaluation.statistical_tests import StatisticalTester
from src.utils.config import load_config

# Load configuration
config = load_config("configs/default_config.yaml")

# Initialize statistical tester
statistical_tester = StatisticalTester(config)

# Run statistical tests on your results
statistical_results = statistical_tester.run_statistical_tests(benchmark_results)

# Access results
summary = statistical_results['summary']
pairwise_results = statistical_results['pairwise_comparisons']
effect_sizes = statistical_results['effect_sizes']
```

### 2. Integrated with Benchmarking Pipeline

The statistical testing is automatically integrated into the main benchmarking pipeline:

```python
from src.benchmark import XAIBenchmark
from src.utils.config import load_config

# Load configuration
config = load_config("configs/default_config.yaml")

# Initialize benchmark
benchmark = XAIBenchmark(config, output_dir)

# Run full pipeline (includes statistical testing)
benchmark.run_full_pipeline()

# Access statistical results
statistical_results = benchmark.results['statistical_results']
```

### 3. Custom Statistical Analysis

```python
# Run specific statistical tests
from src.evaluation.statistical_tests import StatisticalTester

# Initialize tester
tester = StatisticalTester(config)

# Generate mock data for demonstration
import numpy as np
mock_results = {
    'method1': {
        'metrics': {
            'faithfulness': np.random.normal(0.85, 0.1, 20),
            'monotonicity': np.random.normal(0.78, 0.08, 20)
        }
    },
    'method2': {
        'metrics': {
            'faithfulness': np.random.normal(0.72, 0.12, 20),
            'monotonicity': np.random.normal(0.65, 0.10, 20)
        }
    }
}

# Run statistical tests
results = tester.run_statistical_tests(mock_results)

# Apply multiple comparison correction
p_values = [0.01, 0.03, 0.05, 0.08, 0.12]
corrected_p_values, significant = tester.apply_multiple_comparison_correction(
    p_values, method='bonferroni'
)
```

### 4. Running the Statistical Testing Example

```bash
# Run the comprehensive statistical testing example
python statistical_testing_example.py
```

This will:
- Generate mock benchmark results
- Run all statistical tests
- Create visualizations
- Demonstrate multiple comparison corrections

## Interpreting Results

### 1. Summary Statistics

```python
summary = statistical_results['summary']
print(f"Total tests: {summary['total_tests']}")
print(f"Significant tests: {summary['significant_tests']}")
print(f"Significance rate: {summary['significance_rate']:.2%}")
```

### 2. Pairwise Comparisons

```python
# Check significant differences
for metric_name, pairwise_results in statistical_results['pairwise_comparisons'].items():
    for pair_name, test_result in pairwise_results['t_test'].items():
        if test_result['significant']:
            print(f"Significant difference in {metric_name}: {pair_name}")
            print(f"  p-value: {test_result['p_value']:.4f}")
            print(f"  Effect: {test_result['effect']}")
```

### 3. Effect Sizes

```python
# Interpret effect sizes
for metric_name, effect_sizes in statistical_results['effect_sizes'].items():
    for pair_name, cohens_d in effect_sizes['cohens_d'].items():
        if abs(cohens_d) > 0.8:
            print(f"Large effect in {metric_name}: {pair_name} (d={cohens_d:.3f})")
        elif abs(cohens_d) > 0.5:
            print(f"Medium effect in {metric_name}: {pair_name} (d={cohens_d:.3f})")
```

### 4. Power Analysis

```python
# Check power adequacy
for metric_name, power_results in statistical_results['power_analysis'].items():
    for pair_name, power_est in power_results['power_estimates'].items():
        if not power_est['adequate_power']:
            print(f"Inadequate power for {pair_name}: {power_est['power']:.3f}")
            print(f"Recommendation: {power_results['sample_size_analysis'][pair_name]['recommendation']}")
```

## Best Practices

### 1. Sample Size Planning

- **Minimum sample size**: At least 20 samples per method for reliable statistical testing
- **Power analysis**: Aim for 80% power to detect meaningful differences
- **Effect size consideration**: Plan for detecting medium to large effect sizes

### 2. Multiple Comparison Corrections

- **When to use**: Always when performing multiple statistical tests
- **Recommended methods**:
  - **Bonferroni**: Conservative, controls family-wise error rate
  - **Holm**: Less conservative than Bonferroni
  - **FDR methods**: Control false discovery rate

### 3. Test Selection

- **Parametric vs Non-parametric**:
  - Use parametric tests (t-test, ANOVA) when data is normally distributed
  - Use non-parametric tests (Mann-Whitney, Kruskal-Wallis) when assumptions are violated
- **Paired vs Independent**:
  - Use paired tests when comparing the same samples across methods
  - Use independent tests when comparing different samples

### 4. Reporting Results

- **Always report**: p-values, effect sizes, and confidence intervals
- **Include**: Sample sizes, test statistics, and assumptions checked
- **Interpret**: Practical significance in addition to statistical significance

### 5. Visualization

- **P-value heatmaps**: Show significance patterns across method pairs
- **Effect size plots**: Visualize magnitude of differences
- **Power analysis charts**: Identify underpowered comparisons
- **Correlation matrices**: Understand relationships between metrics

## Troubleshooting

### Common Issues

#### 1. Insufficient Sample Size
**Problem**: Low statistical power
**Solution**: Increase sample size or use more sensitive tests

#### 2. Non-normal Data
**Problem**: Violation of parametric test assumptions
**Solution**: Use non-parametric alternatives (Mann-Whitney, Kruskal-Wallis)

#### 3. Multiple Testing Issues
**Problem**: High false positive rate
**Solution**: Apply multiple comparison corrections

#### 4. Missing Data
**Problem**: Incomplete statistical analysis
**Solution**: Use appropriate missing data handling methods

### Error Messages and Solutions

#### "Need at least 2 methods for pairwise comparison"
- **Cause**: Only one method in results
- **Solution**: Include multiple methods in your benchmark

#### "Need at least 3 methods for multiple comparison"
- **Cause**: Only two methods for ANOVA/Kruskal-Wallis
- **Solution**: Include at least three methods or use pairwise tests

#### "Error in statistical test"
- **Cause**: Data format issues or insufficient data
- **Solution**: Check data format and ensure adequate sample sizes

### Performance Optimization

#### 1. Efficient Testing
- Use appropriate tests for your data type
- Consider computational complexity for large datasets

#### 2. Memory Management
- Process large datasets in chunks
- Use efficient data structures

#### 3. Parallel Processing
- Run independent tests in parallel
- Use multiprocessing for large-scale analyses

## Advanced Topics

### 1. Bayesian Statistical Testing

For more advanced analysis, consider implementing Bayesian alternatives:
- Bayesian t-test
- Bayesian ANOVA
- Credible intervals

### 2. Robust Statistical Methods

For data with outliers or violations of assumptions:
- Robust t-test
- Trimmed means
- Winsorized statistics

### 3. Meta-analysis

For combining results across multiple studies:
- Fixed-effects models
- Random-effects models
- Heterogeneity testing

### 4. Machine Learning Integration

- Automated test selection based on data characteristics
- Adaptive significance levels
- Learning-based effect size estimation

## Conclusion

Statistical significance testing is essential for rigorous XAI benchmarking. The integrated statistical testing framework provides comprehensive tools for:

- **Rigorous comparison** of explanation methods
- **Effect size quantification** for practical significance
- **Power analysis** for study design
- **Multiple comparison control** for family-wise error rates
- **Visualization** of statistical relationships

By following the guidelines in this document, you can ensure that your XAI benchmarking results are statistically sound and scientifically rigorous. 