"""
Statistical significance testing for XAI benchmarking
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, friedmanchisquare, kruskal
from scipy.stats import f_oneway, chi2_contingency, pearsonr, spearmanr
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
# from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class StatisticalTester:
    """
    Statistical significance testing for XAI explanation methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize statistical tester
        
        Args:
            config: Configuration dictionary for statistical testing
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alpha = config.get('evaluation', {}).get('significance_level', 0.05)
        self.correction_method = config.get('evaluation', {}).get('multiple_comparison_correction', 'bonferroni')
        
    def run_statistical_tests(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive statistical tests on benchmark results
        
        Args:
            results: Dictionary containing benchmark results
            
        Returns:
            Dictionary containing statistical test results
        """
        self.logger.info("Running statistical significance tests...")
        
        statistical_results = {
            'summary': {},
            'pairwise_comparisons': {},
            'multiple_comparisons': {},
            'correlation_analysis': {},
            'effect_sizes': {},
            'power_analysis': {}
        }
        
        # Extract metrics for testing
        metrics_data = self._extract_metrics_data(results)
        
        if not metrics_data:
            self.logger.warning("No metrics data found for statistical testing")
            return statistical_results
        
        # Run different types of statistical tests
        for metric_name, metric_data in metrics_data.items():
            self.logger.info(f"Testing metric: {metric_name}")
            
            # Pairwise comparisons
            pairwise_results = self._pairwise_comparisons(metric_name, metric_data)
            statistical_results['pairwise_comparisons'][metric_name] = pairwise_results
            
            # Multiple comparisons (ANOVA, Kruskal-Wallis)
            multiple_results = self._multiple_comparisons(metric_name, metric_data)
            statistical_results['multiple_comparisons'][metric_name] = multiple_results
            
            # Effect sizes
            effect_sizes = self._calculate_effect_sizes(metric_name, metric_data)
            statistical_results['effect_sizes'][metric_name] = effect_sizes
            
            # Power analysis
            power_results = self._power_analysis(metric_name, metric_data)
            statistical_results['power_analysis'][metric_name] = power_results
        
        # Correlation analysis between metrics
        correlation_results = self._correlation_analysis(metrics_data)
        statistical_results['correlation_analysis'] = correlation_results
        
        # Generate summary
        statistical_results['summary'] = self._generate_summary(statistical_results)
        
        return statistical_results
    
    def _extract_metrics_data(self, results: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
        """
        Extract metrics data from benchmark results
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Dictionary mapping metric names to method results
        """
        metrics_data = {}
        
        # Extract results for each method
        for method_name, method_results in results.items():
            if isinstance(method_results, dict) and 'metrics' in method_results:
                for metric_name, metric_value in method_results['metrics'].items():
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = {}
                    
                    # Handle different data types
                    if isinstance(metric_value, (list, np.ndarray)):
                        metrics_data[metric_name][method_name] = metric_value
                    elif isinstance(metric_value, (int, float)):
                        # If single value, create a list for statistical testing
                        metrics_data[metric_name][method_name] = [metric_value]
        
        return metrics_data
    
    def _pairwise_comparisons(self, metric_name: str, metric_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform pairwise statistical comparisons between methods
        
        Args:
            metric_name: Name of the metric being tested
            metric_data: Dictionary mapping method names to metric values
            
        Returns:
            Dictionary containing pairwise test results
        """
        methods = list(metric_data.keys())
        if len(methods) < 2:
            return {'error': 'Need at least 2 methods for pairwise comparison'}
        
        results = {
            't_test': {},
            'mann_whitney': {},
            'wilcoxon': {},
            'significant_differences': []
        }
        
        # Perform all pairwise comparisons
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                pair_name = f"{method1}_vs_{method2}"
                data1 = np.array(metric_data[method1])
                data2 = np.array(metric_data[method2])
                
                # T-test (parametric)
                try:
                    t_stat, p_value = ttest_ind(data1, data2)
                    results['t_test'][pair_name] = {
                        'statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < self.alpha,
                        'method1_mean': np.mean(data1),
                        'method2_mean': np.mean(data2),
                        'method1_std': np.std(data1),
                        'method2_std': np.std(data2)
                    }
                except Exception as e:
                    results['t_test'][pair_name] = {'error': str(e)}
                
                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                    results['mann_whitney'][pair_name] = {
                        'statistic': u_stat,
                        'p_value': p_value,
                        'significant': p_value < self.alpha
                    }
                except Exception as e:
                    results['mann_whitney'][pair_name] = {'error': str(e)}
                
                # Wilcoxon signed-rank test (for paired data)
                try:
                    if len(data1) == len(data2):
                        w_stat, p_value = wilcoxon(data1, data2)
                        results['wilcoxon'][pair_name] = {
                            'statistic': w_stat,
                            'p_value': p_value,
                            'significant': p_value < self.alpha
                        }
                except Exception as e:
                    results['wilcoxon'][pair_name] = {'error': str(e)}
                
                # Track significant differences
                if 't_test' in results and pair_name in results['t_test']:
                    if results['t_test'][pair_name].get('significant', False):
                        results['significant_differences'].append({
                            'pair': pair_name,
                            'test': 't_test',
                            'p_value': results['t_test'][pair_name]['p_value'],
                            'effect': 'method1_better' if np.mean(data1) > np.mean(data2) else 'method2_better'
                        })
        
        return results
    
    def _multiple_comparisons(self, metric_name: str, metric_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform multiple comparison tests (ANOVA, Kruskal-Wallis)
        
        Args:
            metric_name: Name of the metric being tested
            metric_data: Dictionary mapping method names to metric values
            
        Returns:
            Dictionary containing multiple comparison test results
        """
        methods = list(metric_data.keys())
        if len(methods) < 3:
            return {'error': 'Need at least 3 methods for multiple comparison'}
        
        results = {
            'anova': {},
            'kruskal_wallis': {},
            'friedman': {},
            'post_hoc': {}
        }
        
        # Prepare data for tests
        data_groups = [np.array(metric_data[method]) for method in methods]
        
        # One-way ANOVA (parametric)
        try:
            f_stat, p_value = f_oneway(*data_groups)
            results['anova'] = {
                'statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha
            }
            
            # Post-hoc Tukey test if ANOVA is significant (simplified version)
            if p_value < self.alpha:
                results['post_hoc']['tukey'] = {
                    'significant_pairs': [],
                    'p_values': {},
                    'note': 'Tukey test requires statsmodels package'
                }
                    
        except Exception as e:
            results['anova'] = {'error': str(e)}
        
        # Kruskal-Wallis test (non-parametric)
        try:
            h_stat, p_value = kruskal(*data_groups)
            results['kruskal_wallis'] = {
                'statistic': h_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha
            }
        except Exception as e:
            results['kruskal_wallis'] = {'error': str(e)}
        
        # Friedman test (for repeated measures)
        try:
            # Reshape data for Friedman test (assuming equal group sizes)
            min_size = min(len(group) for group in data_groups)
            friedman_data = np.array([group[:min_size] for group in data_groups]).T
            
            if friedman_data.shape[0] > 1 and friedman_data.shape[1] > 1:
                stat, p_value = friedmanchisquare(*[friedman_data[:, i] for i in range(friedman_data.shape[1])])
                results['friedman'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
        except Exception as e:
            results['friedman'] = {'error': str(e)}
        
        return results
    
    def _calculate_effect_sizes(self, metric_name: str, metric_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calculate effect sizes for comparisons
        
        Args:
            metric_name: Name of the metric being tested
            metric_data: Dictionary mapping method names to metric values
            
        Returns:
            Dictionary containing effect size calculations
        """
        methods = list(metric_data.keys())
        if len(methods) < 2:
            return {'error': 'Need at least 2 methods for effect size calculation'}
        
        results = {
            'cohens_d': {},
            'hedges_g': {},
            'eta_squared': {},
            'rank_biserial': {}
        }
        
        # Calculate effect sizes for all pairs
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                pair_name = f"{method1}_vs_{method2}"
                data1 = np.array(metric_data[method1])
                data2 = np.array(metric_data[method2])
                
                # Cohen's d
                try:
                    pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                        (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                       (len(data1) + len(data2) - 2))
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                    results['cohens_d'][pair_name] = cohens_d
                except Exception as e:
                    results['cohens_d'][pair_name] = {'error': str(e)}
                
                # Hedges' g (bias-corrected Cohen's d)
                try:
                    n1, n2 = len(data1), len(data2)
                    df = n1 + n2 - 2
                    correction_factor = 1 - 3 / (4 * df - 1)
                    hedges_g = cohens_d * correction_factor
                    results['hedges_g'][pair_name] = hedges_g
                except Exception as e:
                    results['hedges_g'][pair_name] = {'error': str(e)}
                
                # Eta squared (for ANOVA)
                try:
                    ss_between = len(data1) * (np.mean(data1) - np.mean(np.concatenate([data1, data2]))) ** 2 + \
                                len(data2) * (np.mean(data2) - np.mean(np.concatenate([data1, data2]))) ** 2
                    ss_total = np.sum((np.concatenate([data1, data2]) - np.mean(np.concatenate([data1, data2]))) ** 2)
                    eta_squared = ss_between / ss_total
                    results['eta_squared'][pair_name] = eta_squared
                except Exception as e:
                    results['eta_squared'][pair_name] = {'error': str(e)}
        
        return results
    
    def _power_analysis(self, metric_name: str, metric_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform power analysis for statistical tests
        
        Args:
            metric_name: Name of the metric being tested
            metric_data: Dictionary mapping method names to metric values
            
        Returns:
            Dictionary containing power analysis results
        """
        methods = list(metric_data.keys())
        if len(methods) < 2:
            return {'error': 'Need at least 2 methods for power analysis'}
        
        results = {
            'sample_size_analysis': {},
            'power_estimates': {},
            'recommendations': []
        }
        
        # Calculate power for pairwise comparisons
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                pair_name = f"{method1}_vs_{method2}"
                data1 = np.array(metric_data[method1])
                data2 = np.array(metric_data[method2])
                
                try:
                    # Calculate effect size
                    pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                        (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                       (len(data1) + len(data2) - 2))
                    effect_size = abs(np.mean(data1) - np.mean(data2)) / pooled_std
                    
                    # Estimate power using t-test
                    n1, n2 = len(data1), len(data2)
                    df = n1 + n2 - 2
                    
                    # Approximate power calculation
                    from scipy.stats import t
                    critical_t = t.ppf(1 - self.alpha/2, df)
                    noncentrality = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
                    power = 1 - t.cdf(critical_t, df, noncentrality) + t.cdf(-critical_t, df, noncentrality)
                    
                    results['power_estimates'][pair_name] = {
                        'effect_size': effect_size,
                        'power': power,
                        'sample_size_method1': n1,
                        'sample_size_method2': n2,
                        'adequate_power': power >= 0.8
                    }
                    
                    # Sample size recommendations
                    if power < 0.8:
                        # Calculate required sample size for 80% power
                        required_n_per_group = int(2 * (1.96 + 0.84) ** 2 / (effect_size ** 2))
                        results['sample_size_analysis'][pair_name] = {
                            'current_power': power,
                            'required_sample_size_per_group': required_n_per_group,
                            'recommendation': f"Increase sample size to {required_n_per_group} per group for 80% power"
                        }
                        results['recommendations'].append(f"For {pair_name}: {results['sample_size_analysis'][pair_name]['recommendation']}")
                    
                except Exception as e:
                    results['power_estimates'][pair_name] = {'error': str(e)}
        
        return results
    
    def _correlation_analysis(self, metrics_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """
        Analyze correlations between different metrics
        
        Args:
            metrics_data: Dictionary containing all metrics data
            
        Returns:
            Dictionary containing correlation analysis results
        """
        results = {
            'metric_correlations': {},
            'method_rankings': {},
            'correlation_significance': {}
        }
        
        metrics = list(metrics_data.keys())
        if len(metrics) < 2:
            return {'error': 'Need at least 2 metrics for correlation analysis'}
        
        # Calculate correlations between metrics
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics[i+1:], i+1):
                pair_name = f"{metric1}_vs_{metric2}"
                
                # Get common methods for both metrics
                methods1 = set(metrics_data[metric1].keys())
                methods2 = set(metrics_data[metric2].keys())
                common_methods = methods1.intersection(methods2)
                
                if len(common_methods) < 2:
                    continue
                
                # Calculate correlations
                values1 = [np.mean(metrics_data[metric1][method]) for method in common_methods]
                values2 = [np.mean(metrics_data[metric2][method]) for method in common_methods]
                
                try:
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(values1, values2)
                    
                    # Spearman correlation
                    spearman_r, spearman_p = spearmanr(values1, values2)
                    
                    results['metric_correlations'][pair_name] = {
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'methods_compared': list(common_methods)
                    }
                    
                    results['correlation_significance'][pair_name] = {
                        'pearson_significant': pearson_p < self.alpha,
                        'spearman_significant': spearman_p < self.alpha
                    }
                    
                except Exception as e:
                    results['metric_correlations'][pair_name] = {'error': str(e)}
        
        # Calculate method rankings across metrics
        for metric_name, metric_data in metrics_data.items():
            method_means = {method: np.mean(values) for method, values in metric_data.items()}
            sorted_methods = sorted(method_means.items(), key=lambda x: x[1], reverse=True)
            results['method_rankings'][metric_name] = {
                'ranking': [method for method, _ in sorted_methods],
                'scores': {method: score for method, score in sorted_methods}
            }
        
        return results
    
    def _generate_summary(self, statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of statistical test results
        
        Args:
            statistical_results: Complete statistical test results
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_tests': 0,
            'significant_tests': 0,
            'significant_pairs': [],
            'recommendations': [],
            'key_findings': []
        }
        
        # Count significant tests
        for metric_name, pairwise_results in statistical_results.get('pairwise_comparisons', {}).items():
            if 't_test' in pairwise_results:
                for pair_name, test_result in pairwise_results['t_test'].items():
                    summary['total_tests'] += 1
                    if test_result.get('significant', False):
                        summary['significant_tests'] += 1
                        summary['significant_pairs'].append({
                            'metric': metric_name,
                            'pair': pair_name,
                            'p_value': test_result['p_value'],
                            'effect': test_result.get('effect', 'unknown')
                        })
        
        # Calculate significance rate
        if summary['total_tests'] > 0:
            summary['significance_rate'] = summary['significant_tests'] / summary['total_tests']
        else:
            summary['significance_rate'] = 0.0
        
        # Generate recommendations
        if summary['significance_rate'] < 0.1:
            summary['recommendations'].append("Low significance rate - consider increasing sample size or effect size")
        
        if summary['significant_tests'] == 0:
            summary['recommendations'].append("No significant differences found - methods may be equivalent")
        
        # Add power analysis recommendations
        for metric_name, power_results in statistical_results.get('power_analysis', {}).items():
            if 'recommendations' in power_results:
                summary['recommendations'].extend(power_results['recommendations'])
        
        # Generate key findings
        if summary['significant_pairs']:
            summary['key_findings'].append(f"Found {len(summary['significant_pairs'])} significant differences between methods")
        
        # Add correlation findings
        correlation_results = statistical_results.get('correlation_analysis', {})
        if 'metric_correlations' in correlation_results:
            significant_correlations = 0
            for pair_name, corr_result in correlation_results['metric_correlations'].items():
                if corr_result.get('pearson_p', 1) < self.alpha or corr_result.get('spearman_p', 1) < self.alpha:
                    significant_correlations += 1
            
            if significant_correlations > 0:
                summary['key_findings'].append(f"Found {significant_correlations} significant correlations between metrics")
        
        return summary
    
    def apply_multiple_comparison_correction(self, p_values: List[float], method: str = None) -> Tuple[List[float], List[bool]]:
        """
        Apply multiple comparison correction to p-values
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'holm', 'fdr_bh', etc.)
            
        Returns:
            Tuple of (corrected_p_values, significant_after_correction)
        """
        if method is None:
            method = self.correction_method
        
        # Simple Bonferroni correction implementation
        if method == 'bonferroni':
            n_tests = len(p_values)
            corrected_p_values = [min(p * n_tests, 1.0) for p in p_values]
            significant = [p < self.alpha for p in corrected_p_values]
            return corrected_p_values, significant
        else:
            # For other methods, require statsmodels
            self.logger.warning(f"Multiple comparison correction method '{method}' requires statsmodels package")
            return p_values, [p < self.alpha for p in p_values] 