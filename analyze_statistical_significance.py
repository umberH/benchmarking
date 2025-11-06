#!/usr/bin/env python3
"""
Statistical Significance Analysis and Summarization
This script analyzes pairwise statistical significance results and generates
comprehensive summaries by data modality and metric.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json


def analyze_significance_pairs(
    metrics_df: pd.DataFrame,
    data_type: str,
    selected_datasets: List[str],
    metric: str,
    posthoc_results: pd.DataFrame,
    avg_ranks: pd.Series
) -> Dict[str, Any]:
    """
    Analyze all statistical significance pairs and generate comprehensive summary

    Args:
        metrics_df: DataFrame with all metrics
        data_type: Type of data (Binary Tabular, Multiclass Tabular, Image, Text)
        selected_datasets: List of datasets for this modality
        metric: The metric being analyzed
        posthoc_results: Post-hoc test results (Nemenyi)
        avg_ranks: Average ranks of methods

    Returns:
        Dictionary containing comprehensive statistical summary
    """

    # Extract pairwise comparisons
    comparison_results = []
    methods = avg_ranks.index.tolist()

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                p_val = posthoc_results.loc[method1, method2]
                significant = p_val < 0.05
                comparison_results.append({
                    'Method 1': method1,
                    'Method 2': method2,
                    'p-value': p_val,
                    'Significant (α=0.05)': "Yes" if significant else "No"
                })

    comparison_df = pd.DataFrame(comparison_results)

    # Overall statistics
    total_comparisons = len(comparison_df)
    significant_comparisons = len(comparison_df[comparison_df['Significant (α=0.05)'] == "Yes"])

    # P-value statistics
    pval_stats = comparison_df['p-value'].describe()

    # Filter significant comparisons
    sig_df = comparison_df[comparison_df['Significant (α=0.05)'] == "Yes"]

    # Get significant p-value range
    if not sig_df.empty:
        sig_pval_min = sig_df['p-value'].min()
        sig_pval_max = sig_df['p-value'].max()
        sig_pval_mean = sig_df['p-value'].mean()
        sig_pval_range = f"{sig_pval_min:.4f} – {sig_pval_max:.4f}"
    else:
        sig_pval_min = sig_pval_max = sig_pval_mean = None
        sig_pval_range = "N/A"

    # Find most significant comparisons (lowest p-values)
    lowest_pvals = comparison_df.nsmallest(min(10, len(comparison_df)), 'p-value')
    most_significant = lowest_pvals[lowest_pvals['Significant (α=0.05)'] == "Yes"]

    # Group comparisons by methods
    method_involvement = {}
    for _, row in sig_df.iterrows():
        method1 = row['Method 1']
        method2 = row['Method 2']
        p_val = row['p-value']

        if method1 not in method_involvement:
            method_involvement[method1] = []
        if method2 not in method_involvement:
            method_involvement[method2] = []

        method_involvement[method1].append((method2, p_val))
        method_involvement[method2].append((method1, p_val))

    # Identify behavioral clusters based on ranking and significance
    clusters = identify_clusters(avg_ranks, sig_df, methods)

    # Generate interpretation
    interpretation = generate_interpretation(
        total_comparisons=total_comparisons,
        significant_comparisons=significant_comparisons,
        most_significant=most_significant,
        method_involvement=method_involvement,
        clusters=clusters,
        avg_ranks=avg_ranks,
        sig_pval_mean=sig_pval_mean
    )

    summary = {
        'data_type': data_type,
        'metric': metric,
        'datasets': selected_datasets,
        'total_comparisons': total_comparisons,
        'significant_comparisons': significant_comparisons,
        'significance_rate': f"{(significant_comparisons/total_comparisons)*100:.1f}%" if total_comparisons > 0 else "0%",
        'p_value_range_significant': sig_pval_range,
        'mean_p_value_significant': f"{sig_pval_mean:.4f}" if sig_pval_mean is not None else "N/A",
        'most_significant_comparisons': format_most_significant(most_significant),
        'method_involvement': method_involvement,
        'clusters': clusters,
        'interpretation': interpretation,
        'all_comparisons': comparison_df,
        'significant_only': sig_df,
        'average_ranks': avg_ranks.to_dict()
    }

    return summary


def identify_clusters(
    avg_ranks: pd.Series,
    sig_df: pd.DataFrame,
    methods: List[str]
) -> List[List[str]]:
    """
    Identify behavioral clusters based on rankings and significance patterns

    Args:
        avg_ranks: Average ranks of methods
        sig_df: Significant comparisons DataFrame
        methods: List of all methods

    Returns:
        List of clusters, where each cluster is a list of method names
    """
    # Create adjacency matrix for non-significant comparisons
    # Methods that are NOT significantly different should be in same cluster
    adjacency = {}
    for method in methods:
        adjacency[method] = set([method])  # Each method is connected to itself

    # Add connections for non-significant pairs (similar behavior)
    all_pairs = set()
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                all_pairs.add((method1, method2))

    # Get significant pairs
    sig_pairs = set()
    for _, row in sig_df.iterrows():
        sig_pairs.add((row['Method 1'], row['Method 2']))

    # Non-significant pairs are similar
    non_sig_pairs = all_pairs - sig_pairs
    for method1, method2 in non_sig_pairs:
        adjacency[method1].add(method2)
        adjacency[method2].add(method1)

    # Find connected components (clusters)
    visited = set()
    clusters = []

    for method in methods:
        if method not in visited:
            cluster = []
            stack = [method]

            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    cluster.append(current)
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)

            # Sort cluster by rank
            cluster_sorted = sorted(cluster, key=lambda m: avg_ranks[m])
            clusters.append(cluster_sorted)

    # Sort clusters by average rank
    clusters = sorted(clusters, key=lambda c: np.mean([avg_ranks[m] for m in c]))

    return clusters


def format_most_significant(most_significant: pd.DataFrame) -> str:
    """Format most significant comparisons as readable string"""
    if most_significant.empty:
        return "None"

    formatted = []
    for _, row in most_significant.head(5).iterrows():
        formatted.append(f"{row['Method 1']} vs. {row['Method 2']} (p ~= {row['p-value']:.4f})")

    return "; ".join(formatted)


def generate_interpretation(
    total_comparisons: int,
    significant_comparisons: int,
    most_significant: pd.DataFrame,
    method_involvement: Dict[str, List],
    clusters: List[List[str]],
    avg_ranks: pd.Series,
    sig_pval_mean: float
) -> str:
    """
    Generate human-readable interpretation of statistical results

    Args:
        total_comparisons: Total number of pairwise comparisons
        significant_comparisons: Number of significant comparisons
        most_significant: DataFrame of most significant comparisons
        method_involvement: Dictionary of methods and their significant pairs
        clusters: List of behavioral clusters
        avg_ranks: Average ranks of methods
        sig_pval_mean: Mean p-value of significant comparisons

    Returns:
        Formatted interpretation string
    """
    interpretation_parts = []

    # Overall significance
    sig_rate = (significant_comparisons / total_comparisons * 100) if total_comparisons > 0 else 0
    interpretation_parts.append(
        f"{significant_comparisons} of {total_comparisons} method pairs differ significantly at alpha = 0.05 ({sig_rate:.1f}%)."
    )

    # Most significant findings
    if not most_significant.empty:
        top_pair = most_significant.iloc[0]
        min_p = top_pair['p-value']
        interpretation_parts.append(
            f"The strongest separation is between {top_pair['Method 1']} and {top_pair['Method 2']} "
            f"with p ~= {min_p:.4f}."
        )

    # Method involvement patterns
    if method_involvement:
        # Find methods with most significant differences
        method_sig_counts = {method: len(pairs) for method, pairs in method_involvement.items()}
        top_methods = sorted(method_sig_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        if top_methods:
            top_method_name, top_count = top_methods[0]
            interpretation_parts.append(
                f"{top_method_name} shows significant differences from {top_count} other methods, "
                f"indicating distinct behavior."
            )

    # Cluster analysis
    if len(clusters) > 1:
        interpretation_parts.append(
            f"The methods form {len(clusters)} distinct behavioral clusters based on statistical analysis:"
        )

        for idx, cluster in enumerate(clusters, 1):
            avg_cluster_rank = np.mean([avg_ranks[m] for m in cluster])
            cluster_str = ", ".join(cluster)
            interpretation_parts.append(
                f"  Cluster {idx} (avg rank {avg_cluster_rank:.2f}): {cluster_str}"
            )
    else:
        interpretation_parts.append(
            "All methods that are not significantly different form a single behavioral cluster, "
            "suggesting similar performance patterns."
        )

    # Performance ranking insight
    best_method = avg_ranks.idxmin()
    worst_method = avg_ranks.idxmax()
    interpretation_parts.append(
        f"Best performing method: {best_method} (rank {avg_ranks[best_method]:.2f}); "
        f"Worst performing: {worst_method} (rank {avg_ranks[worst_method]:.2f})."
    )

    return " ".join(interpretation_parts)


def create_summary_table(summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary table from multiple statistical analyses

    Args:
        summaries: List of summary dictionaries

    Returns:
        DataFrame with summary statistics
    """
    rows = []

    for summary in summaries:
        row = {
            'Data Type': summary['data_type'],
            'Metric': summary['metric'],
            'Total Comparisons': summary['total_comparisons'],
            'Significant': summary['significant_comparisons'],
            'Significance Rate': summary['significance_rate'],
            'P-value Range (sig)': summary['p_value_range_significant'],
            'Mean P-value (sig)': summary['mean_p_value_significant'],
            'Top Comparison': summary['most_significant_comparisons'].split(';')[0] if summary['most_significant_comparisons'] != "None" else "N/A",
            'Clusters': len(summary['clusters'])
        }
        rows.append(row)

    return pd.DataFrame(rows)


def export_summary_to_csv(summary: Dict[str, Any], output_path: Path):
    """Export summary to CSV file"""

    # Export all comparisons
    if 'all_comparisons' in summary and not summary['all_comparisons'].empty:
        all_comp_path = output_path / f"{summary['data_type']}_{summary['metric']}_all_comparisons.csv"
        summary['all_comparisons'].to_csv(all_comp_path, index=False)

    # Export significant comparisons
    if 'significant_only' in summary and not summary['significant_only'].empty:
        sig_comp_path = output_path / f"{summary['data_type']}_{summary['metric']}_significant_comparisons.csv"
        summary['significant_only'].to_csv(sig_comp_path, index=False)

    # Export summary statistics
    summary_data = {
        'Data Type': summary['data_type'],
        'Metric': summary['metric'],
        'Total Comparisons': summary['total_comparisons'],
        'Significant Comparisons': summary['significant_comparisons'],
        'Significance Rate': summary['significance_rate'],
        'P-value Range (significant)': summary['p_value_range_significant'],
        'Mean P-value (significant)': summary['mean_p_value_significant'],
        'Most Significant Comparisons': summary['most_significant_comparisons'],
        'Number of Clusters': len(summary['clusters']),
        'Interpretation': summary['interpretation']
    }

    summary_path = output_path / f"{summary['data_type']}_{summary['metric']}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)


if __name__ == "__main__":
    print("Statistical Significance Analysis Module")
    print("This module is meant to be imported and used within the Streamlit dashboard")
