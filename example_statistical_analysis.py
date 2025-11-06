#!/usr/bin/env python3
"""
Example: Programmatic Statistical Significance Analysis

This script demonstrates how to use the statistical analysis functions
programmatically outside of the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from analyze_statistical_significance import (
    analyze_significance_pairs,
    create_summary_table,
    export_summary_to_csv
)
from scipy import stats
import scikit_posthocs as sp


def load_experiment_data(experiment_path: str) -> pd.DataFrame:
    """
    Load experiment data from CSV file

    Args:
        experiment_path: Path to the comprehensive_results.csv file

    Returns:
        DataFrame with experiment results
    """
    csv_path = Path(experiment_path) / "csv_exports" / "comprehensive_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Experiment data not found at {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def perform_statistical_analysis(
    metrics_df: pd.DataFrame,
    data_type: str,
    datasets: list,
    metric: str
) -> dict:
    """
    Perform statistical analysis on a specific modality-metric combination

    Args:
        metrics_df: DataFrame with all metrics
        data_type: Data modality name
        datasets: List of dataset names for this modality
        metric: Metric to analyze

    Returns:
        Dictionary with analysis results
    """

    print(f"\nAnalyzing {data_type} - {metric}")
    print("-" * 60)

    # Filter data
    analysis_data = metrics_df[
        (metrics_df['Dataset'].isin(datasets)) &
        (metrics_df[metric].notna())
    ]

    if analysis_data.empty:
        print(f"No data available for {data_type} with {metric}")
        return None

    if len(analysis_data['Method'].unique()) < 2:
        print(f"Need at least 2 methods for comparison")
        return None

    # Create pivot table
    pivot_data = analysis_data.pivot_table(
        values=metric,
        index=['Dataset', 'Model'],
        columns='Method',
        aggfunc='mean'
    ).reset_index().dropna()

    if pivot_data.empty or len(pivot_data.columns) <= 3:
        print("Insufficient data after processing")
        return None

    method_columns = [col for col in pivot_data.columns if col not in ['Dataset', 'Model']]
    method_data = pivot_data[method_columns]

    if len(method_data) < 3:
        print(f"Need at least 3 observations, found {len(method_data)}")
        return None

    # Perform Friedman test
    data_for_friedman = [method_data[col].values for col in method_columns]
    friedman_stat, friedman_p = stats.friedmanchisquare(*data_for_friedman)

    print(f"Friedman test: χ² = {friedman_stat:.4f}, p = {friedman_p:.4f}")

    if friedman_p >= 0.05:
        print("No significant differences found (p >= 0.05)")
        return None

    print("Significant differences detected! Running post-hoc analysis...")

    # Calculate rankings and perform Nemenyi test
    ranking_matrix = method_data.rank(axis=1, method='average', ascending=False)
    avg_ranks = ranking_matrix.mean().sort_values()
    posthoc_results = sp.posthoc_nemenyi_friedman(ranking_matrix)

    # Analyze significance pairs
    summary = analyze_significance_pairs(
        metrics_df=analysis_data,
        data_type=data_type,
        selected_datasets=datasets,
        metric=metric,
        posthoc_results=posthoc_results,
        avg_ranks=avg_ranks
    )

    # Display results
    print(f"\nResults:")
    print(f"  Total Comparisons: {summary['total_comparisons']}")
    print(f"  Significant: {summary['significant_comparisons']} ({summary['significance_rate']})")
    print(f"  P-value Range (sig): {summary['p_value_range_significant']}")
    print(f"  Mean P-value (sig): {summary['mean_p_value_significant']}")
    print(f"  Behavioral Clusters: {len(summary['clusters'])}")

    print(f"\nInterpretation:")
    print(f"  {summary['interpretation']}")

    if summary['clusters']:
        print(f"\nClusters:")
        for idx, cluster in enumerate(summary['clusters'], 1):
            cluster_ranks = [summary['average_ranks'][m] for m in cluster]
            avg_rank = np.mean(cluster_ranks)
            print(f"  {idx}. {', '.join(cluster)} (avg rank: {avg_rank:.2f})")

    return summary


def main():
    """Main execution function"""

    print("=" * 60)
    print("Statistical Significance Analysis - Example Script")
    print("=" * 60)

    # Configuration
    EXPERIMENT_PATH = "results/experiment_20251020_163046"  # Update with your path

    # Define modalities and their datasets
    modalities = {
        'Binary Tabular': ['adult_income', 'compas', 'breast_cancer'],
        'Multiclass Tabular': ['iris', 'wine', 'digits'],
        'Image': ['mnist', 'cifar10', 'fashion_mnist'],
        'Text': ['imdb', 'ag_news', 'yelp']
    }

    # Metrics to analyze (all evaluation metrics from your code)
    metrics_to_analyze = ['faithfulness', 'monotonicity', 'completeness', 'stability',
                         'consistency', 'sparsity', 'simplicity']

    try:
        # Load data
        print(f"\nLoading experiment data from: {EXPERIMENT_PATH}")
        metrics_df = load_experiment_data(EXPERIMENT_PATH)
        print(f"Loaded {len(metrics_df)} records")

        # Get actual datasets from the data
        available_datasets = metrics_df['Dataset'].unique()
        print(f"\nAvailable datasets: {', '.join(available_datasets)}")

        # Perform analyses
        all_summaries = []

        for data_type, datasets in modalities.items():
            # Filter to only available datasets
            actual_datasets = [d for d in datasets if d in available_datasets]

            if not actual_datasets:
                print(f"\nSkipping {data_type} - no datasets available")
                continue

            for metric in metrics_to_analyze:
                summary = perform_statistical_analysis(
                    metrics_df=metrics_df,
                    data_type=data_type,
                    datasets=actual_datasets,
                    metric=metric
                )

                if summary:
                    all_summaries.append(summary)

        # Create summary table
        if all_summaries:
            print("\n" + "=" * 60)
            print("Summary Table")
            print("=" * 60)

            summary_table = create_summary_table(all_summaries)
            print(summary_table.to_string())

            # Export results
            export_path = Path("results") / "statistical_summaries"
            export_path.mkdir(parents=True, exist_ok=True)

            for summary in all_summaries:
                export_summary_to_csv(summary, export_path)

            summary_table.to_csv(export_path / "all_summaries_table.csv", index=False)

            print(f"\nResults exported to: {export_path}")
            print(f"Total analyses completed: {len(all_summaries)}")
        else:
            print("\nNo significant results found across all analyses")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease update EXPERIMENT_PATH with your actual experiment folder")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
