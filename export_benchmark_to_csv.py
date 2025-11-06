import json
import pandas as pd
from pathlib import Path

# Load benchmark results
print("Loading benchmark results...")
with open('results/experiment_20251020_163046/benchmark_results.json', 'r') as f:
    data = json.load(f)

results = data['comprehensive_results']
print(f"Total entries: {len(results)}")

# Create output directory
output_dir = Path('results/experiment_20251020_163046/csv_exports')
output_dir.mkdir(exist_ok=True)
print(f"\nExporting to: {output_dir}")

# ============================================================================
# 1. MAIN COMPREHENSIVE TABLE
# ============================================================================
print("\n1. Creating comprehensive_results.csv...")

comprehensive_data = []
for entry in results:
    row = {
        'dataset': entry['dataset'],
        'dataset_type': entry['dataset_type'],
        'model': entry['model'],
        'model_type': entry['model_type'],
        'explanation_method': entry['explanation_method'],
        'explanation_type': entry['explanation_type'],
        'used_tuned_params': entry['used_tuned_params'],
        'validation_status': entry['validation_status'],
    }

    # Add model performance metrics
    if entry.get('model_performance'):
        perf = entry['model_performance']
        row['model_accuracy'] = perf.get('accuracy')
        row['model_precision'] = perf.get('precision')
        row['model_recall'] = perf.get('recall')
        row['model_f1'] = perf.get('f1')

    # Add all evaluation metrics
    for metric, value in entry['evaluations'].items():
        row[f'eval_{metric}'] = value

    # Add explanation info
    if entry.get('explanation_info'):
        exp_info = entry['explanation_info']
        row['explanation_generation_time'] = exp_info.get('generation_time')
        row['explanation_size'] = exp_info.get('size')

    comprehensive_data.append(row)

df_comprehensive = pd.DataFrame(comprehensive_data)
df_comprehensive.to_csv(output_dir / 'comprehensive_results.csv', index=False)
print(f"   Saved: {len(df_comprehensive)} rows, {len(df_comprehensive.columns)} columns")

# ============================================================================
# 2. BASIC METRICS ONLY
# ============================================================================
print("\n2. Creating basic_metrics.csv...")

basic_metrics = [
    'faithfulness', 'stability', 'monotonicity', 'completeness',
    'simplicity', 'sparsity', 'consistency', 'time_complexity'
]

basic_data = []
for entry in results:
    row = {
        'dataset': entry['dataset'],
        'dataset_type': entry['dataset_type'],
        'model': entry['model'],
        'explanation_method': entry['explanation_method'],
    }

    for metric in basic_metrics:
        row[metric] = entry['evaluations'].get(metric)

    basic_data.append(row)

df_basic = pd.DataFrame(basic_data)
df_basic.to_csv(output_dir / 'basic_metrics.csv', index=False)
print(f"   Saved: {len(df_basic)} rows, {len(df_basic.columns)} columns")

# ============================================================================
# 3. ADVANCED METRICS ONLY
# ============================================================================
print("\n3. Creating advanced_metrics.csv...")

advanced_data = []
for entry in results:
    row = {
        'dataset': entry['dataset'],
        'dataset_type': entry['dataset_type'],
        'model': entry['model'],
        'explanation_method': entry['explanation_method'],
    }

    # Add all advanced metrics
    for metric, value in entry['evaluations'].items():
        if metric.startswith('advanced_'):
            row[metric] = value

    advanced_data.append(row)

df_advanced = pd.DataFrame(advanced_data)
df_advanced.to_csv(output_dir / 'advanced_metrics.csv', index=False)
print(f"   Saved: {len(df_advanced)} rows, {len(df_advanced.columns)} columns")

# ============================================================================
# 4. TEXT-SPECIFIC METRICS
# ============================================================================
print("\n4. Creating text_metrics.csv...")

text_entries = [e for e in results if e['dataset_type'] == 'text']
text_data = []

for entry in text_entries:
    row = {
        'dataset': entry['dataset'],
        'model': entry['model'],
        'explanation_method': entry['explanation_method'],
    }

    # Text-specific metrics
    text_metrics = [
        'context_sensitivity', 'explanation_coverage', 'semantic_coherence',
        'sentiment_consistency', 'syntax_awareness', 'word_significance'
    ]

    for metric in text_metrics:
        row[metric] = entry['evaluations'].get(metric)

    # Compactness metrics for text
    for metric, value in entry['evaluations'].items():
        if 'compactness' in metric:
            row[metric] = value

    text_data.append(row)

df_text = pd.DataFrame(text_data)
df_text.to_csv(output_dir / 'text_metrics.csv', index=False)
print(f"   Saved: {len(df_text)} rows, {len(df_text.columns)} columns")

# ============================================================================
# 5. AGGREGATED BY DATASET
# ============================================================================
print("\n5. Creating aggregated_by_dataset.csv...")

dataset_groups = df_comprehensive.groupby('dataset')

agg_dataset_data = []
for dataset, group in dataset_groups:
    row = {
        'dataset': dataset,
        'dataset_type': group['dataset_type'].iloc[0],
        'num_evaluations': len(group),
        'num_models': group['model'].nunique(),
        'num_methods': group['explanation_method'].nunique(),
    }

    # Average metrics
    for col in df_comprehensive.columns:
        if col.startswith('eval_') and pd.api.types.is_numeric_dtype(df_comprehensive[col]):
            row[f'avg_{col}'] = group[col].mean()
            row[f'std_{col}'] = group[col].std()

    agg_dataset_data.append(row)

df_agg_dataset = pd.DataFrame(agg_dataset_data)
df_agg_dataset.to_csv(output_dir / 'aggregated_by_dataset.csv', index=False)
print(f"   Saved: {len(df_agg_dataset)} rows, {len(df_agg_dataset.columns)} columns")

# ============================================================================
# 6. AGGREGATED BY MODEL
# ============================================================================
print("\n6. Creating aggregated_by_model.csv...")

model_groups = df_comprehensive.groupby('model')

agg_model_data = []
for model, group in model_groups:
    row = {
        'model': model,
        'model_type': group['model_type'].iloc[0],
        'num_evaluations': len(group),
        'num_datasets': group['dataset'].nunique(),
        'num_methods': group['explanation_method'].nunique(),
    }

    # Average metrics
    for col in df_comprehensive.columns:
        if col.startswith('eval_') and pd.api.types.is_numeric_dtype(df_comprehensive[col]):
            row[f'avg_{col}'] = group[col].mean()
            row[f'std_{col}'] = group[col].std()

    agg_model_data.append(row)

df_agg_model = pd.DataFrame(agg_model_data)
df_agg_model.to_csv(output_dir / 'aggregated_by_model.csv', index=False)
print(f"   Saved: {len(df_agg_model)} rows, {len(df_agg_model.columns)} columns")

# ============================================================================
# 7. AGGREGATED BY EXPLANATION METHOD
# ============================================================================
print("\n7. Creating aggregated_by_method.csv...")

method_groups = df_comprehensive.groupby('explanation_method')

agg_method_data = []
for method, group in method_groups:
    row = {
        'explanation_method': method,
        'explanation_type': group['explanation_type'].iloc[0],
        'num_evaluations': len(group),
        'num_datasets': group['dataset'].nunique(),
        'num_models': group['model'].nunique(),
    }

    # Average metrics
    for col in df_comprehensive.columns:
        if col.startswith('eval_') and pd.api.types.is_numeric_dtype(df_comprehensive[col]):
            row[f'avg_{col}'] = group[col].mean()
            row[f'std_{col}'] = group[col].std()

    agg_method_data.append(row)

df_agg_method = pd.DataFrame(agg_method_data)
df_agg_method.to_csv(output_dir / 'aggregated_by_method.csv', index=False)
print(f"   Saved: {len(df_agg_method)} rows, {len(df_agg_method.columns)} columns")

# ============================================================================
# 8. AGGREGATED BY MODALITY (DATASET TYPE)
# ============================================================================
print("\n8. Creating aggregated_by_modality.csv...")

modality_groups = df_comprehensive.groupby('dataset_type')

agg_modality_data = []
for modality, group in modality_groups:
    row = {
        'modality': modality,
        'num_evaluations': len(group),
        'num_datasets': group['dataset'].nunique(),
        'num_models': group['model'].nunique(),
        'num_methods': group['explanation_method'].nunique(),
    }

    # Average metrics
    for col in df_comprehensive.columns:
        if col.startswith('eval_') and pd.api.types.is_numeric_dtype(df_comprehensive[col]):
            row[f'avg_{col}'] = group[col].mean()
            row[f'std_{col}'] = group[col].std()

    agg_modality_data.append(row)

df_agg_modality = pd.DataFrame(agg_modality_data)
df_agg_modality.to_csv(output_dir / 'aggregated_by_modality.csv', index=False)
print(f"   Saved: {len(df_agg_modality)} rows, {len(df_agg_modality.columns)} columns")

# ============================================================================
# 9. BEST PERFORMING METHODS PER METRIC
# ============================================================================
print("\n9. Creating best_methods_per_metric.csv...")

best_methods_data = []
for metric in basic_metrics:
    col_name = f'eval_{metric}'
    if col_name in df_comprehensive.columns:
        # Group by method and get mean
        method_scores = df_comprehensive.groupby('explanation_method')[col_name].mean()

        # Get best method
        best_method = method_scores.idxmax()
        best_score = method_scores.max()

        best_methods_data.append({
            'metric': metric,
            'best_method': best_method,
            'avg_score': best_score,
            'std_score': df_comprehensive[df_comprehensive['explanation_method'] == best_method][col_name].std()
        })

df_best_methods = pd.DataFrame(best_methods_data)
df_best_methods.to_csv(output_dir / 'best_methods_per_metric.csv', index=False)
print(f"   Saved: {len(df_best_methods)} rows, {len(df_best_methods.columns)} columns")

# ============================================================================
# 10. MODEL PERFORMANCE TABLE
# ============================================================================
print("\n10. Creating model_performance.csv...")

model_perf_data = []
for entry in results:
    if entry.get('model_performance'):
        row = {
            'dataset': entry['dataset'],
            'dataset_type': entry['dataset_type'],
            'model': entry['model'],
            'model_type': entry['model_type'],
            'used_tuned_params': entry['used_tuned_params'],
        }

        perf = entry['model_performance']
        row.update({
            'accuracy': perf.get('accuracy'),
            'precision': perf.get('precision'),
            'recall': perf.get('recall'),
            'f1': perf.get('f1'),
        })

        model_perf_data.append(row)

df_model_perf = pd.DataFrame(model_perf_data)
# Remove duplicates (same model-dataset combination appears multiple times)
df_model_perf = df_model_perf.drop_duplicates(subset=['dataset', 'model'])
df_model_perf.to_csv(output_dir / 'model_performance.csv', index=False)
print(f"   Saved: {len(df_model_perf)} rows, {len(df_model_perf.columns)} columns")

# ============================================================================
# 11. CORRELATION MATRIX DATA
# ============================================================================
print("\n11. Creating correlation_data.csv...")

# Select only numeric columns for correlation
numeric_cols = df_comprehensive.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df_comprehensive[numeric_cols]

# Save numeric data for correlation analysis
df_numeric.to_csv(output_dir / 'correlation_data.csv', index=False)
print(f"   Saved: {len(df_numeric)} rows, {len(df_numeric.columns)} columns")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXPORT COMPLETE")
print("="*80)
print(f"""
Files created in: {output_dir}

1. comprehensive_results.csv      - All data with all metrics
2. basic_metrics.csv              - Only basic evaluation metrics
3. advanced_metrics.csv           - Only advanced evaluation metrics
4. text_metrics.csv               - Text-specific metrics
5. aggregated_by_dataset.csv      - Aggregated statistics by dataset
6. aggregated_by_model.csv        - Aggregated statistics by model
7. aggregated_by_method.csv       - Aggregated statistics by explanation method
8. aggregated_by_modality.csv     - Aggregated statistics by modality (tabular/text/image)
9. best_methods_per_metric.csv    - Best performing methods for each metric
10. model_performance.csv         - Model accuracy, precision, recall, F1
11. correlation_data.csv          - Numeric data for correlation analysis

Total entries processed: {len(results)}
""")

print("\nColumn counts:")
print(f"  Comprehensive: {len(df_comprehensive.columns)} columns")
print(f"  Basic metrics: {len(df_basic.columns)} columns")
print(f"  Advanced metrics: {len(df_advanced.columns)} columns")
print(f"  Text metrics: {len(df_text.columns)} columns")
