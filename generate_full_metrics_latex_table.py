import pandas as pd
import numpy as np

# Load the comprehensive results
df = pd.read_csv('results/experiment_20251020_163046/csv_exports/comprehensive_results.csv')

print("Generating LaTeX table with 7 basic metrics (excluding time_complexity)...")

# Define the 7 basic metrics (excluding time_complexity)
metrics = {
    'faithfulness': 'Faithfulness',
    'stability': 'Stability',
    'monotonicity': 'Monotonicity',
    'completeness': 'Completeness',
    'simplicity': 'Simplicity',
    'sparsity': 'Sparsity',
    'consistency': 'Consistency'
}

# Group by modality, explanation method and calculate mean for each metric
grouped = df.groupby(['dataset_type', 'explanation_method'])

results = []
for (modality, method), group in grouped:
    row = {
        'modality': modality,
        'technique': method
    }

    for metric_key in metrics.keys():
        col_name = f'eval_{metric_key}'
        if col_name in df.columns:
            row[metric_key] = group[col_name].mean()
        else:
            row[metric_key] = None

    results.append(row)

df_results = pd.DataFrame(results)

# Sort by modality and technique
df_results = df_results.sort_values(['modality', 'technique'])

# Normalize values to 0-1 range for each metric (for highlighting)
for metric_key in metrics.keys():
    if df_results[metric_key].notna().any():
        min_val = df_results[metric_key].min()
        max_val = df_results[metric_key].max()
        if max_val - min_val > 0:
            df_results[f'{metric_key}_norm'] = (df_results[metric_key] - min_val) / (max_val - min_val)
        else:
            df_results[f'{metric_key}_norm'] = 0

# Map modality names
modality_map = {
    'tabular': 'Tabular',
    'text': 'Text',
    'image': 'Image'
}

# Separate by modality subtype
binary_tabular = df[(df['dataset_type'] == 'tabular') &
                    (df['dataset'].isin(['adult_income', 'breast_cancer', 'compas',
                                        'german_credit', 'heart_disease']))]
multiclass_tabular = df[(df['dataset_type'] == 'tabular') &
                        (df['dataset'].isin(['iris', 'wine_quality', 'diabetes',
                                            'wine_classification', 'digits']))]

# Recalculate with proper grouping
results_detailed = []

# Binary Tabular
for method in sorted(binary_tabular['explanation_method'].unique()):
    group = binary_tabular[binary_tabular['explanation_method'] == method]
    row = {'modality': 'Binary Tabular', 'technique': method}
    for metric_key in metrics.keys():
        col_name = f'eval_{metric_key}'
        row[metric_key] = group[col_name].mean() if col_name in df.columns else None
    results_detailed.append(row)

# Multiclass Tabular
for method in sorted(multiclass_tabular['explanation_method'].unique()):
    group = multiclass_tabular[multiclass_tabular['explanation_method'] == method]
    row = {'modality': 'Multiclass Tabular', 'technique': method}
    for metric_key in metrics.keys():
        col_name = f'eval_{metric_key}'
        row[metric_key] = group[col_name].mean() if col_name in df.columns else None
    results_detailed.append(row)

# Image
image_df = df[df['dataset_type'] == 'image']
for method in sorted(image_df['explanation_method'].unique()):
    group = image_df[image_df['explanation_method'] == method]
    row = {'modality': 'Image', 'technique': method}
    for metric_key in metrics.keys():
        col_name = f'eval_{metric_key}'
        row[metric_key] = group[col_name].mean() if col_name in df.columns else None
    results_detailed.append(row)

# Text
text_df = df[df['dataset_type'] == 'text']
for method in sorted(text_df['explanation_method'].unique()):
    group = text_df[text_df['explanation_method'] == method]
    row = {'modality': 'Text', 'technique': method}
    for metric_key in metrics.keys():
        col_name = f'eval_{metric_key}'
        row[metric_key] = group[col_name].mean() if col_name in df.columns else None
    results_detailed.append(row)

df_detailed = pd.DataFrame(results_detailed)

# Find max value per metric within each modality for highlighting
def find_max_per_modality(df_detailed, metric):
    max_indices = []
    for modality in df_detailed['modality'].unique():
        modality_data = df_detailed[df_detailed['modality'] == modality]
        if modality_data[metric].notna().any():
            max_idx = modality_data[metric].idxmax()
            max_indices.append(max_idx)
    return max_indices

# Generate LaTeX table
print("\n" + "="*100)
print("LATEX TABLE - ALL 8 BASIC METRICS")
print("="*100)

latex_output = []

latex_output.append("\\begin{table}[htbp]")
latex_output.append("\\centering")
latex_output.append("\\scriptsize")
latex_output.append("\\caption{Evaluation metrics across modalities and explanation techniques (maximum per metric highlighted).}")
latex_output.append("\\label{tab:all_metrics_technique}")
latex_output.append("\\resizebox{\\textwidth}{!}{")
latex_output.append("\\begin{tabular}{ll" + "c"*8 + "}")
latex_output.append("\\toprule")

# Header
header = "Modality & Technique"
for metric_name in metrics.values():
    header += f" & {metric_name}"
header += " \\\\"
latex_output.append(header)
latex_output.append("\\midrule")

# Data rows
current_modality = None
for idx, row in df_detailed.iterrows():
    modality = row['modality']
    technique = row['technique'].replace('_', '\\_')

    # Add modality grouping
    if current_modality != modality:
        if current_modality is not None:
            latex_output.append("\\midrule")

        # Count techniques in this modality
        num_techniques = len(df_detailed[df_detailed['modality'] == modality])
        latex_output.append(f"\\multirow{{{num_techniques}}}{{*}}{{{modality}}}")
        current_modality = modality
    else:
        latex_output.append(" ")

    line = f" & {technique}"

    # Add metric values
    for metric_key in metrics.keys():
        value = row[metric_key]

        if pd.isna(value):
            line += " & --"
        else:
            # Check if this is the max value for this metric in this modality
            modality_data = df_detailed[df_detailed['modality'] == modality]
            is_max = False
            if modality_data[metric_key].notna().any():
                max_val = modality_data[metric_key].max()
                if abs(value - max_val) < 0.0001:  # Allow small floating point differences
                    is_max = True

            value_str = f"{value:.3f}"
            if is_max:
                line += f" & \\textbf{{{value_str}}}"
            else:
                line += f" & {value_str}"

    line += " \\\\"
    latex_output.append(line)

latex_output.append("\\bottomrule")
latex_output.append("\\end{tabular}")
latex_output.append("}")
latex_output.append("\\end{table}")

# Print to console
for line in latex_output:
    print(line)

# Save to file
output_file = 'full_metrics_latex_table.tex'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(latex_output))

print(f"\n\nLaTeX table saved to: {output_file}")

# Also create a summary statistics table
print("\n" + "="*100)
print("SUMMARY STATISTICS BY METRIC")
print("="*100)

summary_stats = []
for metric_key, metric_name in metrics.items():
    values = df_detailed[metric_key].dropna()
    if len(values) > 0:
        summary_stats.append({
            'Metric': metric_name,
            'Mean': values.mean(),
            'Std': values.std(),
            'Min': values.min(),
            'Max': values.max(),
            'Non-null Count': len(values)
        })

df_summary = pd.DataFrame(summary_stats)
print(df_summary.to_string(index=False))

# Save summary
df_summary.to_csv('full_metrics_summary_stats.csv', index=False)
print(f"\nSummary statistics saved to: full_metrics_summary_stats.csv")

# Create a separate table for each modality
print("\n" + "="*100)
print("GENERATING SEPARATE TABLES BY MODALITY")
print("="*100)

for modality in df_detailed['modality'].unique():
    modality_data = df_detailed[df_detailed['modality'] == modality]

    print(f"\n{modality}:")
    print("-" * 80)

    latex_modality = []
    latex_modality.append(f"\\begin{{table}}[htbp]")
    latex_modality.append("\\centering")
    latex_modality.append("\\scriptsize")
    latex_modality.append(f"\\caption{{{modality} - Evaluation Metrics}}")
    latex_modality.append(f"\\label{{tab:{modality.lower().replace(' ', '_')}_metrics}}")
    latex_modality.append("\\resizebox{\\textwidth}{!}{")
    latex_modality.append("\\begin{tabular}{l" + "c"*8 + "}")
    latex_modality.append("\\toprule")

    # Header
    header = "Technique"
    for metric_name in metrics.values():
        header += f" & {metric_name}"
    header += " \\\\"
    latex_modality.append(header)
    latex_modality.append("\\midrule")

    # Data rows
    for idx, row in modality_data.iterrows():
        technique = row['technique'].replace('_', '\\_')
        line = technique

        for metric_key in metrics.keys():
            value = row[metric_key]

            if pd.isna(value):
                line += " & --"
            else:
                # Check if max in this modality
                is_max = False
                if modality_data[metric_key].notna().any():
                    max_val = modality_data[metric_key].max()
                    if abs(value - max_val) < 0.0001:
                        is_max = True

                value_str = f"{value:.3f}"
                if is_max:
                    line += f" & \\textbf{{{value_str}}}"
                else:
                    line += f" & {value_str}"

        line += " \\\\"
        latex_modality.append(line)

    latex_modality.append("\\bottomrule")
    latex_modality.append("\\end{tabular}")
    latex_modality.append("}")
    latex_modality.append("\\end{table}")

    # Save individual modality table
    filename = f"metrics_table_{modality.lower().replace(' ', '_')}.tex"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_modality))
    print(f"Saved: {filename}")

print("\n" + "="*100)
print("GENERATION COMPLETE")
print("="*100)
print("""
Files created:
1. full_metrics_latex_table.tex - Complete table with all modalities
2. full_metrics_summary_stats.csv - Summary statistics for all metrics
3. metrics_table_binary_tabular.tex - Binary tabular only
4. metrics_table_multiclass_tabular.tex - Multiclass tabular only
5. metrics_table_image.tex - Image only
6. metrics_table_text.tex - Text only
""")
