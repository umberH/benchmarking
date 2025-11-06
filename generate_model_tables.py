import pandas as pd
import numpy as np

# Read the CSV
df = pd.read_csv('saved_models/all_models_info.csv')

print(f"Total models: {len(df)}")
print(f"\nDatasets: {df['dataset'].nunique()}")
print(f"Model types: {df['model'].nunique()}")

# Clean up model names for display
df['dataset_display'] = df['dataset'].str.replace('_', ' ').str.title()
df['model_display'] = df['model'].str.replace('_', ' ').str.title()

# Filter out models with missing metrics
df_valid = df[df['test_accuracy'].notna()].copy()

print("\n" + "="*100)
print("TABLE 1: COMPLETE MODEL PERFORMANCE SUMMARY (All Models)")
print("="*100)

# Group by dataset and model, take the best performing one
best_models = df_valid.loc[df_valid.groupby(['dataset', 'model'])['test_accuracy'].idxmax()]
best_models = best_models.sort_values(['dataset', 'test_accuracy'], ascending=[True, False])

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Model Performance Summary Across All Datasets}")
print("\\label{tab:all_models_performance}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{llcccccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Train Acc.} & \\textbf{Test Acc.} & \\textbf{Train F1} & \\textbf{Test F1} & \\textbf{Params} & \\textbf{Time (s)} \\\\")
print("\\midrule")

current_dataset = None
for _, row in best_models.iterrows():
    if current_dataset != row['dataset']:
        if current_dataset is not None:
            print("\\midrule")
        current_dataset = row['dataset']

    train_acc = row['train_accuracy'] if pd.notna(row['train_accuracy']) else 0
    test_acc = row['test_accuracy'] if pd.notna(row['test_accuracy']) else 0
    train_f1 = row['train_f1'] if pd.notna(row['train_f1']) else 0
    test_f1 = row['test_f1'] if pd.notna(row['test_f1']) else 0
    n_params = int(row['n_parameters']) if pd.notna(row['n_parameters']) else 0
    time = row['training_time'] if pd.notna(row['training_time']) else 0

    print(f"{row['dataset_display']:<25} & {row['model_display']:<20} & {train_acc:.4f} & {test_acc:.4f} & {train_f1:.4f} & {test_f1:.4f} & {n_params:>10,} & {time:>8.2f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 2: AVERAGE PERFORMANCE BY MODEL TYPE")
print("="*100)

# Calculate average performance by model type across all datasets
model_stats = best_models.groupby('model_display').agg({
    'train_accuracy': ['mean', 'std'],
    'test_accuracy': ['mean', 'std'],
    'train_f1': ['mean', 'std'],
    'test_f1': ['mean', 'std'],
    'training_time': ['mean', 'std'],
    'n_parameters': 'mean'
}).round(4)

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Average Performance by Model Type Across All Datasets}")
print("\\label{tab:model_type_performance}")
print("\\begin{tabular}{lccccc}")
print("\\toprule")
print("\\textbf{Model} & \\textbf{Avg Train Acc.} & \\textbf{Avg Test Acc.} & \\textbf{Avg Test F1} & \\textbf{Avg Params} & \\textbf{Avg Time (s)} \\\\")
print("\\midrule")

# Sort by test accuracy
model_stats_sorted = model_stats.sort_values(('test_accuracy', 'mean'), ascending=False)

for model in model_stats_sorted.index:
    train_mean = model_stats.loc[model, ('train_accuracy', 'mean')]
    test_mean = model_stats.loc[model, ('test_accuracy', 'mean')]
    test_std = model_stats.loc[model, ('test_accuracy', 'std')]
    f1_mean = model_stats.loc[model, ('test_f1', 'mean')]
    params_mean = model_stats.loc[model, ('n_parameters', 'mean')]
    time_mean = model_stats.loc[model, ('training_time', 'mean')]

    print(f"{model:<25} & {train_mean:.4f} & {test_mean:.4f} $\\pm$ {test_std:.4f} & {f1_mean:.4f} & {params_mean:>10,.0f} & {time_mean:>8.2f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 3: AVERAGE PERFORMANCE BY DATASET")
print("="*100)

# Calculate average performance by dataset across all models
dataset_stats = best_models.groupby('dataset_display').agg({
    'train_accuracy': ['mean', 'std'],
    'test_accuracy': ['mean', 'std', 'min', 'max'],
    'test_f1': ['mean', 'std'],
    'training_time': ['mean']
}).round(4)

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Average Performance by Dataset Across All Models}")
print("\\label{tab:dataset_performance}")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Avg Test Acc.} & \\textbf{Min-Max} & \\textbf{Avg Test F1} & \\textbf{Avg Time (s)} \\\\")
print("\\midrule")

# Sort by test accuracy
dataset_stats_sorted = dataset_stats.sort_values(('test_accuracy', 'mean'), ascending=False)

for dataset in dataset_stats_sorted.index:
    test_mean = dataset_stats.loc[dataset, ('test_accuracy', 'mean')]
    test_std = dataset_stats.loc[dataset, ('test_accuracy', 'std')]
    test_min = dataset_stats.loc[dataset, ('test_accuracy', 'min')]
    test_max = dataset_stats.loc[dataset, ('test_accuracy', 'max')]
    f1_mean = dataset_stats.loc[dataset, ('test_f1', 'mean')]
    time_mean = dataset_stats.loc[dataset, ('training_time', 'mean')]

    print(f"{dataset:<25} & {test_mean:.4f} $\\pm$ {test_std:.4f} & [{test_min:.4f}, {test_max:.4f}] & {f1_mean:.4f} & {time_mean:>8.2f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 4: COMPACT BEST MODEL PER DATASET")
print("="*100)

# Best model per dataset
best_per_dataset = df_valid.loc[df_valid.groupby('dataset')['test_accuracy'].idxmax()]
best_per_dataset = best_per_dataset.sort_values('test_accuracy', ascending=False)

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Best Performing Model for Each Dataset}")
print("\\label{tab:best_per_dataset}")
print("\\begin{tabular}{llccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Best Model} & \\textbf{Test Acc.} & \\textbf{Test F1} & \\textbf{Training Time (s)} \\\\")
print("\\midrule")

for _, row in best_per_dataset.iterrows():
    test_acc = row['test_accuracy'] if pd.notna(row['test_accuracy']) else 0
    test_f1 = row['test_f1'] if pd.notna(row['test_f1']) else 0
    time = row['training_time'] if pd.notna(row['training_time']) else 0

    print(f"{row['dataset_display']:<25} & {row['model_display']:<20} & {test_acc:.4f} & {test_f1:.4f} & {time:>8.2f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 5: MODELS BY COMPLEXITY AND PERFORMANCE")
print("="*100)

# Group models by complexity level
df_valid['complexity_category'] = pd.cut(
    df_valid['n_parameters'],
    bins=[0, 100, 10000, 1000000, float('inf')],
    labels=['Simple (<100)', 'Medium (100-10K)', 'Large (10K-1M)', 'Very Large (>1M)']
)

complexity_stats = df_valid.groupby('complexity_category').agg({
    'test_accuracy': ['mean', 'std', 'count'],
    'test_f1': ['mean'],
    'training_time': ['mean'],
    'n_parameters': ['mean', 'min', 'max']
}).round(4)

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Model Performance by Complexity Level}")
print("\\label{tab:complexity_performance}")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("\\textbf{Complexity} & \\textbf{Count} & \\textbf{Avg Test Acc.} & \\textbf{Avg Test F1} & \\textbf{Avg Time (s)} \\\\")
print("\\midrule")

for complexity in complexity_stats.index:
    if pd.notna(complexity):
        count = int(complexity_stats.loc[complexity, ('test_accuracy', 'count')])
        test_mean = complexity_stats.loc[complexity, ('test_accuracy', 'mean')]
        test_std = complexity_stats.loc[complexity, ('test_accuracy', 'std')]
        f1_mean = complexity_stats.loc[complexity, ('test_f1', 'mean')]
        time_mean = complexity_stats.loc[complexity, ('training_time', 'mean')]

        print(f"{complexity:<25} & {count:>5} & {test_mean:.4f} $\\pm$ {test_std:.4f} & {f1_mean:.4f} & {time_mean:>8.2f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n\nAll LaTeX tables generated successfully!")
print(f"\nSource data: saved_models/all_models_info.csv ({len(df_valid)} valid models)")
