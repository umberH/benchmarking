import pandas as pd
import ast
import numpy as np

# Read the CSV
df = pd.read_csv('saved_models/all_models_info.csv')

# Clean up display names
df['dataset_display'] = df['dataset'].str.replace('_', ' ').str.title()
df['model_display'] = df['model'].str.replace('_', ' ').str.title()

# Filter valid models
df_valid = df[df['test_accuracy'].notna()].copy()

print(f"Total models with valid data: {len(df_valid)}")
print(f"Datasets: {df_valid['dataset'].nunique()}")
print(f"Model types: {df_valid['model'].nunique()}")

print("\n" + "="*100)
print("COMPREHENSIVE HYPERPARAMETERS TABLE - ALL MODELS")
print("="*100)

# Create comprehensive table with all models
all_models_data = []

for idx, row in df_valid.iterrows():
    # Parse hyperparameters
    hp = {}
    if pd.notna(row['hyperparameters']):
        try:
            hp = ast.literal_eval(row['hyperparameters'])
        except:
            pass

    # Extract key hyperparameters based on model type
    model_type = row['model']

    # Create a comprehensive hyperparameter string
    hp_details = []

    # Common tree-based parameters
    if 'max_depth' in hp:
        hp_details.append(f"max_depth={hp['max_depth']}")
    if 'criterion' in hp:
        hp_details.append(f"criterion={hp['criterion']}")
    if 'min_samples_split' in hp:
        hp_details.append(f"min_split={hp['min_samples_split']}")
    if 'min_samples_leaf' in hp:
        hp_details.append(f"min_leaf={hp['min_samples_leaf']}")
    if 'splitter' in hp:
        hp_details.append(f"splitter={hp['splitter']}")

    # Ensemble parameters
    if 'n_estimators' in hp:
        hp_details.append(f"n_estimators={hp['n_estimators']}")
    if 'max_features' in hp:
        hp_details.append(f"max_features={hp['max_features']}")
    if 'bootstrap' in hp:
        hp_details.append(f"bootstrap={hp['bootstrap']}")

    # Gradient boosting specific
    if 'learning_rate' in hp:
        hp_details.append(f"lr={hp['learning_rate']}")
    if 'subsample' in hp:
        hp_details.append(f"subsample={hp['subsample']}")

    # Neural network parameters
    if 'hidden_layer_sizes' in hp:
        hp_details.append(f"hidden={hp['hidden_layer_sizes']}")
    if 'activation' in hp:
        hp_details.append(f"activation={hp['activation']}")
    if 'alpha' in hp:
        hp_details.append(f"alpha={hp['alpha']}")
    if 'max_iter' in hp:
        hp_details.append(f"max_iter={hp['max_iter']}")

    # Deep learning parameters
    if 'batch_size' in hp:
        hp_details.append(f"batch={hp['batch_size']}")
    if 'epochs' in hp:
        hp_details.append(f"epochs={hp['epochs']}")
    if 'optimizer' in hp:
        hp_details.append(f"optimizer={hp['optimizer']}")

    hp_string = ', '.join(hp_details) if hp_details else 'default'

    all_models_data.append({
        'dataset': row['dataset_display'],
        'model': row['model_display'],
        'model_id': row['model_id'],
        'test_acc': row['test_accuracy'],
        'test_f1': row['test_f1'],
        'train_acc': row['train_accuracy'],
        'n_params': row['n_parameters'],
        'time': row['training_time'],
        'hyperparams': hp_string,
        'hp_dict': hp
    })

# Create DataFrame
df_all = pd.DataFrame(all_models_data)

# Sort by dataset and test accuracy
df_all = df_all.sort_values(['dataset', 'test_acc'], ascending=[True, False])

print("\n\\begin{table}[h]")
print("\\centering")
print("\\footnotesize")
print("\\caption{Complete Model Inventory with Hyperparameters}")
print("\\label{tab:all_models_hyperparameters}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{llcccp{6cm}}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Test Acc.} & \\textbf{Params} & \\textbf{Time (s)} & \\textbf{Hyperparameters} \\\\")
print("\\midrule")

current_dataset = None
for _, row in df_all.iterrows():
    if current_dataset != row['dataset']:
        if current_dataset is not None:
            print("\\midrule")
        current_dataset = row['dataset']

    # Format hyperparameters for LaTeX (escape underscores)
    hp_latex = row['hyperparams'].replace('_', '\\_')

    print(f"{row['dataset']:<25} & {row['model']:<20} & {row['test_acc']:.4f} & {int(row['n_params']):>10,} & {row['time']:>8.2f} & {hp_latex} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\end{table}")

print("\n\n" + "="*100)
print("DETAILED HYPERPARAMETER BREAKDOWN BY MODEL TYPE")
print("="*100)

# Group by model type and create separate detailed tables
model_types = df_all['model'].unique()

for model_type in sorted(model_types):
    subset = df_all[df_all['model'] == model_type].copy()

    if len(subset) == 0:
        continue

    print(f"\n{'='*100}")
    print(f"{model_type.upper()} - {len(subset)} instances")
    print('='*100)

    # Get all unique hyperparameter keys for this model type
    all_hp_keys = set()
    for hp_dict in subset['hp_dict']:
        if hp_dict:
            all_hp_keys.update(hp_dict.keys())

    if not all_hp_keys:
        print(f"No hyperparameters stored for {model_type}")
        continue

    # Create table header
    hp_keys_sorted = sorted(list(all_hp_keys))

    print("\\n\\\\begin{table}[h]")
    print("\\\\centering")
    print("\\\\caption{" + model_type + " Models - Detailed Hyperparameters}")
    print("\\\\label{tab:" + model_type.lower().replace(' ', '_') + "_hyperparams_all}")
    print("\\resizebox{\\textwidth}{!}{")

    # Create dynamic column definition
    num_cols = min(len(hp_keys_sorted), 8)  # Limit to 8 hyperparameter columns
    col_def = "ll" + "c" * num_cols
    print(f"\\begin{{tabular}}{{{col_def}}}")
    print("\\toprule")

    # Create header
    header = "\\textbf{Dataset} & \\textbf{Test Acc.}"
    for key in hp_keys_sorted[:num_cols]:
        header += f" & \\textbf{{{key.replace('_', ' ').title()}}}"
    header += " \\\\"
    print(header)
    print("\\midrule")

    # Print rows
    for _, row in subset.iterrows():
        hp_dict = row['hp_dict']
        line = f"{row['dataset']:<25} & {row['test_acc']:.4f}"

        for key in hp_keys_sorted[:num_cols]:
            val = hp_dict.get(key, '-')
            if isinstance(val, float):
                line += f" & {val:.4f}"
            else:
                line += f" & {val}"

        line += " \\\\"
        print(line)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\end{table}")

print("\n\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)

print(f"\nTotal models analyzed: {len(df_all)}")
print(f"\nModels by type:")
for model_type in sorted(df_all['model'].value_counts().index):
    count = len(df_all[df_all['model'] == model_type])
    with_hp = len(df_all[(df_all['model'] == model_type) & (df_all['hyperparams'] != 'default')])
    print(f"  {model_type}: {count} total, {with_hp} with stored hyperparameters")

print(f"\nModels by dataset:")
for dataset in sorted(df_all['dataset'].value_counts().index):
    count = len(df_all[df_all['dataset'] == dataset])
    print(f"  {dataset}: {count} models")

# Save comprehensive CSV
output_file = 'saved_models/all_models_with_hyperparameters_detailed.csv'
df_all.to_csv(output_file, index=False)
print(f"\n\nDetailed data saved to: {output_file}")

print("\n" + "="*100)
print("NOTES FOR YOUR PAPER")
print("="*100)
print("""
1. Main comprehensive table shows ALL models with their hyperparameters
2. Individual model-type tables provide detailed hyperparameter breakdowns
3. Models with 'default' have no stored hyperparameters (using framework defaults)
4. Model IDs are shown for reproducibility and traceability
5. Use the detailed CSV for further analysis or custom table generation
""")
