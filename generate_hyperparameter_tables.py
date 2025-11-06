import pandas as pd
import ast
import json

# Read the CSV
df = pd.read_csv('saved_models/all_models_info.csv')

# Clean up display names
df['dataset_display'] = df['dataset'].str.replace('_', ' ').str.title()
df['model_display'] = df['model'].str.replace('_', ' ').str.title()

# Filter valid models
df_valid = df[df['test_accuracy'].notna()].copy()

print("="*100)
print("TABLE 1: DECISION TREE HYPERPARAMETERS")
print("="*100)

dt_models = df_valid[df_valid['model'] == 'decision_tree'].copy()
dt_best = dt_models.loc[dt_models.groupby('dataset')['test_accuracy'].idxmax()]
dt_best = dt_best.sort_values('test_accuracy', ascending=False)

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Decision Tree Hyperparameters Across Datasets}")
print("\\label{tab:decision_tree_hyperparams}")
print("\\begin{tabular}{lccccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Test Acc.} & \\textbf{Criterion} & \\textbf{Max Depth} & \\textbf{Min Split} & \\textbf{Min Leaf} \\\\")
print("\\midrule")

for _, row in dt_best.iterrows():
    hp = {}
    if pd.notna(row['hyperparameters']):
        try:
            hp = ast.literal_eval(row['hyperparameters'])
        except:
            pass

    criterion = hp.get('criterion', 'N/A')
    max_depth = hp.get('max_depth', 'N/A')
    min_split = hp.get('min_samples_split', 'N/A')
    min_leaf = hp.get('min_samples_leaf', 'N/A')

    print(f"{row['dataset_display']:<25} & {row['test_accuracy']:.4f} & {criterion:<8} & {max_depth!s:<10} & {min_split!s:<10} & {min_leaf!s:<8} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 2: RANDOM FOREST HYPERPARAMETERS")
print("="*100)

rf_models = df_valid[df_valid['model'] == 'random_forest'].copy()
rf_best = rf_models.loc[rf_models.groupby('dataset')['test_accuracy'].idxmax()]
rf_best = rf_best.sort_values('test_accuracy', ascending=False)

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Random Forest Hyperparameters Across Datasets}")
print("\\label{tab:random_forest_hyperparams}")
print("\\begin{tabular}{lcccccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Test Acc.} & \\textbf{N Est.} & \\textbf{Max Depth} & \\textbf{Max Features} & \\textbf{Min Split} & \\textbf{Min Leaf} \\\\")
print("\\midrule")

for _, row in rf_best.iterrows():
    hp = {}
    if pd.notna(row['hyperparameters']):
        try:
            hp = ast.literal_eval(row['hyperparameters'])
        except:
            pass

    n_est = hp.get('n_estimators', 'N/A')
    max_depth = hp.get('max_depth', 'N/A')
    max_features = hp.get('max_features', 'N/A')
    min_split = hp.get('min_samples_split', 'N/A')
    min_leaf = hp.get('min_samples_leaf', 'N/A')

    print(f"{row['dataset_display']:<25} & {row['test_accuracy']:.4f} & {n_est!s:<8} & {max_depth!s:<10} & {max_features!s:<12} & {min_split!s:<10} & {min_leaf!s:<8} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 3: GRADIENT BOOSTING HYPERPARAMETERS")
print("="*100)

gb_models = df_valid[df_valid['model'] == 'gradient_boosting'].copy()
gb_best = gb_models.loc[gb_models.groupby('dataset')['test_accuracy'].idxmax()]
gb_best = gb_best.sort_values('test_accuracy', ascending=False)

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Gradient Boosting Hyperparameters Across Datasets}")
print("\\label{tab:gradient_boosting_hyperparams}")
print("\\begin{tabular}{lcccccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Test Acc.} & \\textbf{N Est.} & \\textbf{LR} & \\textbf{Max Depth} & \\textbf{Min Split} & \\textbf{Subsample} \\\\")
print("\\midrule")

for _, row in gb_best.iterrows():
    hp = {}
    if pd.notna(row['hyperparameters']):
        try:
            hp = ast.literal_eval(row['hyperparameters'])
        except:
            pass

    n_est = hp.get('n_estimators', 'N/A')
    lr = hp.get('learning_rate', 'N/A')
    max_depth = hp.get('max_depth', 'N/A')
    min_split = hp.get('min_samples_split', 'N/A')
    subsample = hp.get('subsample', 'N/A')

    print(f"{row['dataset_display']:<25} & {row['test_accuracy']:.4f} & {n_est!s:<8} & {lr!s:<6} & {max_depth!s:<10} & {min_split!s:<10} & {subsample!s:<10} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 4: MLP HYPERPARAMETERS")
print("="*100)

mlp_models = df_valid[df_valid['model'] == 'mlp'].copy()
mlp_best = mlp_models.loc[mlp_models.groupby('dataset')['test_accuracy'].idxmax()]
mlp_best = mlp_best.sort_values('test_accuracy', ascending=False)

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{MLP Hyperparameters Across Datasets}")
print("\\label{tab:mlp_hyperparams}")
print("\\begin{tabular}{lccccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Test Acc.} & \\textbf{Hidden Layers} & \\textbf{Activation} & \\textbf{Alpha} & \\textbf{Max Iter} \\\\")
print("\\midrule")

for _, row in mlp_best.iterrows():
    hp = {}
    if pd.notna(row['hyperparameters']):
        try:
            hp = ast.literal_eval(row['hyperparameters'])
        except:
            pass

    hidden = hp.get('hidden_layer_sizes', 'N/A')
    activation = hp.get('activation', 'N/A')
    alpha = hp.get('alpha', 'N/A')
    max_iter = hp.get('max_iter', 'N/A')

    print(f"{row['dataset_display']:<25} & {row['test_accuracy']:.4f} & {hidden!s:<20} & {activation!s:<10} & {alpha!s:<8} & {max_iter!s:<8} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 5: COMBINED HYPERPARAMETERS (COMPACT)")
print("="*100)

# Get best tabular models
tabular_datasets = ['adult_income', 'breast_cancer', 'compas', 'diabetes', 'german_credit', 'heart_disease']
tabular_models_list = ['decision_tree', 'random_forest', 'gradient_boosting', 'mlp']

best_tabular = []
for dataset in tabular_datasets:
    for model in tabular_models_list:
        subset = df_valid[(df_valid['dataset'] == dataset) & (df_valid['model'] == model)]
        if not subset.empty:
            best = subset.loc[subset['test_accuracy'].idxmax()]

            hp = {}
            if pd.notna(best['hyperparameters']):
                try:
                    hp = ast.literal_eval(best['hyperparameters'])
                except:
                    pass

            # Create compact hyperparameter string
            hp_str = []
            if model == 'decision_tree':
                if 'max_depth' in hp:
                    hp_str.append(f"d={hp['max_depth']}")
                if 'criterion' in hp:
                    hp_str.append(f"c={hp['criterion'][:3]}")
            elif model == 'random_forest':
                if 'n_estimators' in hp:
                    hp_str.append(f"n={hp['n_estimators']}")
                if 'max_depth' in hp:
                    hp_str.append(f"d={hp['max_depth']}")
            elif model == 'gradient_boosting':
                if 'n_estimators' in hp:
                    hp_str.append(f"n={hp['n_estimators']}")
                if 'learning_rate' in hp:
                    hp_str.append(f"lr={hp['learning_rate']}")
                if 'max_depth' in hp:
                    hp_str.append(f"d={hp['max_depth']}")
            elif model == 'mlp':
                if 'hidden_layer_sizes' in hp:
                    hp_str.append(f"h={hp['hidden_layer_sizes']}")
                if 'activation' in hp:
                    hp_str.append(f"a={hp['activation'][:4]}")

            best_tabular.append({
                'dataset': best['dataset_display'],
                'model': best['model_display'],
                'test_acc': best['test_accuracy'],
                'hp_str': ', '.join(hp_str) if hp_str else 'default'
            })

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Best Model Hyperparameters for Tabular Datasets (Compact)}")
print("\\label{tab:hyperparameters_compact}")
print("\\begin{tabular}{llcl}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Test Acc.} & \\textbf{Key Hyperparameters} \\\\")
print("\\midrule")

current_dataset = None
for row in best_tabular:
    if current_dataset != row['dataset']:
        if current_dataset is not None:
            print("\\midrule")
        current_dataset = row['dataset']

    print(f"{row['dataset']:<20} & {row['model']:<20} & {row['test_acc']:.4f} & {row['hp_str']} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n\nAll hyperparameter tables generated successfully!")
