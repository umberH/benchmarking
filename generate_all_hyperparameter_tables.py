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

# Categorize datasets
binary_tabular = ['adult_income', 'breast_cancer', 'compas', 'german_credit', 'heart_disease']
multiclass_tabular = ['iris', 'wine_quality', 'diabetes', 'wine_classification', 'digits']
image_datasets = ['mnist', 'cifar10', 'fashion_mnist']
text_datasets = ['imdb', '20newsgroups', 'ag_news']

print("="*100)
print("TABLE 1: BINARY CLASSIFICATION - TABULAR DATASETS HYPERPARAMETERS")
print("="*100)

binary_data = []
for dataset in binary_tabular:
    for model in ['decision_tree', 'random_forest', 'gradient_boosting', 'mlp']:
        subset = df_valid[(df_valid['dataset'] == dataset) & (df_valid['model'] == model)]
        if not subset.empty:
            best = subset.loc[subset['test_accuracy'].idxmax()]
            hp = {}
            if pd.notna(best['hyperparameters']):
                try:
                    hp = ast.literal_eval(best['hyperparameters'])
                except:
                    pass
            binary_data.append({
                'dataset': best['dataset_display'],
                'model': best['model_display'],
                'test_acc': best['test_accuracy'],
                'hp': hp
            })

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Hyperparameters for Binary Classification Datasets}")
print("\\label{tab:binary_hyperparams}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{llcccccccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Test Acc.} & \\textbf{Max Depth} & \\textbf{N Est.} & \\textbf{LR} & \\textbf{Hidden} & \\textbf{Activation} & \\textbf{Criterion} \\\\")
print("\\midrule")

current_dataset = None
for row in binary_data:
    if current_dataset != row['dataset']:
        if current_dataset is not None:
            print("\\midrule")
        current_dataset = row['dataset']

    hp = row['hp']
    max_depth = str(hp.get('max_depth', '-'))
    n_est = str(hp.get('n_estimators', '-'))
    lr = str(hp.get('learning_rate', '-'))
    hidden = str(hp.get('hidden_layer_sizes', '-'))
    activation = str(hp.get('activation', '-'))
    criterion = str(hp.get('criterion', '-'))

    print(f"{row['dataset']:<20} & {row['model']:<20} & {row['test_acc']:.4f} & {max_depth:<10} & {n_est:<8} & {lr:<6} & {hidden:<15} & {activation:<10} & {criterion:<8} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 2: MULTICLASS CLASSIFICATION - TABULAR DATASETS HYPERPARAMETERS")
print("="*100)

multiclass_data = []
for dataset in multiclass_tabular:
    for model in ['decision_tree', 'random_forest', 'gradient_boosting', 'mlp']:
        subset = df_valid[(df_valid['dataset'] == dataset) & (df_valid['model'] == model)]
        if not subset.empty:
            best = subset.loc[subset['test_accuracy'].idxmax()]
            hp = {}
            if pd.notna(best['hyperparameters']):
                try:
                    hp = ast.literal_eval(best['hyperparameters'])
                except:
                    pass
            multiclass_data.append({
                'dataset': best['dataset_display'],
                'model': best['model_display'],
                'test_acc': best['test_accuracy'],
                'hp': hp
            })

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Hyperparameters for Multiclass Classification Datasets}")
print("\\label{tab:multiclass_hyperparams}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{llcccccccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Test Acc.} & \\textbf{Max Depth} & \\textbf{N Est.} & \\textbf{LR} & \\textbf{Hidden} & \\textbf{Activation} & \\textbf{Criterion} \\\\")
print("\\midrule")

current_dataset = None
for row in multiclass_data:
    if current_dataset != row['dataset']:
        if current_dataset is not None:
            print("\\midrule")
        current_dataset = row['dataset']

    hp = row['hp']
    max_depth = str(hp.get('max_depth', '-'))
    n_est = str(hp.get('n_estimators', '-'))
    lr = str(hp.get('learning_rate', '-'))
    hidden = str(hp.get('hidden_layer_sizes', '-'))
    activation = str(hp.get('activation', '-'))
    criterion = str(hp.get('criterion', '-'))

    print(f"{row['dataset']:<20} & {row['model']:<20} & {row['test_acc']:.4f} & {max_depth:<10} & {n_est:<8} & {lr:<6} & {hidden:<15} & {activation:<10} & {criterion:<8} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 3: IMAGE DATASETS HYPERPARAMETERS")
print("="*100)

image_data = []
for dataset in image_datasets:
    for model in ['cnn', 'vit', 'resnet']:
        subset = df_valid[(df_valid['dataset'] == dataset) & (df_valid['model'] == model)]
        if not subset.empty:
            best = subset.loc[subset['test_accuracy'].idxmax()]
            hp = {}
            if pd.notna(best['hyperparameters']):
                try:
                    hp = ast.literal_eval(best['hyperparameters'])
                except:
                    pass
            image_data.append({
                'dataset': best['dataset_display'],
                'model': best['model_display'],
                'test_acc': best['test_accuracy'],
                'train_acc': best['train_accuracy'],
                'n_params': best['n_parameters'],
                'time': best['training_time'],
                'hp': hp
            })

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Hyperparameters and Performance for Image Datasets}")
print("\\label{tab:image_hyperparams}")
print("\\begin{tabular}{llcccccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Train Acc.} & \\textbf{Test Acc.} & \\textbf{Parameters} & \\textbf{Time (s)} & \\textbf{Batch Size} & \\textbf{Epochs} \\\\")
print("\\midrule")

current_dataset = None
for row in image_data:
    if current_dataset != row['dataset']:
        if current_dataset is not None:
            print("\\midrule")
        current_dataset = row['dataset']

    hp = row['hp']
    batch_size = str(hp.get('batch_size', '-'))
    epochs = str(hp.get('epochs', '-'))

    print(f"{row['dataset']:<20} & {row['model']:<20} & {row['train_acc']:.4f} & {row['test_acc']:.4f} & {int(row['n_params']):>12,} & {row['time']:>8.2f} & {batch_size:<10} & {epochs:<8} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 4: TEXT DATASETS HYPERPARAMETERS")
print("="*100)

text_data = []
for dataset in text_datasets:
    for model in ['bert', 'lstm', 'roberta', 'naive_bayes_text', 'svm_text', 'xgboost_text']:
        subset = df_valid[(df_valid['dataset'] == dataset) & (df_valid['model'] == model)]
        if not subset.empty:
            best = subset.loc[subset['test_accuracy'].idxmax()]
            hp = {}
            if pd.notna(best['hyperparameters']):
                try:
                    hp = ast.literal_eval(best['hyperparameters'])
                except:
                    pass
            text_data.append({
                'dataset': best['dataset_display'],
                'model': best['model_display'],
                'test_acc': best['test_accuracy'],
                'train_acc': best['train_accuracy'],
                'n_params': best['n_parameters'],
                'time': best['training_time'],
                'hp': hp
            })

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Hyperparameters and Performance for Text Datasets}")
print("\\label{tab:text_hyperparams}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{llcccccc}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Train Acc.} & \\textbf{Test Acc.} & \\textbf{Parameters} & \\textbf{Time (s)} & \\textbf{Batch Size} & \\textbf{Max Length} \\\\")
print("\\midrule")

current_dataset = None
for row in text_data:
    if current_dataset != row['dataset']:
        if current_dataset is not None:
            print("\\midrule")
        current_dataset = row['dataset']

    hp = row['hp']
    batch_size = str(hp.get('batch_size', '-'))
    max_length = str(hp.get('max_length', '-'))

    print(f"{row['dataset']:<20} & {row['model']:<20} & {row['train_acc']:.4f} & {row['test_acc']:.4f} & {int(row['n_params']):>12,} & {row['time']:>8.2f} & {batch_size:<10} & {max_length:<10} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\end{table}")

print("\n\n" + "="*100)
print("TABLE 5: COMPREHENSIVE SUMMARY - BEST HYPERPARAMETERS BY MODALITY")
print("="*100)

print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Best Model and Hyperparameters per Dataset (All Modalities)}")
print("\\label{tab:comprehensive_hyperparams}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{lllcl}")
print("\\toprule")
print("\\textbf{Modality} & \\textbf{Dataset} & \\textbf{Best Model} & \\textbf{Test Acc.} & \\textbf{Key Hyperparameters} \\\\")
print("\\midrule")

# Binary Tabular
print("\\multirow{5}{*}{Binary Tabular}")
for dataset in binary_tabular:
    subset = df_valid[df_valid['dataset'] == dataset]
    if not subset.empty:
        best = subset.loc[subset['test_accuracy'].idxmax()]
        hp = {}
        if pd.notna(best['hyperparameters']):
            try:
                hp = ast.literal_eval(best['hyperparameters'])
            except:
                pass

        # Create compact hyperparameter string
        hp_parts = []
        if 'max_depth' in hp:
            hp_parts.append(f"depth={hp['max_depth']}")
        if 'n_estimators' in hp:
            hp_parts.append(f"n\\_est={hp['n_estimators']}")
        if 'learning_rate' in hp:
            hp_parts.append(f"lr={hp['learning_rate']}")
        if 'hidden_layer_sizes' in hp:
            hp_parts.append(f"hidden={hp['hidden_layer_sizes']}")

        hp_str = ', '.join(hp_parts[:3]) if hp_parts else 'default'

        print(f" & {best['dataset_display']:<20} & {best['model_display']:<20} & {best['test_accuracy']:.4f} & {hp_str} \\\\")

print("\\midrule")
print("\\multirow{5}{*}{Multiclass Tabular}")
for dataset in multiclass_tabular:
    subset = df_valid[df_valid['dataset'] == dataset]
    if not subset.empty:
        best = subset.loc[subset['test_accuracy'].idxmax()]
        hp = {}
        if pd.notna(best['hyperparameters']):
            try:
                hp = ast.literal_eval(best['hyperparameters'])
            except:
                pass

        hp_parts = []
        if 'max_depth' in hp:
            hp_parts.append(f"depth={hp['max_depth']}")
        if 'n_estimators' in hp:
            hp_parts.append(f"n\\_est={hp['n_estimators']}")
        if 'learning_rate' in hp:
            hp_parts.append(f"lr={hp['learning_rate']}")
        if 'hidden_layer_sizes' in hp:
            hp_parts.append(f"hidden={hp['hidden_layer_sizes']}")

        hp_str = ', '.join(hp_parts[:3]) if hp_parts else 'default'

        print(f" & {best['dataset_display']:<20} & {best['model_display']:<20} & {best['test_accuracy']:.4f} & {hp_str} \\\\")

print("\\midrule")
print("\\multirow{3}{*}{Image}")
for dataset in image_datasets:
    subset = df_valid[df_valid['dataset'] == dataset]
    if not subset.empty:
        best = subset.loc[subset['test_accuracy'].idxmax()]
        hp = {}
        if pd.notna(best['hyperparameters']):
            try:
                hp = ast.literal_eval(best['hyperparameters'])
            except:
                pass

        hp_parts = []
        if 'batch_size' in hp:
            hp_parts.append(f"batch={hp['batch_size']}")
        if 'epochs' in hp:
            hp_parts.append(f"epochs={hp['epochs']}")
        if 'learning_rate' in hp:
            hp_parts.append(f"lr={hp['learning_rate']}")

        hp_str = ', '.join(hp_parts) if hp_parts else 'default'

        print(f" & {best['dataset_display']:<20} & {best['model_display']:<20} & {best['test_accuracy']:.4f} & {hp_str} \\\\")

print("\\midrule")
print("\\multirow{3}{*}{Text}")
for dataset in text_datasets:
    subset = df_valid[df_valid['dataset'] == dataset]
    if not subset.empty:
        best = subset.loc[subset['test_accuracy'].idxmax()]
        hp = {}
        if pd.notna(best['hyperparameters']):
            try:
                hp = ast.literal_eval(best['hyperparameters'])
            except:
                pass

        hp_parts = []
        if 'batch_size' in hp:
            hp_parts.append(f"batch={hp['batch_size']}")
        if 'max_length' in hp:
            hp_parts.append(f"max\\_len={hp['max_length']}")
        if 'learning_rate' in hp:
            hp_parts.append(f"lr={hp['learning_rate']}")

        hp_str = ', '.join(hp_parts) if hp_parts else 'default'

        print(f" & {best['dataset_display']:<20} & {best['model_display']:<20} & {best['test_accuracy']:.4f} & {hp_str} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\end{table}")

print("\n\nAll comprehensive hyperparameter tables generated successfully!")
print(f"Total datasets covered: {len(binary_tabular) + len(multiclass_tabular) + len(image_datasets) + len(text_datasets)}")
print(f"  - Binary Tabular: {len(binary_tabular)}")
print(f"  - Multiclass Tabular: {len(multiclass_tabular)}")
print(f"  - Image: {len(image_datasets)}")
print(f"  - Text: {len(text_datasets)}")
